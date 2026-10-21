"""에이전트 도구. 수집·계산 함수를 감싸고, 모든 도구가 as_of 시점에 알 수 있던 정보만 돌려준다.

as_of 규칙 (SERVICE.md 섹션 6과 같음)
- as_of 는 'YYYY-MM-DD'(그날 15:15 신호 시점) 또는 생략(지금). 생략하면 오늘(KST)로 본다.
- 일봉·지표·수급·해외 지표: as_of 날짜보다 앞선 날의 값만 (t-1 종가까지, 해외는 현지 날짜 < t).
- 공시: as_of 보다 먼저 접수된 것만. 뉴스: [t-1 15:15, t 15:15) KST.
저장된 데이터(data/raw, data/processed)를 먼저 쓰고, as_of 가 저장 범위를 넘으면 실시간으로 받는다(파일에 쓰지 않음).

요청 상한: /query 는 질문의 시점을 request_as_of() 로 정한다. 그 안에서는 어떤 도구도 그 날짜보다 뒤를 조회하지 않는다.
날짜를 비우면 상한(없으면 오늘 KST)을 쓰고, 상한보다 늦은 날짜는 상한으로 낮추거나 거부하고 그렇다고 알린다.
"""
import contextlib
import contextvars
import functools
import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from src import config
from src.pipeline import collect_dart, collect_macro, collect_news, collect_prices, features

KST = config.KST
_REQUEST_CAP: contextvars.ContextVar[pd.Timestamp | None] = contextvars.ContextVar("request_as_of", default=None)


def resolve_as_of(as_of: str | None) -> pd.Timestamp:
    """as_of 를 한국 날짜(자정, 그날 15:15 신호 시점)로 바꾼다. 없으면 오늘(KST).

    시각이 붙어 있어도 날짜만 남긴다(도구는 'YYYY-MM-DD' 만 받는다).
    """
    if not as_of:
        return pd.Timestamp.now(tz=KST).tz_localize(None).normalize()
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(KST).tz_localize(None)
    return ts.normalize()


@contextlib.contextmanager
def request_as_of(as_of: str | None):
    """요청 하나의 시점 상한을 정한다. 이 안에서 부른 도구는 상한보다 뒤를 조회하지 않는다."""
    token = _REQUEST_CAP.set(resolve_as_of(as_of))
    try:
        yield _REQUEST_CAP.get()
    finally:
        _REQUEST_CAP.reset(token)


def capped(as_of: str | None) -> tuple[pd.Timestamp, str | None]:
    """도구 인자의 날짜를 요청 상한 안으로 맞춘다. (날짜, 낮췄다면 알림)"""
    cap = _REQUEST_CAP.get()
    if not as_of:
        return (cap if cap is not None else resolve_as_of(None)), None
    t = resolve_as_of(as_of)
    if cap is not None and t > cap:
        return cap, f"요청 시점 {cap.date()} 이후({t.date()})는 조회하지 않아 {cap.date()} 기준으로 답함"
    return t, None


def _with_note(out: dict, note: str | None) -> dict:
    """상한 알림이 있으면 결과에 붙인다."""
    return {**out, "as_of_note": note} if note else out


def _load_raw(kind: str, name: str) -> pd.DataFrame:
    """저장된 원본을 읽는다. 없으면 빈 표."""
    path = Path(config.RAW_DIR) / kind / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@functools.lru_cache(maxsize=64)
def _live(kind: str, name: str, start: str, end: str) -> pd.DataFrame:
    """저장 범위를 넘는 기간을 실시간으로 받는다. end 는 그 시장 종가가 확정된 날 이하라 장중 값은 들어오지 않는다. 같은 요청은 캐시한다."""
    if kind == "prices" and name == "samsung_ohlcv":
        return collect_prices.fetch_ohlcv(config.SAMSUNG_CODE, start, end)
    if kind == "prices" and name == "samsung_investor_flows":
        return collect_prices.fetch_foreign_net(config.SAMSUNG_CODE, start, end)
    if kind == "macro":
        return collect_macro.fetch_yf(config.YF_TICKERS[name], start, end)
    raise ValueError(f"실시간으로 받을 수 없는 데이터: {kind}/{name}")


# 시장별 시간대와 마감 시각. 실시간 조회는 마감이 지난 날까지만 받는다(장중 미확정 값 제외)
MARKET_CLOSE = config.MARKET_CLOSE
US_CLOSE = config.US_MARKET_CLOSE


def last_closed_date(kind: str, name: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
    """그 시장에서 종가가 확정된 가장 최근 현지 날짜."""
    tz, close = MARKET_CLOSE.get(kind if kind == "prices" else name, US_CLOSE)
    local = (now if now is not None else pd.Timestamp.now(tz="UTC")).tz_convert(tz).tz_localize(None)
    hour, minute = map(int, close.split(":"))
    day = local.normalize()
    return day if local - day >= pd.Timedelta(hours=hour, minutes=minute) else day - pd.Timedelta(days=1)


def series_until(kind: str, name: str, cut: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    """cut 날짜보다 앞선 행만 돌려준다. 저장본이 cut 전날까지 덮지 못하면 실시간으로 받는다. (표, 출처 설명)"""
    stored = _load_raw(kind, name)
    note = "저장된 데이터"
    if stored.empty or stored.index.max() < cut - pd.Timedelta(days=1):
        end = min(cut - pd.Timedelta(days=1), last_closed_date(kind, name))
        # 저장본은 그대로 두고 그 뒤 날짜만 이어 붙인다(학습 기간 전체의 값이 바뀌지 않게)
        start = pd.Timestamp(config.DATA_START) if stored.empty else stored.index.max() + pd.Timedelta(days=1)
        try:
            if start <= end:
                fresh = _live(kind, name, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                fresh = fresh[fresh.index > stored.index.max()] if not stored.empty else fresh
                stored = pd.concat([stored, fresh]) if not stored.empty else fresh
                note = f"저장된 데이터 + 실시간 조회({len(fresh)}일 추가)"
        except Exception as e:  # 실패하면 저장본으로 답하고 그렇다고 밝힌다
            note = f"실시간 조회 실패({type(e).__name__}), 저장된 데이터의 마지막 날까지로 답함"
    out = stored[stored.index < cut]
    assert out.empty or out.index.max() < cut
    return out, note


def _records(df: pd.DataFrame, cols: list[str], digits: int = 4) -> list[dict]:
    """표를 날짜가 붙은 dict 목록으로 바꾼다."""
    view = df[cols].round(digits)
    return [{"date": str(i.date()), **row} for i, row in zip(view.index, view.to_dict("records"))]


def price_history(as_of: str | None = None, days: int = 20) -> dict:
    """as_of 전까지 최근 days 거래일의 삼성전자 일봉과 일간 수익률."""
    cut = resolve_as_of(as_of)
    df, note = series_until("prices", "samsung_ohlcv", cut)
    df = df.assign(ret_1d=df["Close"].pct_change()).tail(days)
    return {"as_of": str(cut.date()), "source": f"FinanceDataReader 삼성전자 일봉 ({note})",
            "rows": _records(df, ["Open", "High", "Low", "Close", "Volume", "ret_1d"])}


def investor_flows(as_of: str | None = None, days: int = 20) -> dict:
    """as_of 전까지 최근 days 거래일의 외국인·기관·개인 순매수(주)와 외국인 보유비율."""
    cut = resolve_as_of(as_of)
    df, note = series_until("prices", "samsung_investor_flows", cut)
    cols = ["foreign_net", "institution_net", "individual_net", "foreign_hold_ratio"]
    return {"as_of": str(cut.date()), "source": f"네이버 증권 모바일 API ({note})", "rows": _records(df.tail(days), cols)}


def macro_snapshot(as_of: str | None = None, days: int = 5) -> dict:
    """as_of 전(현지 날짜 기준)까지 해외·거시 지표 7종의 최근 값과 변화."""
    cut = resolve_as_of(as_of)
    out = {}
    for name, ticker in config.YF_TICKERS.items():
        df, note = series_until("macro", name, cut)
        df = df.assign(chg_1d=df["value"].pct_change()).tail(days)
        out[name] = {"ticker": ticker, "source": note, "rows": _records(df, ["value", "chg_1d"])}
    return {"as_of": str(cut.date()), "rule": "현지 날짜가 as_of 보다 앞선 값만", "series": out}


KEY_INDICATORS = ["ret_1d", "ret_5d", "ret_20d", "ma5_gap", "ma20_gap", "ma60_gap", "ma120_gap", "bb_pctb", "bb_width",
                  "ichi_tenkan_kijun", "ichi_cloud_top_gap", "ichi_cloud_bottom_gap", "psar_gap", "psar_trend",
                  "volume_ratio", "foreign_net_ratio_5d"]


def raw_until(cut: pd.Timestamp) -> tuple[dict, dict]:
    """신호일 cut 의 특징을 만들 원천 데이터(cut 이전 값만)와 출처 설명을 모은다.

    거래일 목록에 cut 이 있어야 cut 행이 생기므로, 일봉 끝에 cut 날짜의 빈 행을 덧붙인다(값은 쓰지 않는다).
    """
    loaded = {"ohlcv": series_until("prices", "samsung_ohlcv", cut),
              "flows": series_until("prices", "samsung_investor_flows", cut),
              **{name: series_until("macro", name, cut) for name in config.YF_TICKERS}}
    raw = {k: v[0] for k, v in loaded.items()}
    sources = {k: {"note": v[1], "last_date": str(v[0].index.max().date()) if len(v[0]) else None} for k, v in loaded.items()}
    ohlcv = raw["ohlcv"][["Open", "High", "Low", "Close", "Volume"]].astype(float)
    raw["ohlcv"] = pd.concat([ohlcv, pd.DataFrame(index=[cut], columns=ohlcv.columns, dtype=float)])
    return raw, sources


def indicators(as_of: str | None = None) -> dict:
    """as_of 신호 시점의 차트·수급 지표(전날 종가 기준)와 해외 지표 변화. 학습에 쓰는 특징과 같은 계산이다."""
    cut = resolve_as_of(as_of)
    raw, sources = raw_until(cut)
    table = features.build_features(as_of=cut, raw=raw)
    row = table.iloc[-1]
    foreign = {c: round(float(row[c]), 4) for c in table.columns if c.endswith(("_ret_1d", "_chg_1d", "_level"))}
    return {"as_of": str(cut.date()), "signal_row_date": str(table.index[-1].date()),
            "rule": "국내 지표는 전날 종가까지, 해외는 현지 날짜 < as_of",
            "chart_and_flow": {c: round(float(row[c]), 4) for c in KEY_INDICATORS if c in row and pd.notna(row[c])},
            "foreign": foreign, "data_sources": sources}


def company_report(as_of: str | None = None, max_chars: int = 1500) -> dict:
    """as_of 보다 먼저 접수된 가장 최근 정기보고서의 사업 개요."""
    cut = resolve_as_of(as_of)
    index = pd.read_parquet(Path(config.PROCESSED_DIR) / "dart_index.parquet")
    reports = collect_dart.dart_as_of(index, cut).sort_values("rcept_dt")
    if reports.empty:
        return {"as_of": str(cut.date()), "source": "DART", "report": None}
    r = reports.iloc[-1]
    return {"as_of": str(cut.date()), "source": "DART 정기공시",
            "report": {"name": r["report_nm"], "rcept_dt": str(r["rcept_dt"].date()), "overview": r["overview"][:max_chars]}}


def news_for_signal(as_of: str | None = None, keyword: str | None = None, limit: int = 20) -> dict:
    """신호일 as_of 의 뉴스([전 거래일 15:15, 당일 15:15) KST). 저장된 수집본을 먼저 쓰고, 없으면 Google 뉴스 RSS 로 받는다."""
    t = resolve_as_of(as_of)
    days = collect_news.trading_days()
    if t > days.max():  # 저장된 거래일 목록 밖이면 평일을 거래일로 가정한다
        days = days.append(pd.bdate_range(days.max() + pd.Timedelta(days=1), t + pd.Timedelta(days=7)))
    path = Path(config.PROCESSED_DIR) / "news.parquet"
    news = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    day = news[news["signal_date"] == t] if not news.empty else news
    source = "저장된 수집본(GDELT GKG + Google 뉴스 RSS)"
    if t not in days:
        return {"as_of": str(t.date()), "source": "휴장일이라 신호일이 아니다", "articles": []}
    if day.empty and pd.Timestamp(config.NEWS_START) <= t <= pd.Timestamp(config.NEWS_END):
        return {"as_of": str(t.date()), "source": "수집 구간 안이지만 이 날짜 뉴스는 아직 수집·정리되지 않았다", "articles": []}
    if day.empty:
        try:
            day = collect_news.collect_rss(t, days)
            source = "Google 뉴스 RSS 실시간 조회(발행일만 믿을 수 있어 하루 늦게 배정됨)"
        except Exception as e:
            return {"as_of": str(t.date()), "source": f"뉴스 조회 실패({type(e).__name__})", "articles": []}
    if day.empty:
        return {"as_of": str(t.date()), "source": source, "articles": []}
    start, end = collect_news.signal_window(t, days)
    published = pd.to_datetime(day["published_utc"], utc=True)
    day = day[(published >= start) & (published < end)]
    if keyword:
        day = day[day["title"].str.contains(keyword, case=False, na=False, regex=False)]
    day = day.sort_values("published_utc", ascending=False).head(limit)
    return {"as_of": str(t.date()), "source": source, "window_kst": [str(start.tz_convert(KST)), str(end.tz_convert(KST))],
            "articles": [{"published_kst": pd.Timestamp(r.published_utc).tz_convert(KST).strftime("%Y-%m-%d %H:%M"),
                          "source": r.source, "title": r.title, "url": r.url} for r in day.itertuples()]}


def signal_for(date: str) -> dict:
    """그날 숫자 모델 신호와 확률, 뉴스 보정 결과."""
    t = resolve_as_of(date)
    out = {"date": str(t.date())}
    numeric = pd.read_parquet(config.SIGNALS_NUMERIC_PATH)
    if t in numeric.index:
        r = numeric.loc[t]
        out["numeric"] = {"signal": r["signal"], "confidence": round(float(r["confidence"]), 3),
                          "probabilities": {c: round(float(r[f"p_{c}"]), 3) for c in ("상승", "보합", "하락")},
                          "model_block_start": str(r["retrained_on"].date()),
                          "train_label_last_date": str(r["train_last_future_date"].date())}
    live_path = Path(config.PROCESSED_DIR) / "signals_live.parquet"
    if "numeric" not in out and live_path.exists():  # 검증 구간 뒤의 날짜는 실시간으로 만든 신호를 쓴다
        live = pd.read_parquet(live_path)
        if t in live.index:
            r = live.loc[t]
            out["numeric"] = {"signal": r["numeric_signal_label"], "confidence": round(float(r["confidence"]), 3),
                              "probabilities": {c: round(float(r[f"p_{c}"]), 3) for c in ("상승", "보합", "하락")},
                              "model_block_start": str(pd.Timestamp(r["retrained_on"]).date()),
                              "train_label_last_date": str(pd.Timestamp(r["train_last_future_date"]).date()),
                              "source": "실시간 생성(live_signal)"}
            if pd.notna(r.get("news_consensus")):
                out["with_news"] = {"signal": r["signal"], "adjusted": r["signal"] != r["numeric_signal_label"],
                                    "news_consensus": int(r["news_consensus"]), "news_confidence": round(float(r["news_consensus_conf"]), 2)}
    news_path = Path(config.PROCESSED_DIR) / "signals_news.parquet"
    if news_path.exists() and "with_news" not in out:
        news = pd.read_parquet(news_path)
        if t in news.index and news.loc[t, "news_scored"]:
            r = news.loc[t]
            out["with_news"] = {"signal": r["signal"], "adjusted": bool(r["adjusted"]),
                                "news_consensus": int(r["news_consensus"]), "news_confidence": round(float(r["news_consensus_conf"]), 2)}
    if len(out) == 1:
        days = collect_news.trading_days()
        if t.weekday() >= 5 or (days.min() <= t <= days.max() and t not in days):
            out["note"] = "휴장일(주말·공휴일)이라 신호가 없다. 신호는 거래일 15:15 에만 나온다"
        elif t < pd.Timestamp(config.VALID_START):
            out["note"] = f"검증 구간({config.VALID_START}) 이전이라 신호가 없다"
        else:
            out["note"] = f"아직 신호를 만들지 않은 날짜다(저장된 신호는 {config.DATA_END} 까지)"
    return out


def news_judgement(date: str) -> dict:
    """그날 뉴스 채점의 판단 기록: 채점자 3명의 토론 뒤 의견, 중재자 정리, 핵심 기사 제목."""
    t = resolve_as_of(date)
    path = Path(config.TRACES_DIR) / f"{t.date()}.json"
    if not path.exists():
        return {"date": str(t.date()), "note": "이 날짜는 뉴스 채점을 하지 않았다"}
    trace = json.loads(path.read_text(encoding="utf-8"))
    if trace["signal_date"] != str(t.date()):
        raise ValueError(f"판단 기록 날짜 {trace['signal_date']} 가 요청 날짜 {t.date()} 와 다르다")
    titles = {a["no"]: a["title"] for a in trace["articles"]}
    final = trace["final"]
    return {"date": trace["signal_date"], "as_of": trace["as_of"], "model": trace["model"], "company": trace["company"],
            "final": {**{k: final[k] for k in ("score", "confidence", "rationale", "disagreement")},
                      "key_articles": [titles.get(n) for n in final["key_articles"]]},
            "personas": {p: {k: o[k] for k in ("score", "confidence", "rationale", "change_reason")} for p, o in trace["round2"].items()}}


@functools.lru_cache(maxsize=1)
def backtest_summary() -> dict:
    """검증 구간 전체 모의 매매 결과(기준 A)와 보조 지표 요약."""
    from src.pipeline import backtest  # 무거운 계산이라 필요할 때만 불러온다
    r = backtest.evaluate(pd.read_parquet(config.SIGNALS_NUMERIC_PATH))
    a = r["judge_a"]
    return {"period": r["period"], "criterion_a": {"passed": a["passed"], "mdd_ok": a["mdd_ok"], "return_ok": a["return_ok"]},
            "model": {k: round(v, 4) for k, v in a["model"].items()}, "hold": {k: round(v, 4) for k, v in a["hold"].items()},
            "direction": {k: round(v, 4) for k, v in r["direction"].items()},
            "report": "evaluation/reports/day3_backtest.md"}


# ---- 에이전트에 등록하는 도구 (설명은 모델이 읽는다) ----

def _call(fn, as_of: str, *args) -> dict:
    """요청 상한 안으로 날짜를 맞춘 뒤 도구 함수를 부른다."""
    t, note = capped(as_of or None)
    return _with_note(fn(str(t.date()), *args), note)


def _refuse_after_cap(date: str) -> dict | None:
    """신호·판단 기록 조회에서 요청 시점보다 뒤 날짜면 거부 결과를, 아니면 None 을 돌려준다."""
    cap = _REQUEST_CAP.get()
    if cap is not None and resolve_as_of(date) > cap:
        return {"date": date, "refused": f"요청 시점 {cap.date()} 이후 날짜라 조회하지 않는다"}
    return None


@tool
def get_price(as_of: str = "", days: int = 20) -> dict:
    """삼성전자 일봉(시가·고가·저가·종가·거래량)과 일간 수익률. as_of(YYYY-MM-DD) 전날까지의 최근 days 거래일. as_of 를 비우면 질문 시점 기준."""
    return _call(price_history, as_of, days)


@tool
def get_investor_flows(as_of: str = "", days: int = 20) -> dict:
    """삼성전자 외국인·기관·개인 순매수(주)와 외국인 보유비율. as_of 전날까지의 최근 days 거래일."""
    return _call(investor_flows, as_of, days)


@tool
def get_macro(as_of: str = "", days: int = 5) -> dict:
    """미국 10년물 금리, VIX, WTI 유가, 원/달러, SOX, 엔비디아, 항셍테크의 최근 값과 변화. 현지 날짜가 as_of 보다 앞선 값만."""
    return _call(macro_snapshot, as_of, days)


@tool
def get_indicators(as_of: str = "") -> dict:
    """as_of 신호 시점의 차트 지표(이동평균 괴리, 볼린저, 일목, 파라볼릭 SAR, 거래량)와 수급·해외 지표. 모델 학습에 쓰는 계산과 같다."""
    return _call(indicators, as_of)


@tool
def get_company_report(as_of: str = "") -> dict:
    """as_of 보다 먼저 공시된 가장 최근 삼성전자 정기보고서의 사업 개요(DART)."""
    return _call(company_report, as_of)


@tool
def search_news(as_of: str = "", keyword: str = "", limit: int = 20) -> dict:
    """신호일 as_of 의 뉴스 제목(전 거래일 15:15 ~ 당일 15:15 KST). keyword 로 제목을 거를 수 있다."""
    return _call(news_for_signal, as_of, keyword or None, limit)


@tool
def get_signal(date: str) -> dict:
    """그날의 모델 신호(상승·보합·하락), 확률, 뉴스 보정 결과. 질문 시점보다 뒤 날짜는 조회하지 않는다."""
    return _refuse_after_cap(date) or signal_for(date)


@tool
def get_news_judgement(date: str) -> dict:
    """그날 뉴스 채점의 판단 기록: 채점자 3명(단기 트레이더·가치투자자·리스크 관리자) 의견, 중재자 정리, 핵심 기사. 질문 시점보다 뒤 날짜는 조회하지 않는다."""
    return _refuse_after_cap(date) or news_judgement(date)


@tool
def get_backtest_summary() -> dict:
    """검증 구간 전체의 모의 매매 성과(기준 A 판정, 수익률, MDD, 방향 적중률). 사후 평가다.
    모델 성과를 묻는 질문에만 쓰고, 특정 날짜의 판단 근거로 쓰지 않는다. 질문 시점이 검증 구간 끝 이전이면 제공하지 않는다."""
    cap = _REQUEST_CAP.get()
    available = pd.Timestamp(config.DATA_END) + pd.Timedelta(days=1)
    if cap is not None and cap < available:
        return {"refused": f"검증 구간 전체 결과는 {available.date()} 이후에 알 수 있어 요청 시점 {cap.date()} 에는 제공하지 않는다"}
    return backtest_summary()


TOOLS = [get_price, get_investor_flows, get_macro, get_indicators, get_company_report,
         search_news, get_signal, get_news_judgement, get_backtest_summary]
