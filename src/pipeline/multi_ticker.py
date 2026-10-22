"""7일차 다종목 분석 (DAY7_PLAN.md, SERVICE2.md). 분석용 신호만 만들고 매매·주문 경로와 연결하지 않는다.

- 수집: 종목별 일봉(FinanceDataReader)·수급(네이버)과 분야 ETF(yfinance)를 data/raw/tickers/ 에 따로 둔다.
  이미 저장한 행은 바꾸지 않고, 갱신은 새 확정 날짜만 덧붙인다.
- 특징: 국내 지표(가격·차트·수급)는 그 종목, 거시·해외·지수는 기존 원본(확장 원본 포함)으로 만든다. 시점 규칙은 features.build_features 그대로.
- 학습: 운영 설정(지표 묶음 base, 5거래일 ±3%, 매주 재학습, 누적)으로 종목별 단독 워크포워드와 4종목 공동 워크포워드.
- 판정: 사전 등록 Q1 ~ Q3(evaluation/reports/preregistration.md)을 확장 구간에서 한 번 계산해 리포트로 남긴다.

명령줄: python -m src.pipeline.multi_ticker collect | study | live [YYYY-MM-DD]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src import config
from src.pipeline import backtest as bt
from src.pipeline import collect_macro, collect_prices, features
from src.pipeline.labels import DOWN, UP, make_labels
from src.pipeline.train import CLASSES, feature_columns, prediction_chunks, walk_forward

SINGLE_MODEL_TAG = "S0 단독"  # 분야 알고리즘이 채택되지 않은 종목의 참고 신호 꼬리표
SECTOR_PREFIX = "sector"  # 분야 ETF 지표 열 이름 앞부분(종목마다 ETF 가 달라도 같은 열 이름)


def raw_path(name: str) -> Path:
    """종목 원본 파일 경로."""
    return Path(config.TICKER_RAW_DIR) / f"{name}.parquet"


def append_only(path: Path, fresh: pd.DataFrame) -> int:
    """저장본 끝날 뒤의 새 행만 덧붙인다(이미 저장한 행은 바꾸지 않는다). 덧붙인 행 수를 돌려준다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fresh.sort_index().to_parquet(path)
        return len(fresh)
    stored = pd.read_parquet(path)
    new = fresh[fresh.index > stored.index.max()]
    if len(new):
        pd.concat([stored, new]).sort_index().to_parquet(path)
    return len(new)


def collect(end: str | None = None, tickers=None) -> pd.DataFrame:
    """종목별 일봉·수급과 분야 ETF 를 받아 덧붙인다. 삼성전자는 기존 원본을 쓰므로 받지 않는다."""
    end = end or config.DATA_END
    rows = []
    for code in tickers or [c for c in config.MULTI_TICKERS if c != config.SAMSUNG_CODE]:
        for kind, fetch in [("ohlcv", collect_prices.fetch_ohlcv), ("flows", collect_prices.fetch_foreign_net)]:
            path = raw_path(f"{code}_{kind}")
            start = config.MULTI_DATA_START if not path.exists() else str((pd.read_parquet(path).index.max() + pd.Timedelta(days=1)).date())
            if pd.Timestamp(start) > pd.Timestamp(end):
                rows.append({"name": path.stem, "added": 0})
                continue
            df = fetch(code, start, end)
            rows.append({"name": path.stem, "added": append_only(path, df.loc[:end])})
    for name, code in config.SECTOR_KR_INDEX.items():  # 8일차 국내 지수(코스닥). 종목과 같은 끝날까지
        path = raw_path(f"kr_{name}")
        start = config.MULTI_DATA_START if not path.exists() else str((pd.read_parquet(path).index.max() + pd.Timedelta(days=1)).date())
        if pd.Timestamp(start) <= pd.Timestamp(end):
            rows.append({"name": path.stem, "added": append_only(path, collect_prices.fetch_close(code, start, end).loc[:end])})
    from src.pipeline.refresh_data import last_closed_date
    for name, ticker in config.SECTOR_ETFS.items():
        path = raw_path(f"sector_{name}")
        etf_end = str(min(pd.Timestamp(end), last_closed_date("macro", ticker)).date())  # 미국 장중 값을 종가로 저장하지 않는다
        start = config.MULTI_DATA_START if not path.exists() else str((pd.read_parquet(path).index.max() + pd.Timedelta(days=1)).date())
        if pd.Timestamp(start) <= pd.Timestamp(etf_end):
            rows.append({"name": path.stem, "added": append_only(path, collect_macro.fetch_yf(ticker, start, etf_end).loc[:etf_end])})
    return pd.DataFrame(rows)


def base_raw() -> dict:
    """거시·해외·지수 원본(확장 원본을 이어 붙인 것). 종목 일봉·수급은 ticker_raw 가 바꿔 끼운다."""
    return features.load_raw_extended()


def ticker_raw(code: str, shared: dict | None = None) -> dict:
    """그 종목의 원본 묶음. 삼성전자는 기존(확장) 원본 그대로."""
    raw = dict(shared or base_raw())
    if code != config.SAMSUNG_CODE:
        raw["ohlcv"] = pd.read_parquet(raw_path(f"{code}_ohlcv"))
        raw["flows"] = pd.read_parquet(raw_path(f"{code}_flows"))
    return raw


def sector_features(code: str, days: pd.DatetimeIndex, as_of=None) -> pd.DataFrame | None:
    """분야 ETF 1·5일 수익률과 묵은 일수(해외 지표와 같은 시점 규칙: 현지 날짜 < t). 분야 ETF 가 없는 종목은 None."""
    etf = config.MULTI_TICKERS[code][2]
    if etf is None:
        return None
    series = pd.read_parquet(raw_path(f"sector_{etf}"))
    if as_of is not None:
        series = series[series.index < pd.Timestamp(as_of)]
    attached = features.attach_foreign(days, features.foreign_series_features(SECTOR_PREFIX, series), SECTOR_PREFIX)
    features.assert_no_lookahead(attached)
    return attached.drop(columns=[f"{SECTOR_PREFIX}_src_date"])


def sector_columns() -> list[str]:
    """분야 ETF 지표 열."""
    return [f"{SECTOR_PREFIX}_ret_{w}d" for w in config.FOREIGN_CHANGE_WINDOWS] + [f"{SECTOR_PREFIX}_age_days"]


def build_tables(shared: dict | None = None) -> dict[str, dict]:
    """종목마다 특징 표(분야 ETF 지표 포함)와 정답 표를 만든다."""
    shared = shared or base_raw()
    out = {}
    for code in config.MULTI_TICKERS:
        raw = ticker_raw(code, shared)
        table = features.build_features(raw=raw)
        sector = sector_features(code, table.index)
        if sector is not None:
            table = table.join(sector)
        out[code] = {"features": table, "labels": make_labels(raw["ohlcv"]["Close"].dropna().astype(float)),
                     "close": raw["ohlcv"]["Close"].astype(float)}
    return out


def pooled_walk_forward(tables: dict[str, dict], cols: list[str], valid_start, freq: str | None = None) -> dict[str, pd.DataFrame]:
    """공동 모델: 재학습일마다 모든 종목의 '정답 기간이 재학습일 전에 끝난' 행을 합쳐 학습하고 종목마다 예측한다.
    재학습 일정은 종목 공통 거래일로 정한다(한국 종목은 거래일이 같다)."""
    for t in tables.values():  # train.walk_forward 와 같은 시점 검사
        features.assert_no_lookahead(t["features"][cols])
    usable = {c: t["features"][cols].dropna() for c, t in tables.items()}
    pools = {c: usable[c].join(tables[c]["labels"][["label", "future_date"]], how="inner") for c in tables}
    days = sorted(set().union(*[u.index for u in usable.values()]))
    preds = {c: [] for c in tables}
    for start, chunk in prediction_chunks(pd.DatetimeIndex(days), valid_start, freq):
        train = pd.concat([p[p["future_date"] < start] for p in pools.values()])
        if config.TRAIN_WINDOW_YEARS:  # train.walk_forward 와 같은 학습 창
            train = train[train.index >= start - pd.DateOffset(years=config.TRAIN_WINDOW_YEARS)]
        if len(train) < config.MIN_TRAIN_ROWS:
            raise ValueError(f"{start.date()} 공동 재학습: 학습 행 {len(train)}개 < 최소 {config.MIN_TRAIN_ROWS}개")
        if train["future_date"].max() >= start:
            raise AssertionError("학습 정답 기간이 예측 시작일과 겹친다")
        model = LGBMClassifier(**config.LGBM_PARAMS).fit(train[cols], train["label"].map(CLASSES.index))
        for c, u in usable.items():
            rows = u.index[u.index.isin(chunk)]
            if not len(rows):
                continue
            proba = pd.DataFrame(model.predict_proba(u.loc[rows, cols]), index=rows,
                                 columns=[CLASSES[i] for i in model.classes_]).reindex(columns=CLASSES, fill_value=0.0)
            preds[c].append(pd.DataFrame({"signal": proba.idxmax(axis=1), "confidence": proba.max(axis=1),
                                          **{f"p_{k}": proba[k] for k in CLASSES}, "retrained_on": start,
                                          "train_rows": len(train), "train_last_future_date": train["future_date"].max()}, index=rows))
    return {c: pd.concat(v) for c, v in preds.items() if v}


def period_report(signal: pd.Series, close: pd.Series, labels: pd.DataFrame, start, end) -> dict:
    """한 구간을 현금에서 다시 시작해 모의 매매(하락 = 현금)하고 성공 기준 A·B·C 와 보조 숫자를 계산한다."""
    sig = signal.loc[start:end]
    sim = bt.simulate(sig, close, pd.Series(dtype=float), "cash")
    hold = bt.buy_and_hold(close, sig.index)
    m, h = bt.summary(sim["equity"]), bt.summary(hold)
    known = labels[labels["future_date"] <= pd.Timestamp(sig.index.max())]["label"].reindex(sig.index)
    has = known.notna()
    down = sig[has] == DOWN
    weights = (sim["position"] == bt.LONG).astype(float)
    up_share, up_months = bt.up_month_hold(weights, close)
    a = bt.judge_a(sim["equity"], hold)
    down_hit = float((known[has][down] == DOWN).mean()) if down.any() else float("nan")
    actual_down = float((known[has] == DOWN).mean()) if has.any() else float("nan")
    return {"days": len(sig), "return": m["total_return"], "mdd": m["mdd"], "calmar": m["calmar"],
            "hold_return": h["total_return"], "hold_mdd": h["mdd"], "A": bool(a["passed"]),
            "B": None if np.isnan(down_hit) else bool(down_hit > actual_down), "down_hit": down_hit, "actual_down": actual_down,
            "C": None if up_months == 0 else bool(up_share >= config.UP_MONTH_MIN_HOLD), "up_hold": up_share, "up_months": up_months,
            "accuracy": float((sig[has] == known[has]).mean()) if has.any() else float("nan"),
            "always_up": float((known[has] == UP).mean()) if has.any() else float("nan"),
            "most_common": float(known[has].value_counts(normalize=True).max()) if has.any() else float("nan"),
            "hold_days": float(weights.mean())}


EXT = (config.DEV_START, config.EXT_VALID_END)          # 확장 구간(판정): 2022-03-04 ~ 2025-04-22
VALID = (config.VALID_START, None)                      # 검증 구간(참고): 2025-04-23 ~ 데이터 끝


def study(save: bool = True) -> dict:
    """사전 등록 Q1 ~ Q3 을 한 번 계산하고 리포트를 쓴다. 신호 표도 저장한다."""
    tables = build_tables()
    base_cols = list(config.FEATURE_SET_BASE)
    start = pd.Timestamp(config.DEV_START)
    single, pooled_sig, sector_sig = {}, {}, {}
    skipped = {}
    for code, t in tables.items():  # 종목별 단독(운영 설정 그대로). 학습 행이 모자라면(상장이 늦은 종목) 건너뛰고 이유를 남긴다
        try:
            single[code] = walk_forward(t["features"], t["labels"], valid_start=start, feature_set="base")
        except ValueError as e:
            skipped[code] = str(e)
    pooled_sig = pooled_walk_forward(tables, base_cols, start)          # Q2 공동 모델
    for code, t in tables.items():  # Q3 분야 ETF 지표 추가(해당 종목만, 단독 모델)
        if config.MULTI_TICKERS[code][2]:
            cols = base_cols + sector_columns()
            sub = {"features": t["features"][cols], "labels": t["labels"]}
            sector_sig[code] = _walk_forward_cols(sub["features"], sub["labels"], cols, start)
    rows = []
    for code, t in tables.items():
        name = config.MULTI_TICKERS[code][0]
        for model, sigs in [("단독", single), ("공동", pooled_sig), ("단독+분야 ETF", sector_sig)]:
            if code not in sigs:
                continue
            for period, (a, b) in [("확장(판정)", EXT), ("검증(참고)", VALID)]:
                r = period_report(sigs[code]["signal"], t["close"], t["labels"], a, b)
                rows.append({"code": code, "name": name, "model": model, "period": period,
                             "first_train_rows": int(sigs[code]["train_rows"].iloc[0]), **r})
    result = pd.DataFrame(rows)
    verdicts = judge(result)
    verdicts["skipped"] = skipped
    if save:
        out = Path(config.MULTI_SIGNALS_DIR)
        out.mkdir(parents=True, exist_ok=True)
        for label, sigs in [("single", single), ("pooled", pooled_sig), ("sector", sector_sig)]:
            for code, s in sigs.items():
                s.to_parquet(out / f"signals_{label}_{code}.parquet")
        result.to_csv(out / "study.csv", index=False, encoding="utf-8-sig")
        Path(config.DAY7_REPORT_PATH).write_text(report(result, verdicts), encoding="utf-8", newline="\n")
    return {"result": result, "verdicts": verdicts}


def _walk_forward_cols(feats: pd.DataFrame, labels: pd.DataFrame, cols: list[str], valid_start) -> pd.DataFrame:
    """지표 열을 직접 지정한 단독 워크포워드(분야 ETF 지표 추가용). 규칙은 train.walk_forward 와 같다."""
    return pooled_walk_forward({"x": {"features": feats, "labels": labels}}, cols, valid_start)["x"]


def judge(result: pd.DataFrame) -> dict:
    """사전 등록 기준으로 Q1 ~ Q3 판정(확장 구간)."""
    ext = result[result["period"] == "확장(판정)"].set_index(["code", "model"])
    has = lambda c, m: (c, m) in ext.index
    q1 = {c: {k: ext.loc[(c, "단독"), k] for k in ("A", "B", "C")} for c in config.MULTI_TICKERS
          if c != config.SAMSUNG_CODE and has(c, "단독")}  # 학습 행이 모자라 건너뛴 종목은 판정하지 않는다
    better = lambda c, m: bool(ext.loc[(c, m), "calmar"] > ext.loc[(c, "단독"), "calmar"]
                               and abs(ext.loc[(c, m), "mdd"]) <= abs(ext.loc[(c, "단독"), "mdd"]))
    q2 = {c: better(c, "공동") for c in config.MULTI_TICKERS if has(c, "공동") and has(c, "단독")}
    q3 = {c: better(c, "단독+분야 ETF") for c in config.MULTI_TICKERS if has(c, "단독+분야 ETF") and has(c, "단독")}
    return {"Q1": q1, "Q2": q2, "Q2_adopt": sum(q2.values()) >= config.MULTI_POOLED_MIN_BETTER, "Q3": q3}


def report(result: pd.DataFrame, v: dict) -> str:
    """7일차 리포트(마크다운)."""
    pct = lambda x: "" if pd.isna(x) else f"{x:.1%}"
    mark = lambda x: "-" if x is None or (isinstance(x, float) and np.isnan(x)) else ("통과" if x else "실패")
    view = pd.DataFrame({
        "종목": result["name"], "모델": result["model"], "구간": result["period"], "첫 학습 행": result["first_train_rows"],
        "수익 / MDD": [f"{r:+.1%} / {m:.1%}" for r, m in zip(result["return"], result["mdd"])], "칼마": result["calmar"].round(3),
        "들고 있기": [f"{r:+.1%} / {m:.1%}" for r, m in zip(result["hold_return"], result["hold_mdd"])],
        "A": result["A"].map(mark), "B": result["B"].map(mark), "C": result["C"].map(mark),
        "정확도": result["accuracy"].map(pct), "항상 상승": result["always_up"].map(pct), "최다 정답": result["most_common"].map(pct),
        "하락 적중 / 실제 하락": [f"{pct(a)} / {pct(b)}" for a, b in zip(result["down_hit"], result["actual_down"])],
        "보유일": result["hold_days"].map(pct)})
    names = {c: v_[0] for c, v_ in config.MULTI_TICKERS.items()}
    lines = ["# 7일차 다종목 분석", "", f"계산 시각 {pd.Timestamp.now(tz=config.KST):%Y-%m-%d %H:%M} (KST) · 계획 [DAY7_PLAN.md](../../DAY7_PLAN.md) · "
             "정리 [SERVICE2.md](../../SERVICE2.md) · 장부 [preregistration.md](preregistration.md)", "",
             f"- 운영 설정 그대로: 지표 {len(config.FEATURE_SET_BASE)}개, 정답 {config.LABEL_HORIZON_DAYS}거래일 ±{config.LABEL_THRESHOLD:.0%}, "
             f"재학습 {config.RETRAIN_FREQS[config.RETRAIN_FREQ]}, 누적 학습, 하락 = 현금, 수수료 {config.FEE_RATE:.1%}",
             f"- 판정 구간 {EXT[0]} ~ {EXT[1]}(한 번만). 삼성전자는 이 구간을 4일차에 본 적이 있다. 검증 구간 {VALID[0]} ~ 은 참고",
             "- 다른 종목은 분석용이고 매매 대상이 아니다", "",
             "## 사전 등록 판정", "",
             "- **Q1 (설정이 다른 종목에도 통하나):** " + "; ".join(
                 f"{names[c]} A {mark(x['A'])}·B {mark(x['B'])}·C {mark(x['C'])}" for c, x in v["Q1"].items()),
             "- **Q2 (공동 학습이 단독보다 나은가):** " + "; ".join(f"{names[c]} {'나음' if x else '아님'}" for c, x in v["Q2"].items())
             + f" → {f'채택 후보({config.MULTI_POOLED_MIN_BETTER}종목 이상)' if v['Q2_adopt'] else '채택 안 함'}",
             "- **Q3 (분야 ETF 지표가 도움이 되나):** " + "; ".join(f"{names[c]} {'나음' if x else '아님'}" for c, x in v["Q3"].items()), "",
             "기준: A = MDD ≤ 들고 있기의 절반 그리고 수익 ≥ 들고 있기의 절반, B = 하락 신호 적중 > 실제 하락 비율, "
             f"C = +{config.UP_MONTH_MIN_RETURN:.0%} 달 보유일 ≥ {config.UP_MONTH_MIN_HOLD:.0%}. "
             "Q2·Q3 은 칼마가 단독보다 높고 MDD 가 더 깊지 않으면 '나음'. '최다 정답' = 늘 가장 많은 정답만 말했을 때의 정확도.", "",
             "## 판정에 대한 메모", "",
             "- Q2 의 SK하이닉스는 칼마 차이가 매우 작아(소수 셋째 자리) 판정이 아슬아슬하다. 사전 등록대로 판정은 그대로 둔다.",
             "- 한화에어로스페이스는 2024-08-29 ~ 2024-09-26 거래정지(거래량 0, 종가 고정 18행)가 원본에 있다. 특징이 비어 신호가 없고"
             " 포지션은 유지된다. 정지 직전 행의 5거래일 정답은 굳은 가격으로 매겨졌다(판정 차이가 커 결론은 같다).",
             "- 확장 구간 단독 정확도는 4종목 모두 '최다 정답'보다 낮다.", "",
             "## 결과 표", "", view.to_markdown(index=False), ""]
    return "\n".join(lines) + "\n"


# ---- 실시간 참고 신호 (D7-07): 다른 종목의 오늘 신호. 매매·주문과 연결하지 않는다 ----

def refresh(today: pd.Timestamp | None = None) -> pd.DataFrame:
    """다른 종목 원본에 어제까지의 새 확정 날짜만 덧붙인다(오늘 장중 값은 저장하지 않는다)."""
    today = (pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz=config.KST).tz_localize(None)).normalize()
    return collect(end=str((today - pd.Timedelta(days=1)).date()), tickers=live_tickers())


def live_tickers() -> list[str]:
    """매일 덧붙일 종목: 다른 종목과, 채택된 분야 알고리즘이 함께 쓰는 같은 분야 종목(삼성전자 제외)."""
    from src.pipeline.sector_models import sector_of
    codes = [c for c in config.MULTI_TICKERS if c != config.SAMSUNG_CODE]
    peers = [p for c in config.MULTI_TICKERS if c in config.SECTOR_TICKER_NAMES and sector_of(c) in config.SECTOR_ADOPTED
             for p in config.SECTOR_UNIVERSE[sector_of(c)]]
    return list(dict.fromkeys(codes + [p for p in peers if p != config.SAMSUNG_CODE]))


def _live_one(code: str, as_of: str | None) -> dict:
    """채택된 분야 알고리즘이 있으면 그것으로, 없으면 단독 모델(운영 설정)로 참고 신호를 만든다."""
    from src.pipeline import sector_models
    if code in config.SECTOR_TICKER_NAMES and sector_models.sector_of(code) in config.SECTOR_ADOPTED:
        return sector_models.live_signal(code, as_of)
    return {**live_signal(code, as_of), "model": SINGLE_MODEL_TAG}


def live_shared(t: pd.Timestamp) -> dict:
    """t 일 신호용 공통 원본(t 전 값만, 운영 설정이면 확장 원본을 이어 붙인다)."""
    from src import tools
    from src.pipeline.live_signal import prepend_extended
    shared, _ = tools.raw_until(t)
    return prepend_extended(shared) if config.OPERATIONAL_EXTENDED_HISTORY else shared


def live_raw(code: str, t: pd.Timestamp, shared: dict) -> dict:
    """t 일 신호용 종목 원본: t 전 값만 두고 t 행을 만들 빈 행을 덧붙인다. 원본이 직전 거래일까지 없으면 멈춘다."""
    raw = ticker_raw(code, shared)
    if code != config.SAMSUNG_CODE:
        before = lambda df: df[df.index < t]
        prev_day = shared["ohlcv"].index[shared["ohlcv"].index < t].max()  # 공통 원본의 t 직전 거래일
        for kind in ("ohlcv", "flows"):  # 종목 원본이 그날까지 없으면 과거 검증과 다른 t 행이 되므로 만들지 않는다
            if before(raw[kind]).index.max() != prev_day:
                raise ValueError(f"{code} {kind} 원본 끝 {before(raw[kind]).index.max().date()} ≠ 직전 거래일 {prev_day.date()}")
        ohlcv = before(raw["ohlcv"])[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        raw["ohlcv"] = pd.concat([ohlcv, pd.DataFrame(index=[t], columns=ohlcv.columns, dtype=float)])  # t 행을 만들기 위한 빈 행
        raw["flows"] = before(raw["flows"])
    return raw


def live_signal(code: str, as_of: str | None = None, valid_start=None) -> dict:
    """그 종목의 t 일 15:15 신호(운영 설정, 단독 모델). 원본은 t 전날까지, 재학습은 그 주 첫 거래일(과거 검증과 같은 규칙)."""
    from src import tools
    from src.pipeline.live_signal import block_start
    t = tools.resolve_as_of(as_of)
    raw = live_raw(code, t, live_shared(t))
    table = features.build_features(as_of=t, raw=raw)
    if table.index[-1] != t:
        raise ValueError(f"{code} {t.date()} 행을 만들지 못했다")
    cols = feature_columns(table, "base")
    labels = make_labels(raw["ohlcv"]["Close"].dropna().astype(float))
    start = block_start(t, table.index[table[cols].notna().all(axis=1)], valid_start)
    train = table[cols].dropna().join(labels[["label", "future_date"]], how="inner")
    train = train[train["future_date"] < start]
    if config.TRAIN_WINDOW_YEARS:
        train = train[train.index >= start - pd.DateOffset(years=config.TRAIN_WINDOW_YEARS)]
    if len(train) < config.MIN_TRAIN_ROWS:
        raise ValueError(f"{code} 학습 행 {len(train)}개 < 최소 {config.MIN_TRAIN_ROWS}개")
    model = LGBMClassifier(**config.LGBM_PARAMS).fit(train[cols], train["label"].map(CLASSES.index))
    row = table.loc[[t], cols]
    if row.isna().any(axis=None):
        raise ValueError(f"{code} {t.date()} 특징에 빈 값이 있다")
    proba = pd.Series(model.predict_proba(row)[0], index=[CLASSES[i] for i in model.classes_]).reindex(CLASSES, fill_value=0.0)
    return {"date": t, "code": code, "name": config.MULTI_TICKERS[code][0], "signal": proba.idxmax(),
            "confidence": float(proba.max()), **{f"p_{c}": float(proba[c]) for c in CLASSES}, "retrained_on": start,
            "train_rows": len(train), "train_last_future_date": train["future_date"].max()}


def live_signals(as_of: str | None = None) -> list[dict]:
    """다른 종목 오늘 신호를 만들어 MULTI_LIVE_PATH 에 (날짜, 종목) 기준으로 남긴다. 한 종목이 실패해도 나머지는 만든다."""
    out = []
    for code in [c for c in config.MULTI_TICKERS if c != config.SAMSUNG_CODE]:
        try:
            out.append(_live_one(code, as_of))
        except Exception as e:
            out.append({"code": code, "name": config.MULTI_TICKERS[code][0], "error": f"{type(e).__name__}: {e}"})
    ok = [r for r in out if "error" not in r]
    if ok:
        path = Path(config.MULTI_LIVE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(ok).set_index("date")
        new["created_at"] = pd.Timestamp.now(tz=config.KST).tz_localize(None)
        if path.exists():
            old = pd.read_parquet(path)
            keep = ~(old.index.isin(new.index) & old["code"].isin(new["code"]))
            new = pd.concat([old[keep], new]).sort_index()
        new.to_parquet(path)
    return out


def summary_lines(results: list[dict]) -> list[str]:
    """메일용 요약(참고, 매매 대상 아님)."""
    return [f"{r['name']}({r['code']}): " + (f"{r['signal']} (확신도 {r['confidence']:.0%}, {r.get('model', SINGLE_MODEL_TAG)})" if "error" not in r
                                             else f"실패 {r['error'][:60]}")
            for r in results]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "collect":
        print(collect().to_string(index=False))
    elif cmd == "study":
        res = study()
        print(res["verdicts"])
        print("저장:", config.DAY7_REPORT_PATH)
    elif cmd == "live":
        print("\n".join(summary_lines(live_signals(sys.argv[2] if len(sys.argv) > 2 else None))))
    else:
        print(__doc__)
