"""종목 레지스트리(config.MULTI_TICKERS) 종목의 수집·신호·분석. 7일차 다종목 분석에서 시작해 10일차에 종목 추가 명령(add)으로 모듈화했다
(DAY7_PLAN.md, DAY10_PLAN.md, SERVICE.md 13절). 매매·주문 경로와 연결하지 않는다.

- 종목 추가: config.MULTI_TICKERS 에 한 줄 넣고 `python -m src.pipeline.multi_ticker add <코드>` → 원본 수집 → 과거 신호(backfill) → 종목 분석.
  그 뒤 실시간 신호·성적표·메일·도구·화면은 레지스트리를 따라 자동으로 포함된다.

- 수집: 종목별 일봉(FinanceDataReader)·수급(네이버)과 분야 ETF(yfinance)를 data/raw/tickers/ 에 따로 둔다.
  이미 저장한 행은 바꾸지 않고, 갱신은 새 확정 날짜만 덧붙인다.
- 특징: 국내 지표(가격·차트·수급)는 그 종목, 거시·해외·지수는 기존 원본(확장 원본 포함)으로 만든다. 시점 규칙은 features.build_features 그대로.
- 학습: 운영 설정(지표 묶음 base, 5거래일 ±3%, 매주 재학습, 누적)으로 종목별 단독 워크포워드와 4종목 공동 워크포워드.
- 판정: 사전 등록 Q1 ~ Q3(evaluation/reports/preregistration.md)을 확장 구간에서 한 번 계산해 리포트로 남긴다.

명령줄: python -m src.pipeline.multi_ticker collect | study | analysis | add <코드> | backfill <코드> | live [YYYY-MM-DD]
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

SINGLE_MODEL_TAG = "S0 단독"  # 분야 알고리즘이 채택되지 않은 종목의 신호 꼬리표
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
    for code in tickers or [c for c in config.MULTI_TICKERS if c != config.MAIN_TICKER]:
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
    if code != config.MAIN_TICKER:
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
             "정리 [SERVICE.md](../../SERVICE.md) 11.6 · 장부 [preregistration.md](preregistration.md)", "",
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


# ---- 실시간 신호 (D7-07, 2026-10-23 부터 정식 신호·SERVICE.md 5.6): 다른 종목의 오늘 신호. 매매·주문과 연결하지 않는다 ----

# ---- 종목별 분석·비교 (2026-10-23 사용자 요청, 뉴스 제외). 각 종목의 선정 알고리즘 워크포워드 신호로 계산하며 모두 탐색이다 ----

def selected_signals(code: str) -> tuple[str, pd.DataFrame]:
    """그 종목의 선정 알고리즘(확인까지 통과한 분야면 그 후보, 아니면 S0)과 8일차 워크포워드 신호."""
    from src import tools
    alg = tools.ticker_algorithm(code)
    path = Path(config.SECTOR_DIR) / f"signals_{alg}_{code}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없다. python -m src.pipeline.sector_models run 으로 다시 만든다")
    return alg, pd.read_parquet(path)


def data_profile(close: pd.Series, labels: pd.DataFrame, start, end) -> dict:
    """구간의 데이터 특성: 거래일, 연 환산 변동성, 들고 있기 수익, 정답 분포, 거래정지(거래량 0)는 원본에서 이미 뺐다."""
    c = close.loc[start:end].dropna()
    lab = labels["label"].loc[start:end].dropna()
    dist = lab.value_counts(normalize=True)
    return {"days": len(c), "vol": float(c.pct_change().std() * np.sqrt(bt.TRADING_DAYS_PER_YEAR)), "hold": float(c.iloc[-1] / c.iloc[0] - 1) if len(c) > 1 else float("nan"),
            **{f"label_{k}": float(dist.get(k, 0.0)) for k in CLASSES}}


def yearly_report(signal: pd.Series, close: pd.Series, labels: pd.DataFrame) -> pd.DataFrame:
    """연도마다 현금에서 다시 시작한 모의 매매와 장세(들고 있기 수익 기준)."""
    rows = []
    for year, sig in signal.groupby(signal.index.year):
        if len(sig) < config.TICKER_ANALYSIS_MIN_DAYS:
            continue
        r = period_report(signal, close, labels, sig.index.min(), sig.index.max())
        rows.append({"year": int(year), "regime": bt.regime(r["hold_return"]), **r})
    return pd.DataFrame(rows)


def feature_ic(table: pd.DataFrame, close: pd.Series, cols: list[str], start, end) -> pd.DataFrame:
    """지표마다 5일 뒤 수익률(t 종가 → t+5 종가)과의 순위 상관(IC)과 겹침 보정 t. 지표는 t 에 알 수 있던 값(특징 표 규칙)이다."""
    from src.pipeline.experiment import score_metrics
    h = config.LABEL_HORIZON_DAYS
    fwd = (close.shift(-h) / close - 1).loc[start:end]
    rows = []
    for c in cols:
        if c not in table:
            continue
        m = score_metrics(table[c].loc[start:end], fwd, h - 1)
        group = next((g for g, members in config.FEATURE_GROUPS.items() if c in members), "-")
        rows.append({"feature": c, "group": group, "ic": m["ic"], "nw_t": m["nw_t"], "n": m["n"]})
    return pd.DataFrame(rows)


def analysis(save: bool = True) -> dict:
    """4종목(삼성전자 포함)을 각 선정 알고리즘 신호로 같은 규칙(하락 = 현금, 수수료 0.1%)에서 분석·비교한다.
    데이터 개요, 구간별 성적(A·B·C), 연도별 장세, 방향별 적중, 확신도 구간별 적중, 지표별 5일 IC. 뉴스는 쓰지 않는다."""
    shared = base_raw()
    out = {}
    for code in config.MULTI_TICKERS:
        alg, sig = selected_signals(code)
        from src.pipeline.sector_models import clean_raw
        raw = clean_raw(ticker_raw(code, shared))  # 거래정지일(거래량 0)을 일봉·수급에서 뺀다(8일차 신호와 같은 원본)
        close = raw["ohlcv"]["Close"].astype(float)
        labels = make_labels(close)
        table = features.build_features(raw=raw)
        periods = {}
        for name, (start, end) in config.TICKER_ANALYSIS_PERIODS.items():
            end = end or str(sig.index.max().date())
            s = sig["signal"].loc[start:end]
            if len(s) < config.TICKER_ANALYSIS_MIN_DAYS:
                continue
            known = labels["label"].reindex(s.index)
            matured = known.notna() & (labels["future_date"].reindex(s.index) <= pd.Timestamp(end))  # period_report 와 같게 구간 끝까지 확정된 정답만
            hit = bt.direction_hit_rate(s[matured], known[matured])
            by_dir = {d: float((known[matured][s[matured] == d] == d).mean()) if (s[matured] == d).any() else float("nan") for d in CLASSES}
            share = s.value_counts(normalize=True)
            periods[name] = {"start": start, "end": end, **data_profile(close, labels, start, end),
                             **period_report(sig["signal"], close, labels, start, end),
                             "direction_hit": hit["hit_rate"], "critical_error": hit["critical_error_rate"],
                             **{f"hit_{d}": v for d, v in by_dir.items()}, **{f"share_{d}": float(share.get(d, 0.0)) for d in CLASSES}}
        correct = (sig["signal"] == labels["label"].reindex(sig.index)).where(labels["label"].reindex(sig.index).notna())
        calib = bt.calibration(sig["confidence"], correct.dropna().reindex(sig.index))
        ic = feature_ic(table, close, config.FEATURE_SET_BASE, config.DEV_START, str(sig.index.max().date()))
        sim = bt.simulate(sig["signal"].loc[config.VALID_START:], close, pd.Series(dtype=float), "cash")
        equity = pd.DataFrame({"model": sim["equity"], "hold": bt.buy_and_hold(close, sim.index)})
        out[code] = {"name": config.MULTI_TICKERS[code][0], "sector": config.MULTI_TICKERS[code][1], "algorithm": alg,
                     "periods": periods, "yearly": yearly_report(sig["signal"], close, labels), "calibration": calib,
                     "ic": ic, "equity": equity, "last": str(sig.index.max().date())}
    numeric = pd.read_parquet(config.SIGNALS_NUMERIC_PATH)["signal"]
    _, samsung = selected_signals(config.MAIN_TICKER)
    both = samsung["signal"].to_frame().join(numeric.rename("op"), how="inner")
    match = (int((both["signal"] == both["op"]).sum()), len(both))  # 삼성전자 선정 신호(S0)와 운영 신호가 같은 날 수
    report = analysis_report(out, match)
    if save:
        Path(config.TICKER_ANALYSIS_REPORT_PATH).write_text(report, encoding="utf-8", newline="\n")
        d = Path(config.TICKER_ANALYSIS_DIR)
        d.mkdir(parents=True, exist_ok=True)
        rows = [{"code": c, "name": r["name"], "algorithm": r["algorithm"], "period": p, **v} for c, r in out.items() for p, v in r["periods"].items()]
        pd.DataFrame(rows).to_parquet(d / "periods.parquet")
        pd.concat([r["yearly"].assign(code=c) for c, r in out.items()]).to_parquet(d / "yearly.parquet")
        pd.concat([r["ic"].assign(code=c) for c, r in out.items()]).to_parquet(d / "feature_ic.parquet")
        pd.concat([r["equity"].assign(code=c) for c, r in out.items()]).to_parquet(d / "equity.parquet")
        pd.concat([r["calibration"].reset_index().assign(bin=lambda x: x["bin"].astype(str), code=c) for c, r in out.items()]).to_parquet(d / "calibration.parquet")
    return out


def analysis_report(out: dict, samsung_match: tuple[int, int] | None = None) -> str:
    """종목별 분석·비교 리포트 본문. samsung_match 는 (운영 신호와 같은 날, 겹치는 날)."""
    pct = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"
    num = lambda v, p=2: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"
    mark = lambda v: "-" if v is None else ("통과" if v else "실패")
    last = max(r["last"] for r in out.values())
    lines = ["# 종목별 분석·비교 (뉴스 제외)", "",
             f"> 작성 {pd.Timestamp.now(tz=config.KST):%Y-%m-%d %H:%M} · 신호 끝 {last} · 2026-10-23 사용자 요청 · 기준 [SERVICE.md](../../SERVICE.md) 5.6 · 종목 확장 기록 [SERVICE.md](../../SERVICE.md) 11.6 ~ 11.8",
             f"> 각 종목의 **선정 알고리즘**(8일차 확인 결과) 워크포워드 신호로, 삼성전자와 같은 규칙({config.RETRAIN_FREQS[config.RETRAIN_FREQ]} 재학습, "
             f"{config.LABEL_HORIZON_DAYS}거래일 ±{config.LABEL_THRESHOLD:.0%}, 하락 = 현금, 수수료 {config.FEE_RATE:.1%}, 구간마다 현금에서 시작)으로 계산했다."
             + (f" 삼성전자 S0 신호는 운영 신호와 겹치는 {samsung_match[1]}일 중 {samsung_match[0]}일 같다." if samsung_match else ""),
             "> **이 구간들은 7·8일차에 이미 본 구간이라 모든 숫자는 탐색이다.** 판정은 실시간 성적표(5.2·5.6)로만 한다. 뉴스는 쓰지 않았다.", "",
             f"## 1. 한눈에 비교 (검증 구간 {config.VALID_START} ~)", "",
             "| 종목 | 분야 | 알고리즘 | 모델 수익 | 들고 있기 | 모델 MDD | 들고 있기 MDD | 칼마 | 정확도 | 늘 최빈 정답 | 방향 적중 | 보유일 | A | B | C |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in out.items():
        p = r["periods"].get("검증 구간")
        if p:
            lines.append(f"| {r['name']} | {r['sector']} | {r['algorithm']} | {pct(p['return'])} | {pct(p['hold_return'])} | {pct(p['mdd'])} | {pct(p['hold_mdd'])} | "
                         f"{num(p['calmar'])} | {pct(p['accuracy'])} | {pct(p['most_common'])} | {pct(p['direction_hit'])} | {pct(p['hold_days'])} | "
                         f"{mark(p['A'])} | {mark(p['B'])} | {mark(p['C'])} |")
    lines += ["", f"A = MDD 가 들고 있기의 절반 이하이고 수익이 들고 있기의 절반 이상, B = 하락 신호 적중 > 실제 하락 비율, "
              f"C = +{config.UP_MONTH_MIN_RETURN:.0%} 이상 오른 달 보유일 ≥ {config.UP_MONTH_MIN_HOLD:.0%}.", "",
              "## 2. 데이터 개요", "", "| 종목 | 구간 | 거래일 | 연 변동성 | 들고 있기 | 정답 상승 | 보합 | 하락 |", "|---|---|---|---|---|---|---|---|"]
    for c, r in out.items():
        for name, p in r["periods"].items():
            lines.append(f"| {r['name']} | {name} | {p['days']} | {pct(p['vol'])} | {pct(p['hold'])} | {pct(p['label_상승'])} | {pct(p['label_보합'])} | {pct(p['label_하락'])} |")
    lines += ["", f"들고 있기는 수수료 없는 단순 가격 변화다(1·3절은 매수 수수료 포함이라 조금 다르다). 변동성이 클수록 ±{config.LABEL_THRESHOLD:.0%} 기준에서 "
              "'보합'이 드물어 문제의 성격이 달라진다(7일차에서 본 것).", "",
              "## 3. 구간별 성적", ""]
    for c, r in out.items():
        lines += [f"### {r['name']} ({c}, {r['algorithm']})", "",
                  "| 구간 | 일수 | 모델 수익 / MDD | 들고 있기 수익 / MDD | 칼마 | 정확도 (최빈) | 상승 신호 적중 | 하락 신호 적중 (실제 하락) | 정반대 오류 | 신호 비중 상승·보합·하락 | A | B | C (상승달 보유) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, p in r["periods"].items():
            lines.append(f"| {name} | {p['days']} | {pct(p['return'])} / {pct(p['mdd'])} | {pct(p['hold_return'])} / {pct(p['hold_mdd'])} | {num(p['calmar'])} | "
                         f"{pct(p['accuracy'])} ({pct(p['most_common'])}) | {pct(p['hit_상승'])} | {pct(p['hit_하락'])} ({pct(p['actual_down'])}) | {pct(p['critical_error'])} | "
                         f"{pct(p['share_상승'])}·{pct(p['share_보합'])}·{pct(p['share_하락'])} | {mark(p['A'])} | {mark(p['B'])} | {mark(p['C'])} ({pct(p['up_hold'])}) |")
        lines.append("")
    lines += ["## 4. 연도별 장세 (각 연도 현금에서 시작)", "", "| 종목 | 연도 | 장세 | 모델 수익 / MDD | 들고 있기 수익 / MDD | 정확도 | 보유일 |", "|---|---|---|---|---|---|---|"]
    for c, r in out.items():
        for _, y in r["yearly"].iterrows():
            lines.append(f"| {r['name']} | {y['year']} | {y['regime']} | {pct(y['return'])} / {pct(y['mdd'])} | {pct(y['hold_return'])} / {pct(y['hold_mdd'])} | {pct(y['accuracy'])} | {pct(y['hold_days'])} |")
    lines += ["", "## 5. 확신도 구간별 적중 (전체 기간)", "", "| 종목 | " + " | ".join(str(b) for b in next(iter(out.values()))["calibration"].index) + " |",
              "|---|" + "---|" * len(next(iter(out.values()))["calibration"])]
    for c, r in out.items():
        cal = r["calibration"]
        lines.append(f"| {r['name']} | " + " | ".join(f"{pct(row['hit_rate'])} ({int(row['count'])})" for _, row in cal.iterrows()) + " |")
    lines += ["", "확신도가 높을수록 적중률이 오르면 확신도를 믿을 만하다. 괄호는 신호 수.", "",
              f"## 6. 지표별 {config.LABEL_HORIZON_DAYS}일 수익률 관련도 ({config.DEV_START} ~ 신호 끝, 운영 지표 {len(config.FEATURE_SET_BASE)}개 중 |IC| 상위 {config.TICKER_ANALYSIS_TOP_FEATURES}개)", "",
              "IC = 그날 지표 값과 5거래일 뒤 수익률의 순위 상관(t 에 알 수 있던 값), NW t = 5일 겹침 보정 t 값. |t| 가 2 보다 작으면 우연과 구분하기 어렵다. 분야 알고리즘(S4)의 추가 지표는 넣지 않았다.", ""]
    for c, r in out.items():
        ic = r["ic"].assign(abs_ic=r["ic"]["ic"].abs()).sort_values("abs_ic", ascending=False).head(config.TICKER_ANALYSIS_TOP_FEATURES)
        lines += [f"**{r['name']}** — " + ", ".join(f"`{x['feature']}`({x['group']}) {num(x['ic'], 3)} (t {num(x['nw_t'])})" for _, x in ic.iterrows()), ""]
    lines += ["### 지표 그룹별 평균 |IC|", "", "| 그룹 | " + " | ".join(r["name"] for r in out.values()) + " |", "|---|" + "---|" * len(out)]
    groups = sorted({g for r in out.values() for g in r["ic"]["group"]})
    for g in groups:
        vals = [r["ic"][r["ic"]["group"] == g]["ic"].abs().mean() for r in out.values()]
        lines.append(f"| {g} | " + " | ".join(num(v, 3) for v in vals) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---- 종목 추가 (10일차 모듈화) ----

def backfill(code: str, save: bool = True) -> pd.DataFrame:
    """레지스트리 종목의 과거 워크포워드 신호를 만든다(분석·화면·도구용). 분야 채택이 없는 종목은 S0 단독
    (운영 지표 46개, 거래정지일 제외, 매주 재학습, 정답 간격) — 8일차 S0 와 같은 계산이다.
    분야 알고리즘이 채택된 분야(예: 로봇 S4)는 같은 분야 종목을 함께 학습하므로 sector_models 로 만든다."""
    from src import tools
    from src.pipeline.sector_models import clean_raw
    if code not in config.MULTI_TICKERS:
        raise ValueError(f"{code} 는 종목 레지스트리에 없다. 먼저 config.MULTI_TICKERS 에 한 줄 넣는다")
    alg = tools.ticker_algorithm(code)
    if alg != "S0":
        raise ValueError(f"{code} 는 분야 알고리즘 {alg} 종목이다. python -m src.pipeline.sector_models run 으로 분야 전체를 만든다")
    raw = clean_raw(ticker_raw(code))
    table = features.build_features(raw=raw)
    labels = make_labels(raw["ohlcv"]["Close"].dropna().astype(float))
    sig = pooled_walk_forward({code: {"features": table, "labels": labels}}, list(config.FEATURE_SET_BASE), config.SECTOR_SELECT_PERIOD[0])[code]
    if save:
        path = Path(config.SECTOR_DIR)
        path.mkdir(parents=True, exist_ok=True)
        sig.to_parquet(path / f"signals_S0_{code}.parquet")
    return sig


def add(code: str) -> dict:
    """새 종목을 시스템에 넣는다: 원본 수집(어제까지) → 과거 신호(backfill) → 종목 분석·비교 다시 계산.
    실시간 신호·성적표·메일·도구·화면은 레지스트리를 따라 다음 실행부터 포함된다. 매매 대상은 config.TRADE_TICKERS 로 따로 정한다."""
    if code not in config.MULTI_TICKERS:
        raise ValueError(f"{code} 는 종목 레지스트리에 없다. 먼저 config.MULTI_TICKERS 에 (이름, 분야, 분야 ETF 또는 None) 한 줄을 넣는다")
    if code == config.MAIN_TICKER:
        raise ValueError("운영 종목은 기존 원본·운영 파이프라인을 쓴다")
    yesterday = str((pd.Timestamp.now(tz=config.KST).tz_localize(None).normalize() - pd.Timedelta(days=1)).date())
    collected = collect(end=yesterday, tickers=[code])
    sig = backfill(code)
    analysis()
    return {"code": code, "name": config.MULTI_TICKERS[code][0], "collected": collected,
            "signals": len(sig), "first": str(sig.index.min().date()), "last": str(sig.index.max().date()),
            "next": "다음 평일 15:20 자동 실행부터 실시간 신호·성적표·메일에 포함된다. 화면·/query 에서는 바로 조회된다"}


def refresh(today: pd.Timestamp | None = None) -> pd.DataFrame:
    """다른 종목 원본에 어제까지의 새 확정 날짜만 덧붙인다(오늘 장중 값은 저장하지 않는다)."""
    today = (pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz=config.KST).tz_localize(None)).normalize()
    return collect(end=str((today - pd.Timedelta(days=1)).date()), tickers=live_tickers())


def live_tickers() -> list[str]:
    """매일 덧붙일 종목: 다른 종목과, 채택된 분야 알고리즘이 함께 쓰는 같은 분야 종목(삼성전자 제외)."""
    from src.pipeline.sector_models import sector_of
    codes = [c for c in config.MULTI_TICKERS if c != config.MAIN_TICKER]
    peers = [p for c in config.MULTI_TICKERS if c in config.SECTOR_TICKER_NAMES and sector_of(c) in config.SECTOR_ADOPTED
             for p in config.SECTOR_UNIVERSE[sector_of(c)]]
    return list(dict.fromkeys(codes + [p for p in peers if p != config.MAIN_TICKER]))


def _live_one(code: str, as_of: str | None) -> dict:
    """채택된 분야 알고리즘이 있으면 그것으로, 없으면 단독 모델(운영 설정)로 신호를 만든다."""
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
    if code != config.MAIN_TICKER:
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
    for code in [c for c in config.MULTI_TICKERS if c != config.MAIN_TICKER]:
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
            from src.pipeline.scoreboard import realtime_mask
            old = pd.read_parquet(path)
            key = lambda df: pd.Series(list(zip(df.index, df["code"])), index=df.index)
            protected = set(key(old)[realtime_mask(old)])  # 장 마감 전 실시간 기록은 다시 실행해도 덮어쓰지 않는다(성적표 대상)
            new = new[~key(new).isin(protected)]
            keep = ~key(old).isin(set(key(new)))
            new = pd.concat([old[keep], new]).sort_index()
        new.to_parquet(path)
    return out


def summary_lines(results: list[dict]) -> list[str]:
    """메일용 요약(성적표 판정 대상, 매매·주문 대상 아님)."""
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
    elif cmd in ("add", "backfill") and len(sys.argv) > 2:
        if cmd == "add":
            print(add(sys.argv[2]))
        else:
            print(len(backfill(sys.argv[2])), "일 신호 저장:", config.SECTOR_DIR)
    elif cmd == "analysis":
        analysis()
        print("저장:", config.TICKER_ANALYSIS_REPORT_PATH, config.TICKER_ANALYSIS_DIR)
    elif cmd == "live":
        print("\n".join(summary_lines(live_signals(sys.argv[2] if len(sys.argv) > 2 else None))))
    else:
        print(__doc__)
