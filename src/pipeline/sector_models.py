"""8일차 분야별 알고리즘 (DAY8_PLAN.md, SERVICE2.md). 분석·참고 신호만 만들고 매매·주문 경로와 연결하지 않는다.

- 분야(반도체·우주항공·로봇)마다 종목 3개로 후보 S0 ~ S5 를 같은 규칙(매주 재학습, 정답 간격)으로 워크포워드한다.
- 고르는 구간에서 분야마다 후보를 하나 고르고(select), 확인 구간에서 한 번 확인한다(confirm). 규칙은 사전 등록 그대로.
- 추가 지표의 시점 규칙: 국내 지수·동료 종목·해외 시계열 모두 '그 시계열 날짜 < t' 인 가장 최근 값만 쓴다(t-1 종가까지).

명령줄: python -m src.pipeline.sector_models run | select | confirm
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.pipeline import features
from src.pipeline import multi_ticker as mt
from src.pipeline.labels import DOWN, FLAT, UP, make_labels
from src.pipeline.train import CLASSES

KR_PREFIX = "kx"      # 분야 기준 국내 지수 열 앞부분(코스피 또는 코스닥)
PEER_PREFIX = "peer"  # 같은 분야 동료 평균 열 앞부분


def sector_of(code: str) -> str:
    """종목의 분야 이름."""
    return next(s for s, codes in config.SECTOR_UNIVERSE.items() if code in codes)


def clean_raw(raw: dict) -> dict:
    """거래량 0 인 날(거래정지)을 일봉·수급에서 뺀다(사전 등록한 공통 데이터 정리)."""
    out = dict(raw)
    ohlcv = raw["ohlcv"]
    out["ohlcv"] = ohlcv[ohlcv["Volume"].fillna(0) > 0]
    out["flows"] = raw["flows"][raw["flows"].index.isin(out["ohlcv"].index)]
    return out


def shared_raw() -> dict:
    """공통 원본(확장 포함)에 분야 지수 원본(해외 ETF·종목, 코스닥)을 더한다."""
    raw = mt.base_raw()
    for name in config.SECTOR_ETFS:
        raw[f"sector_{name}"] = pd.read_parquet(mt.raw_path(f"sector_{name}"))
    for name in config.SECTOR_KR_INDEX:
        raw[name] = pd.read_parquet(mt.raw_path(f"kr_{name}"))
    return raw


def series_of(raw: dict, name: str) -> pd.Series:
    """이름으로 시계열(종가 또는 값)을 꺼낸다. 공통 원본 이름이 먼저, 없으면 분야 ETF 원본."""
    df = raw[name] if name in raw else raw[f"sector_{name}"]
    col = "value" if "value" in df.columns else "Close"
    return df[col].astype(float).dropna().sort_index()


def lagged_returns(series: pd.Series, days: pd.DatetimeIndex, windows: list[int], prefix: str) -> pd.DataFrame:
    """t 마다 그 시계열 날짜가 t 보다 앞선(같은 날 제외) 가장 최근 w일 수익률을 붙인다. 시점 검사를 거친다."""
    feats = pd.DataFrame({f"{prefix}_ret_{w}d": series.pct_change(w) for w in windows})
    feats[f"{prefix}_src_date"] = feats.index
    attached = features.attach_foreign(days, feats, prefix)
    features.assert_no_lookahead(attached)
    return attached.drop(columns=[f"{prefix}_src_date", f"{prefix}_age_days"])


def index_columns(sector: str) -> list[str]:
    """S1 분야 지수 지표 열."""
    spec = config.SECTOR_INDEX[sector]
    return ([f"sx_{n}_ret_{w}d" for n in spec["foreign"] for w in config.FOREIGN_CHANGE_WINDOWS]
            + [f"{KR_PREFIX}_ret_{w}d" for w in config.SECTOR_PEER_WINDOWS]
            + [f"{KR_PREFIX}_rel_{w}d" for w in config.SECTOR_PEER_WINDOWS])


def peer_columns() -> list[str]:
    """S2 동료 지표 열."""
    return ([f"{PEER_PREFIX}_ret_{w}d" for w in config.SECTOR_PEER_WINDOWS]
            + [f"{PEER_PREFIX}_excess_{w}d" for w in config.SECTOR_PEER_WINDOWS])


def candidate_columns(candidate: str, sector: str) -> list[str]:
    """후보마다 쓰는 지표 열(사전 등록 그대로)."""
    cols = list(config.FEATURE_SET_BASE)
    if candidate == "S0":
        return cols
    cols += index_columns(sector)
    if candidate == "S1":
        return cols
    cols += peer_columns()
    if candidate == "S2":
        return cols
    return cols + list(config.SECTOR_TECH_FEATURES)  # S3·S4·S5


def own_return(close: pd.Series, days: pd.DatetimeIndex, w: int) -> pd.Series:
    """그 종목의 t-1 종가까지 w일 수익률(국내 지표와 같은 규칙: 그날 종가 수익률을 하루 밀어 붙인다)."""
    return close.pct_change(w).reindex(days).shift(1)


def extra_features(code: str, table: pd.DataFrame, raw: dict, peer_closes: dict[str, pd.Series], as_of=None) -> pd.DataFrame:
    """분야 지수·동료 지표를 만든다. as_of 를 주면 그 날 전 값만 쓴다."""
    sector = sector_of(code)
    days = table.index
    cut = (lambda s: s[s.index < pd.Timestamp(as_of)]) if as_of is not None else (lambda s: s)
    close = cut(raw["ohlcv"]["Close"].astype(float).dropna())
    spec = config.SECTOR_INDEX[sector]
    parts = [lagged_returns(cut(series_of(raw, n)), days, list(config.FOREIGN_CHANGE_WINDOWS), f"sx_{n}") for n in spec["foreign"]]
    kr = lagged_returns(cut(series_of(raw, spec["kr"])), days, list(config.SECTOR_PEER_WINDOWS), KR_PREFIX)
    for w in config.SECTOR_PEER_WINDOWS:
        kr[f"{KR_PREFIX}_rel_{w}d"] = own_return(close, days, w) - kr[f"{KR_PREFIX}_ret_{w}d"]
    parts.append(kr)
    peers = [lagged_returns(cut(c), days, list(config.SECTOR_PEER_WINDOWS), f"p{i}")
             for i, (pc, c) in enumerate(sorted(peer_closes.items())) if pc != code]
    peer = pd.DataFrame(index=days)
    for w in config.SECTOR_PEER_WINDOWS:
        cols = [p.filter(like=f"_ret_{w}d").iloc[:, 0] for p in peers]
        avg = pd.concat(cols, axis=1).mean(axis=1) if cols else pd.Series(np.nan, index=days)  # 상장 전 동료는 빼고 평균
        peer[f"{PEER_PREFIX}_ret_{w}d"] = avg
        peer[f"{PEER_PREFIX}_excess_{w}d"] = own_return(close, days, w) - avg
    parts.append(peer)
    return pd.concat(parts, axis=1)


def make_vol_labels(close: pd.Series, k: float | None = None, window: int | None = None, horizon: int | None = None) -> pd.DataFrame:
    """S5 정답: 기준을 ±(k × 지난 window 거래일 일간 변동성 × √horizon)로 종목마다·날마다 맞춘다.
    변동성은 t 종가까지(신호 체결 시점에 아는 값)로 계산한다. 정답은 학습 입력에 넣지 않는다."""
    k = config.SECTOR_VOL_LABEL_K if k is None else k
    window = config.SECTOR_VOL_WINDOW if window is None else window
    horizon = config.LABEL_HORIZON_DAYS if horizon is None else horizon
    close = close.sort_index()
    base = make_labels(close, horizon=horizon)
    threshold = (k * close.pct_change().rolling(window).std() * np.sqrt(horizon)).reindex(base.index)
    ret = base["future_return"]
    label = pd.Series(FLAT, index=base.index)
    label[ret >= threshold] = UP
    label[ret <= -threshold] = DOWN
    out = base.assign(label=label, threshold=threshold)
    return out[threshold.notna()]


def build_sector_tables(shared: dict | None = None) -> dict[str, dict]:
    """9종목의 특징 표(기본 + 분야 지수 + 동료 + 기술), 정답(±3%, 변동성 맞춤), 종가."""
    shared = shared or shared_raw()
    raws = {code: clean_raw(mt.ticker_raw(code, shared)) for code in config.SECTOR_TICKER_NAMES}
    closes = {code: r["ohlcv"]["Close"].astype(float) for code, r in raws.items()}
    out = {}
    for code, raw in raws.items():
        table = features.build_features(raw=raw)
        sector = sector_of(code)
        peers = {c: closes[c] for c in config.SECTOR_UNIVERSE[sector]}
        table = table.join(extra_features(code, table, raw, peers))
        out[code] = {"features": table, "labels": make_labels(closes[code]), "labels_vol": make_vol_labels(closes[code]),
                     "close": closes[code]}
    return out


def run_candidates(tables: dict[str, dict], start=None) -> dict[str, dict[str, pd.DataFrame]]:
    """후보 S0 ~ S5 워크포워드. 결과: {후보: {종목: 신호 표}}. S4 는 분야 종목을 합쳐 학습한다."""
    start = pd.Timestamp(start or config.SECTOR_SELECT_PERIOD[0])
    out = {c: {} for c in config.SECTOR_CANDIDATES}
    for sector, codes in config.SECTOR_UNIVERSE.items():
        for cand in config.SECTOR_CANDIDATES:
            cols = candidate_columns(cand, sector)
            key = "labels_vol" if cand == "S5" else "labels"
            if cand == "S4":
                group = {c: {"features": tables[c]["features"], "labels": tables[c]["labels"]} for c in codes}
                out[cand].update(mt.pooled_walk_forward(group, cols, start))
                continue
            for code in codes:
                single = {code: {"features": tables[code]["features"], "labels": tables[code][key]}}
                out[cand].update(mt.pooled_walk_forward(single, cols, start))
    return out


def balanced_accuracy(sig: pd.Series, known: pd.Series) -> float:
    """세 정답의 재현율 평균(그 구간에 나온 정답만)."""
    recalls = [float((sig[known == c] == c).mean()) for c in CLASSES if (known == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def evaluate(signals: dict[str, dict[str, pd.DataFrame]], tables: dict[str, dict], start, end) -> pd.DataFrame:
    """한 구간의 후보·종목별 성과. 정확도 비교는 그 후보의 정답(S5 는 변동성 맞춤) 기준이다."""
    rows = []
    for cand, by_code in signals.items():
        for code, sig in by_code.items():
            t = tables[code]
            labels = t["labels_vol"] if cand == "S5" else t["labels"]
            r = mt.period_report(sig["signal"], t["close"], labels, start, end)
            s = sig["signal"].loc[start:end]
            known = labels[labels["future_date"] <= s.index.max()]["label"].reindex(s.index).dropna()
            rows.append({"sector": sector_of(code), "code": code, "name": config.SECTOR_TICKER_NAMES[code], "candidate": cand,
                         "excess": r["accuracy"] - r["most_common"], "balanced": balanced_accuracy(s.reindex(known.index), known),
                         **r})
    return pd.DataFrame(rows)


def select_rule(ev: pd.DataFrame) -> dict[str, dict]:
    """고르기(사전 등록): 분야 평균 초과 정확도 최고 후보. S0 보다 평균 SECTOR_SELECT_MIN_GAIN 이상 높고 종목 과반에서 높아야 한다."""
    out = {}
    for sector, g in ev.groupby("sector"):
        piv = g.pivot(index="code", columns="candidate", values="excess")
        means = piv.mean()
        best = means.drop("S0").idxmax()
        wins = int((piv[best] > piv["S0"]).sum())
        ok = means[best] - means["S0"] >= config.SECTOR_SELECT_MIN_GAIN and wins > len(piv) / 2
        out[sector] = {"chosen": best if ok else "S0", "best": best, "gain": float(means[best] - means["S0"]), "wins": wins,
                       "n": len(piv), "means": means.round(4).to_dict()}
    return out


def confirm_rule(ev: pd.DataFrame, chosen: dict[str, str]) -> dict[str, dict]:
    """확인(사전 등록, 한 번): 고른 후보가 S0 보다 분야 평균 초과 정확도가 높고 종목 과반에서 높으면 채택."""
    out = {}
    for sector, g in ev.groupby("sector"):
        cand = chosen[sector]
        if cand == "S0":
            out[sector] = {"candidate": "S0", "adopted": False, "note": "고르기에서 S0 유지"}
            continue
        piv = g.pivot(index="code", columns="candidate", values="excess")
        wins = int((piv[cand] > piv["S0"]).sum())
        gain = float(piv[cand].mean() - piv["S0"].mean())
        out[sector] = {"candidate": cand, "adopted": bool(gain > 0 and wins > len(piv) / 2), "gain": gain, "wins": wins, "n": len(piv)}
    return out


def _dir() -> Path:
    path = Path(config.SECTOR_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run() -> None:
    """모든 후보 신호를 계산해 저장한다(성과는 계산하지 않는다)."""
    tables = build_sector_tables()
    signals = run_candidates(tables)
    for cand, by_code in signals.items():
        for code, sig in by_code.items():
            sig.to_parquet(_dir() / f"signals_{cand}_{code}.parquet")


def load_signals() -> dict[str, dict[str, pd.DataFrame]]:
    """저장한 후보 신호를 읽는다."""
    return {cand: {code: pd.read_parquet(_dir() / f"signals_{cand}_{code}.parquet") for code in config.SECTOR_TICKER_NAMES
                   if (_dir() / f"signals_{cand}_{code}.parquet").exists()} for cand in config.SECTOR_CANDIDATES}


def select() -> dict:
    """고르는 구간만 평가하고 분야마다 후보를 고른다. 결과를 select.csv 에 남긴다."""
    tables = build_sector_tables()
    ev = evaluate(load_signals(), tables, *config.SECTOR_SELECT_PERIOD)
    ev.to_csv(_dir() / "select.csv", index=False, encoding="utf-8-sig")
    return select_rule(ev)


def confirm() -> dict:
    """확인 구간을 한 번 평가한다. 고르기 결과는 select.csv 에서 다시 계산한다."""
    tables = build_sector_tables()
    chosen = {s: v["chosen"] for s, v in select_rule(pd.read_csv(_dir() / "select.csv", dtype={"code": str})).items()}
    signals = load_signals()  # 확인 구간은 고른 후보와 S0 만 평가한다(다른 후보 성적을 보고 갈아타지 않게)
    keep = {cand: {c: s for c, s in by_code.items() if cand == "S0" or chosen[sector_of(c)] == cand}
            for cand, by_code in signals.items()}
    ev = evaluate(keep, tables, config.SECTOR_CONFIRM_START, None)
    ev.to_csv(_dir() / "confirm.csv", index=False, encoding="utf-8-sig")
    return confirm_rule(ev, chosen)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run()
        print("저장:", config.SECTOR_DIR)
    elif cmd == "select":
        print(select())
    elif cmd == "confirm":
        print(confirm())
    else:
        print(__doc__)
