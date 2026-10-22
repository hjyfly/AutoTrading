import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.pipeline import sector_models as sm
from src.pipeline.labels import make_labels

DAYS = pd.bdate_range("2024-01-01", periods=200)


@pytest.fixture
def small(monkeypatch):
    """분야 하나(종목 A·B), 해외 시계열 f1, 국내 지수 k1 인 가짜 설정과 원본."""
    monkeypatch.setattr(config, "SECTOR_UNIVERSE", {"X": ["A", "B"]})
    monkeypatch.setattr(config, "SECTOR_INDEX", {"X": {"foreign": ["f1"], "kr": "k1"}})
    rng = np.random.default_rng(3)
    walk = lambda: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(DAYS)))), index=DAYS)
    a, b = walk(), walk()
    raw = {"ohlcv": pd.DataFrame({"Close": a, "Volume": 1.0}), "f1": pd.DataFrame({"value": walk()}),
           "k1": pd.DataFrame({"Close": walk()})}
    return raw, {"A": a, "B": b}


def test_분야_지수와_동료_지표는_t_이후_값이_바뀌어도_같다(small):
    raw, peers = small
    table = pd.DataFrame(index=DAYS)
    t = DAYS[150]
    before = sm.extra_features("A", table, raw, peers).loc[t]
    for s in [raw["ohlcv"], raw["f1"], raw["k1"], peers["A"], peers["B"]]:
        s.loc[s.index >= t] = s.loc[s.index >= t] * 3
    after = sm.extra_features("A", table, raw, peers).loc[t]
    pd.testing.assert_series_equal(before, after)
    assert not before.isna().any()


def test_동료_초과_수익은_t_1_종가까지다(small):
    raw, peers = small
    t = DAYS[150]
    row = sm.extra_features("A", pd.DataFrame(index=DAYS), raw, peers).loc[t]
    own = peers["A"].loc[DAYS[149]] / peers["A"].loc[DAYS[144]] - 1
    peer = peers["B"].loc[DAYS[149]] / peers["B"].loc[DAYS[144]] - 1
    assert row["peer_ret_5d"] == pytest.approx(peer)
    assert row["peer_excess_5d"] == pytest.approx(own - peer)


def test_변동성_맞춤_정답은_정답_간격이_같고_기준은_t_까지다(small):
    _, peers = small
    close = peers["A"]
    lv, base = sm.make_vol_labels(close), make_labels(close)
    assert (lv["future_date"] == base["future_date"].reindex(lv.index)).all()
    t = DAYS[120]
    changed = close.copy()
    changed.loc[changed.index > t] *= 2
    assert sm.make_vol_labels(changed).loc[t, "threshold"] == pytest.approx(lv.loc[t, "threshold"])
    up = lv["future_return"] >= lv["threshold"]
    assert (lv.loc[up, "label"] == "상승").all()


def test_거래정지일은_뺀다():
    days = pd.bdate_range("2024-01-01", periods=4)
    raw = {"ohlcv": pd.DataFrame({"Close": [1.0, 1.0, 1.0, 2.0], "Volume": [5, 0, 0, 5]}, index=days),
           "flows": pd.DataFrame({"foreign_net": [1, 2, 3, 4]}, index=days)}
    out = sm.clean_raw(raw)
    assert list(out["ohlcv"].index) == [days[0], days[3]] and list(out["flows"].index) == [days[0], days[3]]


def _ev(values: dict) -> pd.DataFrame:
    return pd.DataFrame([{"sector": "X", "code": code, "candidate": cand, "excess": v}
                         for (code, cand), v in values.items()])


def test_고르기_규칙():
    base = {(c, "S0"): 0.0 for c in "ABC"}
    # S1 평균 +2%p, 3종목 중 2종목에서 높음 → 고름
    v = sm.select_rule(_ev({**base, ("A", "S1"): 0.05, ("B", "S1"): 0.02, ("C", "S1"): -0.01}))
    assert v["X"]["chosen"] == "S1"
    # 평균 +0.5%p 는 기준(1%p) 미달 → S0 유지
    v = sm.select_rule(_ev({**base, ("A", "S1"): 0.015, ("B", "S1"): 0.0, ("C", "S1"): 0.0}))
    assert v["X"]["chosen"] == "S0"
    # 평균은 높지만 1종목만 높음 → S0 유지
    v = sm.select_rule(_ev({**base, ("A", "S1"): 0.09, ("B", "S1"): -0.01, ("C", "S1"): -0.01}))
    assert v["X"]["chosen"] == "S0"


def test_확인_규칙():
    base = {(c, "S0"): 0.0 for c in "ABC"}
    ev = _ev({**base, ("A", "S2"): 0.01, ("B", "S2"): 0.01, ("C", "S2"): -0.05})
    assert sm.confirm_rule(ev, {"X": "S2"})["X"]["adopted"] is False  # 과반은 높지만 평균이 낮음
    ev = _ev({**base, ("A", "S2"): 0.01, ("B", "S2"): 0.01, ("C", "S2"): -0.01})
    assert sm.confirm_rule(ev, {"X": "S2"})["X"]["adopted"] is True
    assert sm.confirm_rule(ev, {"X": "S0"})["X"]["adopted"] is False


def test_고르기_확인_규칙의_같은_값과_경계():
    base = {(c, "S0"): 0.0 for c in "ABC"}
    # 같은 값은 이긴 것으로 세지 않는다: 1종목만 높음 → S0 유지
    v = sm.select_rule(_ev({**base, ("A", "S1"): 0.0, ("B", "S1"): 0.05, ("C", "S1"): -0.01}))
    assert (v["X"]["chosen"], v["X"]["wins"]) == ("S0", 1)
    # 평균이 정확히 1%p 이고 2/3 → 고름(이상), 조금 모자라면 S0
    assert sm.select_rule(_ev({**base, ("A", "S1"): 0.02, ("B", "S1"): 0.01, ("C", "S1"): 0.0}))["X"]["chosen"] == "S1"
    assert sm.select_rule(_ev({**base, ("A", "S1"): 0.0197, ("B", "S1"): 0.01, ("C", "S1"): 0.0}))["X"]["chosen"] == "S0"
    # 확인: 한 종목이 같아도 나머지 두 종목이 높으면 2/3 채택
    ev = _ev({**base, ("A", "S2"): 0.0, ("B", "S2"): 0.02, ("C", "S2"): 0.01})
    assert sm.confirm_rule(ev, {"X": "S2"})["X"]["adopted"] is True


def test_후보_지표_열은_누적된다():
    sector = "로봇"
    s0, s1, s2, s3 = (sm.candidate_columns(c, sector) for c in ["S0", "S1", "S2", "S3"])
    assert s0 == list(config.FEATURE_SET_BASE)
    assert set(s0) < set(s1) < set(s2) < set(s3)
    assert sm.candidate_columns("S4", sector) == s3 == sm.candidate_columns("S5", sector)


def test_분야_모델은_주문_모듈과_연결하지_않는다():
    text = Path("src/pipeline/sector_models.py").read_text(encoding="utf-8")
    assert not re.search(r"\bbroker\b|kis_client", text)


def test_리포트는_확인_전에는_고르기만_싣는다(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DAY8_REPORT_PATH", str(tmp_path / "r.md"))
    ev = pd.DataFrame([{"sector": "X", "code": c, "name": c, "candidate": cand, "excess": 0.01, "accuracy": 0.5,
                        "most_common": 0.49, "balanced": 0.4, "return": 0.1, "mdd": -0.2, "calmar": 0.5,
                        "hold_return": 0.2, "hold_mdd": -0.3} for c in "AB" for cand in ["S0", "S1"]])
    sel = sm.select_rule(ev)
    text = sm.write_report(ev, sel)
    assert "## 1. 고르기" in text and "## 2. 확인" not in text
    text = sm.write_report(ev, sel, ev, sm.confirm_rule(ev, {"X": sel["X"]["chosen"]}))
    assert "## 2. 확인" in text


# 채택된 분야 알고리즘의 실시간 참고 신호 (D8-06)
from src.pipeline import multi_ticker as mt  # noqa: E402
from test_multi_ticker import CODE, DAYS as MT_DAYS, _flows, _ohlcv, stored  # noqa: E402,F401

PEER = "999991"


@pytest.fixture
def sector_live(stored, monkeypatch, tmp_path):
    """가짜 분야 X(CODE·PEER), 해외 f1·국내 지수 k1, 채택 알고리즘 S4."""
    rng = np.random.default_rng(11)
    _ohlcv(rng).to_parquet(tmp_path / f"{PEER}_ohlcv.parquet")
    _flows(rng).to_parquet(tmp_path / f"{PEER}_flows.parquet")
    walk = lambda: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(MT_DAYS))))
    pd.DataFrame({"value": walk()}, index=MT_DAYS).to_parquet(tmp_path / "sector_f1.parquet")
    pd.DataFrame({"Close": walk()}, index=MT_DAYS).to_parquet(tmp_path / "kr_k1.parquet")
    monkeypatch.setattr(config, "OPERATIONAL_EXTENDED_HISTORY", False)
    monkeypatch.setattr(config, "SECTOR_UNIVERSE", {"X": [CODE, PEER]})
    monkeypatch.setattr(config, "SECTOR_TICKER_NAMES", {CODE: "가짜", PEER: "가짜2"})
    monkeypatch.setattr(config, "SECTOR_INDEX", {"X": {"foreign": ["f1"], "kr": "k1"}})
    monkeypatch.setattr(config, "SECTOR_ETFS", {"f1": "F1"})
    monkeypatch.setattr(config, "SECTOR_KR_INDEX", {"k1": "K1"})
    monkeypatch.setattr(config, "SECTOR_ADOPTED", {"X": "S4"})
    shared = {**stored["shared"], "sector_f1": pd.read_parquet(tmp_path / "sector_f1.parquet"),
              "k1": pd.read_parquet(tmp_path / "kr_k1.parquet")}
    return shared


def test_분야_알고리즘_실시간_신호는_과거_검증과_같다(sector_live):
    tables = sm.build_sector_tables(sector_live)
    cols = sm.candidate_columns("S4", "X")
    wf = mt.pooled_walk_forward({c: {"features": tables[c]["features"], "labels": tables[c]["labels"]} for c in [CODE, PEER]},
                                cols, MT_DAYS[300])[CODE]
    for t in [MT_DAYS[300], MT_DAYS[333], MT_DAYS[400]]:
        live = sm.live_signal(CODE, str(t.date()), valid_start=MT_DAYS[300])
        assert live["signal"] == wf.loc[t, "signal"]
        assert live["confidence"] == pytest.approx(wf.loc[t, "confidence"])
        assert (live["retrained_on"], live["train_rows"]) == (wf.loc[t, "retrained_on"], wf.loc[t, "train_rows"])
        assert live["model"].startswith("S4")


def test_분야_알고리즘_실시간_신호는_t_이후_동료_값을_쓰지_않는다(sector_live):
    t = MT_DAYS[333]
    before = sm.live_signal(CODE, str(t.date()))
    for name in [f"{PEER}_ohlcv", f"{PEER}_flows", "sector_f1", "kr_k1"]:
        path = Path(config.TICKER_RAW_DIR) / f"{name}.parquet"
        df = pd.read_parquet(path)
        df.loc[df.index >= t] = df.loc[df.index >= t] * 3
        df.to_parquet(path)
    after = sm.live_signal(CODE, str(t.date()))
    assert (before["signal"], before["confidence"]) == (after["signal"], pytest.approx(after["confidence"]))


def test_채택_분야_종목은_분야_알고리즘으로_매일_동료까지_받는다(sector_live, monkeypatch):
    assert mt.live_tickers() == [CODE, PEER]
    monkeypatch.setattr(config, "SECTOR_ADOPTED", {})
    assert mt.live_tickers() == [CODE]
    assert mt._live_one(CODE, str(MT_DAYS[333].date()))["model"] == "S0 단독"


def test_분야_알고리즘은_동료_원본이_직전_거래일까지_없으면_멈춘다(sector_live):
    t = MT_DAYS[333]
    path = Path(config.TICKER_RAW_DIR) / f"{PEER}_ohlcv.parquet"
    df = pd.read_parquet(path)
    df[df.index < MT_DAYS[331]].to_parquet(path)
    with pytest.raises(ValueError, match="직전 거래일"):
        sm.live_signal(CODE, str(t.date()))


def test_분야_알고리즘은_국내_지수가_묵으면_멈춘다(sector_live):
    t = MT_DAYS[333]
    path = Path(config.TICKER_RAW_DIR) / "kr_k1.parquet"
    df = pd.read_parquet(path)
    df[df.index < MT_DAYS[331]].to_parquet(path)
    with pytest.raises(ValueError, match="kr_k1"):
        sm.live_signal(CODE, str(t.date()))
