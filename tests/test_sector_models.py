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
