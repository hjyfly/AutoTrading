from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, tools
from src.pipeline import live_signal as ls
from src.pipeline import train as tr
from src.pipeline.labels import make_labels

DAYS = pd.bdate_range("2024-01-01", periods=420)


@pytest.fixture
def stored(monkeypatch):
    """저장된 원본 대신 가짜 데이터, 실시간 조회는 금지. 작은 LightGBM."""
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(DAYS)))), index=DAYS)
    data = {
        ("prices", "samsung_ohlcv"): pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                                                   "Volume": rng.integers(1000, 2000, len(DAYS)).astype(float)}, index=DAYS),
        ("prices", "samsung_investor_flows"): pd.DataFrame({"foreign_net": rng.normal(0, 100, len(DAYS)),
                                                            "institution_net": rng.normal(0, 100, len(DAYS)),
                                                            "individual_net": rng.normal(0, 100, len(DAYS)),
                                                            "foreign_hold_ratio": 50 + rng.normal(0, 0.1, len(DAYS))}, index=DAYS),
    }
    for name in config.YF_TICKERS:
        data[("macro", name)] = pd.DataFrame({"value": 100 + np.cumsum(rng.normal(0, 1, len(DAYS)))}, index=DAYS)
    monkeypatch.setattr(tools, "_load_raw", lambda kind, name: data[(kind, name)])
    monkeypatch.setattr(tools, "_live", lambda *a: (_ for _ in ()).throw(RuntimeError("실시간 금지")))
    monkeypatch.setattr(config, "LGBM_PARAMS", {**config.LGBM_PARAMS, "n_estimators": 20, "min_child_samples": 5})
    monkeypatch.setattr(config, "MIN_TRAIN_ROWS", 30)
    return data


def test_저장된_날짜는_과거_검증_신호와_같다(stored):
    """실시간 경로와 워크포워드 경로가 같은 데이터·같은 규칙이면 같은 신호·확률이 나와야 한다."""
    from src.pipeline import features
    raw = {"ohlcv": stored[("prices", "samsung_ohlcv")], "flows": stored[("prices", "samsung_investor_flows")],
           **{n: stored[("macro", n)] for n in config.YF_TICKERS}}
    table = features.build_features(raw=raw)
    labels = make_labels(raw["ohlcv"]["Close"])
    wf = tr.walk_forward(table, labels, valid_start=str(DAYS[300].date()))
    for t in [DAYS[300], DAYS[333], DAYS[400]]:
        live = ls.numeric_signal(str(t.date()), valid_start=str(DAYS[300].date()))
        assert live["signal"] == wf.loc[t, "signal"]
        assert live["confidence"] == pytest.approx(wf.loc[t, "confidence"])
        assert live["retrained_on"] == wf.loc[t, "retrained_on"]


def test_미래_데이터가_바뀌어도_신호는_같다(stored):
    t = DAYS[333]
    before = ls.numeric_signal(str(t.date()))
    for df in stored.values():
        df.loc[df.index >= t] = df.loc[df.index >= t] * 3
    after = ls.numeric_signal(str(t.date()))
    assert (before["signal"], before["confidence"]) == (after["signal"], pytest.approx(after["confidence"]))


def test_주말은_만들지_않는다(stored):
    with pytest.raises(ValueError):
        ls.numeric_signal("2025-03-08")


def test_뉴스는_15시15분_이후에만():
    t = pd.Timestamp("2026-10-21")
    assert not ls.news_ready(t, pd.Timestamp("2026-10-21 15:14", tz=config.KST))
    assert ls.news_ready(t, pd.Timestamp("2026-10-21 15:15", tz=config.KST))


def test_결과는_날짜별로_덮어쓴다(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "LIVE_PATH", tmp_path / "live.parquet")
    ls.save({"date": pd.Timestamp("2026-10-21"), "signal": "보합"})
    ls.save({"date": pd.Timestamp("2026-10-21"), "signal": "상승"})
    ls.save({"date": pd.Timestamp("2026-10-22"), "signal": "하락"})
    saved = pd.read_parquet(tmp_path / "live.parquet")
    assert saved["signal"].tolist() == ["상승", "하락"]
