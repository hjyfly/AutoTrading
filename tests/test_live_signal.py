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
    for name in {**config.YF_TICKERS, **config.YF_TICKERS_EXT}:
        data[("macro", name)] = pd.DataFrame({"value": 100 + np.cumsum(rng.normal(0, 1, len(DAYS)))}, index=DAYS)
    for name in config.KR_EXTRA:
        data[("prices", name)] = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(DAYS))))}, index=DAYS)
    monkeypatch.setattr(tools, "_load_raw", lambda kind, name: data.get((kind, name), pd.DataFrame(index=pd.DatetimeIndex([]))))
    monkeypatch.setattr(tools, "_live", lambda *a: (_ for _ in ()).throw(RuntimeError("실시간 금지")))
    monkeypatch.setattr(config, "LGBM_PARAMS", {**config.LGBM_PARAMS, "n_estimators": 20, "min_child_samples": 5})
    monkeypatch.setattr(config, "MIN_TRAIN_ROWS", 30)
    return data


@pytest.mark.parametrize("feature_set", ["base", "candidate", "groups:price+chart+flow+macro+peer"])
def test_저장된_날짜는_과거_검증_신호와_같다(stored, feature_set):
    """실시간 경로와 워크포워드 경로가 같은 데이터·같은 규칙·같은 지표 묶음이면 같은 신호·확률이 나와야 한다."""
    from src.pipeline import features
    raw = {"ohlcv": stored[("prices", "samsung_ohlcv")], "flows": stored[("prices", "samsung_investor_flows")],
           **{n: stored[("macro", n)] for n in {**config.YF_TICKERS, **config.YF_TICKERS_EXT}},
           **{n: stored[("prices", n)] for n in config.KR_EXTRA}}
    table = features.build_features(raw=raw)
    labels = make_labels(raw["ohlcv"]["Close"])
    wf = tr.walk_forward(table, labels, valid_start=str(DAYS[300].date()), feature_set=feature_set)
    for t in [DAYS[300], DAYS[333], DAYS[400]]:
        live = ls.numeric_signal(str(t.date()), valid_start=str(DAYS[300].date()), feature_set=feature_set)
        assert live["signal"] == wf.loc[t, "signal"]
        assert live["confidence"] == pytest.approx(wf.loc[t, "confidence"])
        assert live["retrained_on"] == wf.loc[t, "retrained_on"]


@pytest.mark.parametrize("feature_set", ["base", "groups:price+chart+flow+macro+peer"])
def test_미래_데이터가_바뀌어도_신호는_같다(stored, feature_set):
    t = DAYS[333]
    before = ls.numeric_signal(str(t.date()), feature_set=feature_set)
    for df in stored.values():
        df.loc[df.index >= t] = df.loc[df.index >= t] * 3
    after = ls.numeric_signal(str(t.date()), feature_set=feature_set)
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


def test_뉴스_없이_다시_실행해도_이전_뉴스_보정을_유지한다(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "LIVE_PATH", tmp_path / "live.parquet")
    t = pd.Timestamp("2026-10-21")
    ls.save({"date": t, "model": "base", "signal": "상승", "numeric_signal_label": "보합", "news_consensus": 4.0, "news_consensus_conf": 0.8})
    monkeypatch.setattr(ls, "numeric_signal", lambda *a, **k: {"date": t, "model": tools.horizon_model(k["horizon"]) if k.get("horizon") else k.get("feature_set", "base"), "signal": "보합",
                                                               "confidence": 0.5, "data_sources": {}})
    monkeypatch.setattr(ls, "risk_signal", lambda *a: (_ for _ in ()).throw(ValueError("시험에서는 건너뛴다")))
    monkeypatch.setattr(config, "NEWS_ADJUST_IN_SIGNAL", True)
    out = ls.make_signal("2026-10-21", with_news=False)
    assert out["news_consensus"] == 4.0 and out["signal"] == "상승"
    monkeypatch.setattr(config, "NEWS_ADJUST_IN_SIGNAL", False)  # 2026-10-22 부터: 뉴스는 참고용, 신호는 숫자 모델 그대로
    out = ls.make_signal("2026-10-21", with_news=False)
    assert out["news_consensus"] == 4.0 and out["signal"] == "보합"


@pytest.mark.parametrize("freq", ["W", "M"])
def test_실시간_재학습일은_과거_검증_덩어리와_같다(freq):
    days = pd.bdate_range("2025-04-01", "2025-08-29").drop(pd.Timestamp("2025-06-02"))  # 월요일 휴장 포함
    valid_start = "2025-04-23"  # 수요일
    chunks = tr.prediction_chunks(days, valid_start, freq)
    expected = {d: start for start, block in chunks for d in block}
    for t, start in expected.items():
        assert ls.block_start(t, days, valid_start, freq) == start


def test_하락_위험_비교_신호는_시점_규칙을_지킨다(stored):
    t = DAYS[333]
    before = ls.risk_signal(str(t.date()))
    assert before["train_last_future_date"] < before["retrained_on"] <= t
    for df in stored.values():
        df.loc[df.index >= t] = df.loc[df.index >= t] * 3
    after = ls.risk_signal(str(t.date()))
    assert (before["weight"], before["p_risk"]) == (after["weight"], pytest.approx(after["p_risk"]))


def test_확장_원본은_앞에만_붙이고_날짜가_겹치면_멈춘다(monkeypatch):
    days = pd.bdate_range("2026-01-05", periods=4)
    raw = {"ohlcv": pd.DataFrame({"Close": [1.0, 2, 3, 4]}, index=days)}
    before = pd.DataFrame({"Close": [9.0, 9.0]}, index=pd.bdate_range("2026-01-01", periods=2))
    overlap = pd.DataFrame({"Close": [9.0, 9.0]}, index=days[:2])
    for ext, expected in [(before, 6), (overlap, None)]:  # 앞 날짜만 있으면 붙이고, 겹치면 멈춘다
        monkeypatch.setattr(tools, "_load_raw", lambda kind, name, ext=ext: ext if name.endswith(config.EXT_SUFFIX) else pd.DataFrame())
        if expected is None:
            with pytest.raises(ValueError):
                ls.prepend_extended(raw)
        else:
            assert len(ls.prepend_extended(raw)["ohlcv"]) == expected


def test_뉴스_채점이_실패해도_운영_신호는_남는다(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "LIVE_PATH", tmp_path / "live.parquet")
    t = pd.Timestamp("2026-10-21")
    monkeypatch.setattr(ls, "numeric_signal", lambda *a, **k: {"date": t, "model": tools.horizon_model(k["horizon"]) if k.get("horizon") else k.get("feature_set", "base"), "signal": "상승",
                                                               "confidence": 0.5, "data_sources": {}})
    monkeypatch.setattr(ls, "risk_signal", lambda *a: (_ for _ in ()).throw(ValueError("시험에서는 건너뛴다")))
    monkeypatch.setattr(ls, "news_ready", lambda t: True)
    monkeypatch.setattr(ls, "news_score", lambda t: (_ for _ in ()).throw(RuntimeError("Too many tokens per day")))
    out = ls.make_signal("2026-10-21", with_news=True)
    assert out["signal"] == "상승" and "뉴스 채점 실패" in out["news_status"]
    saved = pd.read_parquet(tmp_path / "live.parquet")
    assert "상승" in saved[saved["model"] == "base"]["signal"].tolist()


@pytest.mark.parametrize("n", [1, 20])
def test_기간별_실시간_신호는_같은_기간_과거_검증과_같다(stored, n):
    """11일차: 기간 N 실시간 신호도 그 기간 정답으로 학습한 워크포워드와 같은 신호·확률·재학습일이어야 한다."""
    from src.pipeline import features
    raw = {"ohlcv": stored[("prices", "samsung_ohlcv")], "flows": stored[("prices", "samsung_investor_flows")],
           **{k: stored[("macro", k)] for k in {**config.YF_TICKERS, **config.YF_TICKERS_EXT}},
           **{k: stored[("prices", k)] for k in config.KR_EXTRA}}
    table = features.build_features(raw=raw)
    wf = tr.walk_forward(table, make_labels(raw["ohlcv"]["Close"], n, config.HORIZONS[n]), valid_start=str(DAYS[300].date()), feature_set="base")
    for t in [DAYS[300], DAYS[333], DAYS[400]]:
        live = ls.numeric_signal(str(t.date()), valid_start=str(DAYS[300].date()), horizon=n)
        assert live["model"] == f"h{n}" and live["horizon"] == n
        assert live["signal"] == wf.loc[t, "signal"] and live["confidence"] == pytest.approx(wf.loc[t, "confidence"])
        assert live["retrained_on"] == wf.loc[t, "retrained_on"] and live["train_last_future_date"] < live["retrained_on"]


def test_기간별_신호는_h_모델로_따로_기록되고_운영_신호를_덮지_않는다(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "LIVE_PATH", tmp_path / "live.parquet")
    t = pd.Timestamp("2026-10-21")
    sig = {None: "상승", 1: "하락", 3: "보합", 10: "하락", 20: "상승"}
    monkeypatch.setattr(ls, "numeric_signal", lambda *a, **k: {"date": t, "model": tools.horizon_model(k["horizon"]) if k.get("horizon") else k.get("feature_set", "base"),
                                                               "horizon": k.get("horizon") or 5, "signal": sig.get(k.get("horizon"), "보합"), "confidence": 0.5, "data_sources": {}})
    monkeypatch.setattr(ls, "risk_signal", lambda *a: (_ for _ in ()).throw(ValueError("시험에서는 건너뛴다")))
    ls.make_signal("2026-10-21")
    saved = pd.read_parquet(tmp_path / "live.parquet")
    by_model = dict(zip(saved["model"], saved["signal"]))
    assert by_model["base"] == "상승" and by_model["h1"] == "하락" and by_model["h20"] == "상승" and "h5" not in by_model
    assert saved.set_index("model").loc["h20", "model_config"].startswith("지표 base") and "정답 20거래일 ±6.0%" in saved.set_index("model").loc["h20", "model_config"]
