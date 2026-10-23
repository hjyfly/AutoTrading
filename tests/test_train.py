import numpy as np
import pandas as pd
import pytest

from src import config
from src.pipeline import train as tr
from src.pipeline.labels import make_labels


@pytest.fixture(autouse=True)
def small_model(monkeypatch):
    """테스트는 작은 LightGBM 으로 빠르게 돈다."""
    monkeypatch.setattr(config, "LGBM_PARAMS", {**config.LGBM_PARAMS, "n_estimators": 20, "min_child_samples": 5})
    monkeypatch.setattr(config, "MIN_TRAIN_ROWS", 30)


def make_data(n=200, seed=0):
    """무작위 가격과 그 가격에서 만든 정답, 무작위 특징."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=days)
    labels = make_labels(close)
    features = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)}, index=days)
    return features, labels


def test_학습_정답은_예측_시작일_전에_끝난다():
    features, labels = make_data()
    out = tr.walk_forward(features, labels, valid_start="2025-05-01")
    assert (out["train_last_future_date"] < out["retrained_on"]).all()
    assert out["retrained_on"].nunique() == out.index.to_period(config.RETRAIN_FREQ).nunique()


def test_미래_데이터를_바꿔도_과거_예측은_같다():
    features, labels = make_data()
    base = tr.walk_forward(features, labels, valid_start="2025-05-01")
    cut = pd.Timestamp("2025-08-01")
    f2, l2 = features.copy(), labels.copy()
    f2.loc[f2.index >= cut] *= -5
    l2.loc[l2.index >= cut, "label"] = "하락"
    changed = tr.walk_forward(f2, l2, valid_start="2025-05-01")
    before = base.index < cut
    pd.testing.assert_frame_equal(base[before], changed[before])


def test_정답_기간이_재학습일을_넘는_행은_학습에서_빠진다():
    """재학습일 R 직전 5거래일(날짜 < R ≤ 정답 끝날)의 정답만 바꿔도 R 달 예측은 같아야 한다(purge 회귀 방지)."""
    features, labels = make_data()
    base = tr.walk_forward(features, labels, valid_start="2025-05-01")
    r = base["retrained_on"].unique()[2]
    straddle = (labels.index < r) & (labels["future_date"] >= r)
    assert straddle.sum() == config.LABEL_HORIZON_DAYS
    flipped = labels.copy()
    flipped.loc[straddle, "label"] = "하락"
    changed = tr.walk_forward(features, flipped, valid_start="2025-05-01")
    month = base["retrained_on"] == r
    pd.testing.assert_frame_equal(base[month], changed[month])


def test_학습_행이_모자라면_멈춘다(monkeypatch):
    features, labels = make_data()
    monkeypatch.setattr(config, "MIN_TRAIN_ROWS", 10_000)
    with pytest.raises(ValueError):
        tr.walk_forward(features, labels, valid_start="2025-05-01")


def test_확률_합은_1이고_신호는_가장_큰_확률():
    features, labels = make_data()
    out = tr.walk_forward(features, labels, valid_start="2025-05-01")
    probs = out[[f"p_{c}" for c in tr.CLASSES]]
    assert np.allclose(probs.sum(axis=1), 1)
    assert (probs.idxmax(axis=1).str[2:] == out["signal"]).all()


def test_정답_열이_특징에_섞이면_멈춘다():
    features, labels = make_data()
    with pytest.raises(AssertionError):
        tr.walk_forward(features.assign(label_leak=1), labels, valid_start="2025-05-01")


def test_실제_데이터에서_기존과_후보_모델의_학습_조건이_같다():
    """후보 지표의 결측 때문에 학습 행이나 재학습 일정이 달라지면 비교가 불공정하다(공정 비교 회귀 방지)."""
    from pathlib import Path
    sig_base, sig_cand = Path(config.SIGNALS_NUMERIC_PATH), Path(config.SIGNALS_CANDIDATE_PATH)
    if not (sig_base.exists() and sig_cand.exists()):
        pytest.skip("저장된 신호가 없다")
    base, cand = pd.read_parquet(sig_base), pd.read_parquet(sig_cand)
    assert base.index.equals(cand.index)
    for col in ["retrained_on", "train_rows", "train_last_future_date"]:
        assert base[col].equals(cand[col]), col


@pytest.mark.parametrize("freq, expected", [("W", lambda d: d.to_period("W")), ("Q", lambda d: d.to_period("Q"))])
def test_재학습_주기를_바꿔도_정답_기간_간격을_지킨다(freq, expected):
    features, labels = make_data(300)
    out = tr.walk_forward(features, labels, valid_start="2025-05-01", freq=freq)
    assert (out["train_last_future_date"] < out["retrained_on"]).all()
    assert out["retrained_on"].nunique() == expected(out.index).nunique()


def test_알_수_없는_재학습_주기는_거부한다():
    features, labels = make_data()
    with pytest.raises(ValueError):
        tr.walk_forward(features, labels, valid_start="2025-05-01", freq="D")


def test_학습_창을_주면_재학습_시점에서_그_햇수_안의_행만_학습한다():
    features, labels = make_data(500)
    full = tr.walk_forward(features, labels, valid_start="2026-06-01", freq="M")
    window = tr.walk_forward(features, labels, valid_start="2026-06-01", freq="M", window_years=1)
    assert (window["train_rows"] <= full["train_rows"]).all() and (window["train_rows"] < full["train_rows"]).any()
    for start, rows in window.groupby("retrained_on")["train_rows"].first().items():
        in_window = labels.index[(labels.index >= start - pd.DateOffset(years=1)) & (labels["future_date"] < start)]
        assert rows == len(in_window)


def test_하락_위험_워크포워드는_정답_간격을_지키고_미래를_바꿔도_과거_예측이_같다():
    from src.pipeline.labels import make_risk_labels
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2025-01-01", periods=400)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days)))), index=days)
    features = pd.DataFrame({"f1": rng.normal(size=len(days)), "f2": rng.normal(size=len(days))}, index=days)
    labels = make_risk_labels(close)
    out = tr.walk_forward_risk(features, labels, valid_start="2025-10-01", feature_set=None, freq="M")
    assert (out["train_last_future_date"] < out["retrained_on"]).all()
    assert out[["p_risk_raw", "p_risk_cal", "p_logit"]].stack().between(0, 1).all()
    cut = pd.Timestamp("2026-01-01")
    close2 = close.copy()
    close2[close2.index >= cut] *= 0.5
    out2 = tr.walk_forward_risk(features, make_risk_labels(close2), valid_start="2025-10-01", feature_set=None, freq="M")
    before = out.index < pd.Timestamp("2025-12-01")  # 이 덩어리의 학습 정답은 cut 전에 끝난다
    pd.testing.assert_frame_equal(out[before], out2[before])


def test_20거래일_정답도_재학습일을_넘는_행은_학습에서_빠진다():
    """11일차: 정답 기간이 길어도 같은 규칙. 재학습일 R 직전 20거래일 정답만 바꿔도 R 덩어리 예측은 같아야 한다."""
    features, _ = make_data()
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(features)))), index=features.index)
    labels = make_labels(close, 20, 0.06)
    base = tr.walk_forward(features, labels, valid_start="2025-06-02")
    r = base["retrained_on"].unique()[2]
    straddle = (labels.index < r) & (labels["future_date"] >= r)
    assert straddle.sum() == 20
    flipped = labels.copy()
    flipped.loc[straddle, "label"] = "하락"
    changed = tr.walk_forward(features, flipped, valid_start="2025-06-02")
    block = base["retrained_on"] == r
    pd.testing.assert_frame_equal(base[block], changed[block])
