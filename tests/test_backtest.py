import pandas as pd
import pytest

from src import config
from src.pipeline import backtest as bt
from src.pipeline.labels import DOWN, FLAT, UP

DAYS = pd.bdate_range("2026-02-02", periods=3)
SAMSUNG = pd.Series([100.0, 110.0, 99.0], index=DAYS)
INVERSE = pd.Series([100.0, 95.0, 100.0], index=DAYS)


@pytest.fixture(autouse=True)
def fee(monkeypatch):
    """테스트는 수수료 0.1% 로 고정한다."""
    monkeypatch.setattr(config, "FEE_RATE", 0.001)


def test_손계산_상승_하락_보합():
    result = bt.simulate(pd.Series([UP, DOWN, FLAT], index=DAYS), SAMSUNG, INVERSE)
    e0 = 0.999  # 현금 → 삼성 매수 0.1%
    e1 = e0 * 110 / 100 * 0.999 ** 2  # 삼성 +10% 뒤 인버스로 전환(매도 0.1% + 매수 0.1%)
    e2 = e1 * 100 / 95  # 보합이라 인버스 유지, 수수료 없음
    assert result["equity"].tolist() == pytest.approx([e0, e1, e2])
    assert result["position"].tolist() == [bt.LONG, bt.INVERSE, bt.INVERSE]
    assert result["fee"].iloc[2] == 0


def test_첫날_보합이면_현금을_유지한다():
    result = bt.simulate(pd.Series([FLAT, UP, UP], index=DAYS), SAMSUNG, INVERSE)
    assert result["position"].tolist() == [bt.CASH, bt.LONG, bt.LONG]
    assert result["equity"].tolist() == pytest.approx([1.0, 0.999, 0.999 * 99 / 110])


def test_들고_있기():
    hold = bt.buy_and_hold(SAMSUNG, DAYS)
    assert hold.tolist() == pytest.approx([0.999, 0.999 * 1.1, 0.999 * 0.99])


def test_MDD_는_누적_최고점_대비():
    equity = pd.Series([1.1, 0.88, 1.2, 0.9])
    assert bt.max_drawdown(equity) == pytest.approx(0.9 / 1.2 - 1)
    assert bt.max_drawdown(pd.Series([0.95, 0.97])) == pytest.approx(-0.05)  # 시작 1.0 대비 하락 포함


def eq(values):
    return pd.Series(values, index=pd.bdate_range("2026-01-01", periods=len(values)))


def test_기준A_두_조건_모두_통과():
    hold = eq([1.0, 1.4, 1.05, 1.4])  # MDD -25%, 수익 +40%
    model = eq([1.0, 1.2, 1.1, 1.25])  # MDD -8.3%, 수익 +25%
    v = bt.judge_a(model, hold)
    assert v["mdd_ok"] and v["return_ok"] and v["passed"]


def test_기준A_현금만_들고_있으면_수익_조건에서_실패():
    hold = eq([1.0, 1.4, 1.05, 1.4])
    cash = eq([1.0, 1.0, 1.0, 1.0])
    v = bt.judge_a(cash, hold)
    assert v["mdd_ok"] and not v["return_ok"] and not v["passed"]


def test_기준A_들고_있기가_손실이면_들고_있기_이상이어야_한다():
    hold = eq([1.0, 0.8, 0.9])  # 수익 -10%
    v = bt.judge_a(eq([1.0, 0.95, 0.92]), hold)  # 수익 -8%, MDD -8%
    assert "들고 있기 수익 ≤ 0" in v["return_rule"]
    assert v["return_ok"] and v["mdd_ok"]


def test_기준B():
    numeric = eq([1.0, 0.9, 1.05])
    assert bt.judge_b(eq([1.0, 0.95, 1.06]), numeric)["passed"]
    assert not bt.judge_b(eq([1.0, 0.95, 1.0]), numeric)["passed"]


def test_방향_적중률과_치명적_오류():
    days = pd.bdate_range("2026-02-02", periods=5)
    signals = pd.Series([UP, UP, DOWN, FLAT, DOWN], index=days)
    labels = pd.Series([UP, DOWN, DOWN, UP, UP], index=days)
    r = bt.direction_hit_rate(signals, labels)
    assert r["n"] == 4  # 보합 신호는 빼고 센다
    assert r["hit_rate"] == pytest.approx(2 / 4)
    assert r["critical_error_rate"] == pytest.approx(2 / 4)


def test_인버스에서_삼성으로_전환과_마지막날_전환():
    result = bt.simulate(pd.Series([DOWN, UP, DOWN], index=DAYS), SAMSUNG, INVERSE)
    e0 = 0.999  # 현금 → 인버스
    e1 = e0 * 95 / 100 * 0.999 ** 2  # 인버스 -5% 뒤 삼성으로 전환
    e2 = e1 * 99 / 110 * 0.999 ** 2  # 삼성 -10% 뒤 마지막 날 다시 인버스로 전환(수수료만, 이후 수익 없음)
    assert result["equity"].tolist() == pytest.approx([e0, e1, e2])


def test_기준A_경계값():
    hold = eq([1.0, 1.2, 0.96, 1.2])  # MDD -20%, 수익 +20%
    exactly_half = eq([1.0, 1.1, 0.99, 1.1])  # MDD -10%, 수익 +10%
    v = bt.judge_a(exactly_half, hold)
    assert v["mdd_ok"] and v["return_ok"]
    flat_hold = eq([1.0, 0.9, 1.0])  # 수익 정확히 0 → "들고 있기 이상" 규칙
    assert "들고 있기 수익 ≤ 0" in bt.judge_a(eq([1.0, 0.95, 1.0]), flat_hold)["return_rule"]


def test_기준B_동률은_통과_날짜가_다르면_거부():
    numeric = eq([1.0, 0.9, 1.05])
    assert bt.judge_b(numeric.copy(), numeric)["passed"]
    with pytest.raises(ValueError):
        bt.judge_b(numeric.iloc[1:], numeric)


def test_정답이_없는_날은_적중률에서_뺀다():
    days = pd.bdate_range("2026-02-02", periods=3)
    r = bt.direction_hit_rate(pd.Series([UP, UP, UP], index=days), pd.Series([UP, DOWN, None], index=days))
    assert r["n"] == 2 and r["hit_rate"] == pytest.approx(0.5)


def test_가격이_빠진_날짜가_있으면_멈춘다():
    with pytest.raises(ValueError):
        bt.simulate(pd.Series([UP, UP, UP], index=DAYS), SAMSUNG.iloc[:2], INVERSE)


def test_기준B_는_두_모델을_같은_날_현금으로_시작해_비교한다(monkeypatch, tmp_path):
    days = pd.bdate_range("2026-02-02", periods=3)
    samsung = pd.Series([100.0, 110.0, 99.0], index=days)
    inverse = pd.Series([100.0, 90.0, 100.0], index=days)
    monkeypatch.setattr(bt, "load_prices", lambda: (samsung, inverse))
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(config, "NEWS_START", "2026-02-02")
    monkeypatch.setattr(config, "NEWS_END", "2026-02-04")
    pd.DataFrame({"label": [UP, DOWN, FLAT]}, index=days).to_parquet(tmp_path / "labels.parquet")
    numeric = pd.DataFrame({"signal": [UP] * 6}, index=pd.bdate_range("2026-01-28", periods=3).append(days))
    news = pd.DataFrame({"signal": [FLAT, DOWN, DOWN], "adjusted": [True, True, True], "news_scored": True}, index=days)
    r = bt.evaluate_b(numeric, news)
    numeric_expected = 0.999 * 99 / 100  # 2/2 매수(현금 시작) 후 끝까지 삼성
    news_expected = 1.0 * 0.999 * 100 / 90  # 2/2 보합이라 현금, 2/3 인버스 매수, 2/4 인버스 +11.1%
    assert r["judge_b"]["numeric"]["total_return"] == pytest.approx(numeric_expected - 1)
    assert r["judge_b"]["news"]["total_return"] == pytest.approx(news_expected - 1)
    assert r["period"] == ("2026-02-02", "2026-02-04", 3)
