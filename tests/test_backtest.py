import numpy as np
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
    """테스트는 수수료 0.1% 로, 하락 행동은 손계산을 만든 인버스 규칙으로 고정한다(현금 규칙은 따로 지정해 시험)."""
    monkeypatch.setattr(config, "FEE_RATE", 0.001)
    monkeypatch.setattr(config, "DOWN_ACTION", "inverse")


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
    monkeypatch.setattr(config, "CRITERION_B_START", "2026-02-02")
    monkeypatch.setattr(config, "CRITERION_B_END", "2026-02-04")
    pd.DataFrame({"label": [UP, DOWN, FLAT]}, index=days).to_parquet(tmp_path / "labels.parquet")
    numeric = pd.DataFrame({"signal": [UP] * 6}, index=pd.bdate_range("2026-01-28", periods=3).append(days))
    news = pd.DataFrame({"signal": [FLAT, DOWN, DOWN], "numeric_signal": [UP] * 3, "adjusted": [True, True, True], "news_scored": True}, index=days)
    r = bt.evaluate_b(numeric, news)
    numeric_expected = 0.999 * 99 / 100  # 2/2 매수(현금 시작) 후 끝까지 삼성
    news_expected = 1.0 * 0.999 * 100 / 90  # 2/2 보합이라 현금, 2/3 인버스 매수, 2/4 인버스 +11.1%
    assert r["judge_b"]["numeric"]["total_return"] == pytest.approx(numeric_expected - 1)
    assert r["judge_b"]["news"]["total_return"] == pytest.approx(news_expected - 1)
    assert r["period"] == ("2026-02-02", "2026-02-04", 3)


def test_하락_현금_규칙은_삼성을_팔고_현금으로_있는다():
    result = bt.simulate(pd.Series([UP, DOWN, FLAT], index=DAYS), SAMSUNG, INVERSE, down_action="cash")
    e0 = 0.999  # 현금 → 삼성 매수
    e1 = e0 * 110 / 100 * 0.999  # 삼성 +10% 뒤 매도만(0.1%), 인버스는 사지 않는다
    assert result["position"].tolist() == [bt.LONG, bt.CASH, bt.CASH]
    assert result["equity"].tolist() == pytest.approx([e0, e1, e1])  # 현금이라 2/4 하락 영향 없음


def test_하락_인버스_규칙은_기존과_같다():
    a = bt.simulate(pd.Series([UP, DOWN, FLAT], index=DAYS), SAMSUNG, INVERSE, down_action="inverse")
    assert a["position"].tolist() == [bt.LONG, bt.INVERSE, bt.INVERSE]


def test_알_수_없는_하락_행동은_거부한다():
    with pytest.raises(ValueError):
        bt.simulate(pd.Series([UP], index=DAYS[:1]), SAMSUNG, INVERSE, down_action="short")


def test_현금_규칙_첫날_하락은_수수료_없이_현금():
    days = pd.bdate_range("2026-03-02", periods=4)
    samsung = pd.Series([100.0, 90.0, 95.0, 100.0], index=days)
    result = bt.simulate(pd.Series([DOWN, FLAT, UP, DOWN], index=days), samsung, pd.Series(dtype=float), down_action="cash")
    assert result["position"].tolist() == [bt.CASH, bt.CASH, bt.LONG, bt.CASH]  # 인버스 가격 없이도 계산된다
    assert result["equity"].tolist() == pytest.approx([1.0, 1.0, 0.999, 0.999 * 100 / 95 * 0.999])


def test_적중률은_구간_뒤에_확정된_정답을_쓰지_않는다(monkeypatch, tmp_path):
    days = pd.bdate_range("2026-01-05", periods=12)
    close = pd.Series(range(100, 112), index=days, dtype=float)
    monkeypatch.setattr(bt, "load_prices", lambda: (close, close))
    labels = pd.DataFrame({"label": [UP] * 12, "future_date": days.shift(5, freq="B")}, index=days)
    monkeypatch.setattr(bt.config, "PROCESSED_DIR", str(tmp_path))
    labels.to_parquet(tmp_path / "labels.parquet")
    signals = pd.DataFrame({"signal": [UP] * 12, "confidence": 0.5}, index=days)
    r = bt.evaluate(signals, end=days[9], down_action="cash")
    assert r["direction"]["n"] == 5  # 정답 기간이 구간 끝(10번째 날) 안에 끝난 신호만


def test_공식_구간은_데이터가_늘어도_그대로다(monkeypatch, tmp_path):
    days = pd.bdate_range("2026-01-05", periods=5)
    close = pd.Series([100.0, 110, 99, 120, 60], index=days)
    monkeypatch.setattr(bt, "load_prices", lambda: (close, close))
    monkeypatch.setattr(bt.config, "PROCESSED_DIR", str(tmp_path))
    pd.DataFrame({"label": [UP] * 5, "future_date": days.shift(1, freq="B")}, index=days).to_parquet(tmp_path / "labels.parquet")
    signals = pd.DataFrame({"signal": [UP, FLAT, DOWN, UP, UP], "confidence": 0.5}, index=days)
    short = bt.evaluate(signals.iloc[:3], end=days[2], down_action="cash")
    longer = bt.evaluate(signals, end=days[2], down_action="cash")
    assert short["judge_a"]["model"] == longer["judge_a"]["model"]
    assert longer["judge_a"]["model"]["mdd"] == pytest.approx(0.9 * 0.999 - 1)  # 110 → 99 하락 뒤 매도 수수료
    assert longer["direction"]["n"] == short["direction"]["n"] == 1  # 3일째 신호 정답(4일째 종가)은 구간 뒤라 빠진다


def test_분할_매매_1단계_바닥_0은_전량_현금_규칙과_같다():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2026-01-05", periods=60)
    samsung = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days)))), index=days)
    signals = pd.Series(rng.choice([UP, FLAT, DOWN], len(days)), index=days)
    whole = bt.simulate(signals, samsung, pd.Series(dtype=float), down_action="cash")
    split = bt.simulate_split(signals, samsung, steps=1, floor=0.0)
    assert np.allclose(whole["equity"], split["equity"]) and (whole["fee"] > 0).sum() == (split["fee"] > 0).sum()


def test_분할_매매_손계산():
    days = pd.bdate_range("2026-01-05", periods=4)
    samsung = pd.Series([100.0, 110.0, 110.0, 99.0], index=days)
    result = bt.simulate_split(pd.Series([UP, UP, DOWN, FLAT], index=days), samsung, steps=2, floor=0.0)
    assert result["weight"].tolist() == [0.5, 1.0, 0.5, 0.5]
    fee = config.FEE_RATE
    eq0 = 1 - 0.5 * fee                                   # 첫날 절반 매수(수수료를 뗀 뒤 절반씩)
    stock1 = 0.5 * eq0 * 1.1                              # 주식 절반이 10% 오름
    eq1 = stock1 + 0.5 * eq0
    eq1 -= (eq1 - stock1) * fee                           # 100% 로 늘림
    assert result["equity"].iloc[1] == pytest.approx(eq1)
    assert result["equity"].iloc[3] == pytest.approx(result["equity"].iloc[2] - 0.5 * result["equity"].iloc[2] * 0.1, rel=1e-6)


def test_분할_매매_바닥_비중은_하락에도_남긴다():
    days = pd.bdate_range("2026-01-05", periods=3)
    result = bt.simulate_split(pd.Series([FLAT, DOWN, DOWN], index=days), pd.Series(100.0, index=days), steps=2, floor=0.5)
    assert result["weight"].tolist() == [0.5, 0.5, 0.5]


def test_분할_매매_첫_상승도_바닥_아래로_가지_않고_하락에_사지_않는다():
    days = pd.bdate_range("2026-01-05", periods=3)
    result = bt.simulate_split(pd.Series([UP, DOWN, DOWN], index=days), pd.Series(100.0, index=days), steps=3, floor=0.5)
    assert result["weight"].tolist() == pytest.approx([0.5 + 1 / 3, 0.5, 0.5]) and result["fee"].iloc[2] == 0


def test_분할_단계_0은_거부한다():
    days = pd.bdate_range("2026-01-05", periods=2)
    with pytest.raises(ValueError):
        bt.simulate_split(pd.Series([UP, UP], index=days), pd.Series(100.0, index=days), steps=0)


def test_기준_B는_다른_숫자_모델을_보정한_뉴스_신호와_섞지_않는다(monkeypatch, tmp_path):
    days = pd.bdate_range("2026-02-02", periods=3)
    monkeypatch.setattr(bt, "load_prices", lambda: (pd.Series(100.0, index=days), pd.Series(100.0, index=days)))
    monkeypatch.setattr(config, "CRITERION_B_START", "2026-02-02")
    monkeypatch.setattr(config, "CRITERION_B_END", "2026-02-04")
    numeric = pd.DataFrame({"signal": [UP, UP, UP]}, index=days)
    news = pd.DataFrame({"signal": [UP, UP, UP], "numeric_signal": [UP, DOWN, UP], "adjusted": False, "news_scored": True}, index=days)
    with pytest.raises(ValueError):
        bt.evaluate_b(numeric, news)


def test_장세_구분_경계():
    assert bt.regime(0.15) == "상승장" and bt.regime(-0.15) == "하락장" and bt.regime(0.149) == "횡보장"


def test_구간_성적_손계산과_구간_뒤_정답_제외():
    days = pd.bdate_range("2026-01-05", periods=4)
    close = pd.Series([100.0, 110.0, 99.0, 104.5], index=days)
    signals = pd.Series([UP, FLAT, DOWN, FLAT], index=days)
    labels = pd.DataFrame({"label": [UP, FLAT, DOWN, FLAT],
                           "future_date": [days[1], days[2], days[3], days[3] + pd.Timedelta(days=1)]}, index=days)
    r = bt.period_stats(signals, close, labels, days[0], days[-1])
    assert r["model"]["total_return"] == pytest.approx(0.98901 * 0.999 - 1)
    assert r["model"]["mdd"] == pytest.approx(0.98901 * 0.999 / 1.0989 - 1)
    assert r["hold"]["total_return"] == pytest.approx(0.999 * 1.045 - 1)
    assert r["accuracy"] == 1.0  # 마지막 날 정답은 구간 끝 뒤에 확정돼 빠진다(앞 3일은 모두 맞음)
    assert r["hold_share"] == pytest.approx(0.5)  # 1·2일 보유, 3일 하락으로 현금, 4일 보합 유지
