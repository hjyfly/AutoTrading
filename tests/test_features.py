import numpy as np
import pandas as pd
import pytest

from src import config
from src.pipeline import features as ft


@pytest.fixture(autouse=True)
def small_config(monkeypatch):
    """해외 시계열은 미국(us10y)·홍콩(hs) 두 개만 둔 작은 설정으로 고정한다."""
    monkeypatch.setattr(config, "FRED_SERIES", {"us10y": "DGS10"})
    monkeypatch.setattr(config, "YF_TICKERS", {"hs": "3033.HK"})
    monkeypatch.setattr(config, "LEVEL_SERIES", {"us10y"})


def make_raw(n=260, seed=0) -> dict:
    """무작위 가격 경로로 가짜 원천 데이터를 만든다. 해외 시계열은 한국과 같은 날짜에도 값이 있다."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=days)
    ohlcv = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                          "Volume": rng.integers(1_000, 2_000, n).astype(float)}, index=days)
    flows = pd.DataFrame({"foreign_net": rng.normal(0, 100, n), "institution_net": rng.normal(0, 100, n),
                          "foreign_hold_ratio": 50 + rng.normal(0, 0.1, n)}, index=days)
    foreign = lambda: pd.DataFrame({"value": 100 + np.cumsum(rng.normal(0, 1, n))}, index=days)
    return {"ohlcv": ohlcv, "flows": flows, "us10y": foreign(), "hs": foreign()}


def test_as_of_결과는_전체_결과의_같은_날짜_행과_같다():
    raw = make_raw()
    full = ft.build_features(raw=raw)
    for as_of in raw["ohlcv"].index[[150, 200, 259]]:
        part = ft.build_features(as_of=as_of, raw=raw)
        assert part.index.max() == as_of
        pd.testing.assert_series_equal(part.loc[as_of], full.loc[as_of], check_names=False)


def test_t일_이후_값을_바꿔도_t일_행은_변하지_않는다():
    raw = make_raw()
    t = raw["ohlcv"].index[200]
    before = ft.build_features(raw=raw).loc[t]
    changed = {k: v.copy() for k, v in raw.items()}
    for k, df in changed.items():
        df.loc[df.index >= t] = df.loc[df.index >= t] * 3  # t일 당일 값부터 크게 바꾼다
    after = ft.build_features(raw=changed).loc[t]
    pd.testing.assert_series_equal(before, after)


def test_해외_값은_같은_날짜를_쓰지_않고_전날_값을_쓴다():
    raw = make_raw()
    table = ft.build_features(raw=raw)
    t = raw["ohlcv"].index[100]
    prev = raw["ohlcv"].index[99]
    assert table.loc[t, "us10y_level"] == raw["us10y"].loc[prev, "value"]
    assert table.loc[t, "hs_age_days"] >= 1


def test_국내_특징은_전날_종가_기준이다():
    raw = make_raw()
    table = ft.build_features(raw=raw)
    days = raw["ohlcv"].index
    close = raw["ohlcv"]["Close"]
    assert table.loc[days[10], "ret_1d"] == pytest.approx(close.iloc[9] / close.iloc[8] - 1)


def test_이동평균_괴리율_손계산():
    close = pd.Series([1.0, 2, 3, 4, 5, 6])
    gap = ft.moving_average_features(close)["ma5_gap"]
    assert gap.iloc[4] == pytest.approx(5 / 3 - 1)
    assert np.isnan(gap.iloc[3])


def test_볼린저_pctb_는_중심선에서_0점5():
    close = pd.Series([10.0] * 19 + [9.0, 11.0, 10.0])
    pctb = ft.bollinger_features(close)["bb_pctb"]
    mid = close.rolling(20).mean().iloc[-1]
    assert pctb.iloc[-1] == pytest.approx(0.5 + (10 - mid) / (4 * close.rolling(20).std().iloc[-1]))


def test_파라볼릭_SAR_은_추세가_바뀌면_방향이_뒤집힌다():
    up = np.linspace(100, 120, 20)
    down = np.linspace(119, 90, 20)
    high = pd.Series(np.r_[up, down] + 1)
    low = pd.Series(np.r_[up, down] - 1)
    sar, trend = ft.parabolic_sar(high, low)
    assert trend.iloc[15] == 1 and trend.iloc[-1] == -1
    assert (sar.iloc[1:15] < low.iloc[1:15]).all()


@pytest.mark.parametrize("column", ["label", "label_y", "future_return", "close", "is_dummy"])
def test_정답_관련_열이_섞이면_막는다(column):
    table = pd.DataFrame({"ret_1d": [0.1], column: [1]}, index=pd.to_datetime(["2026-02-02"]))
    with pytest.raises(AssertionError):
        ft.assert_no_lookahead(table)


def test_시간대가_붙은_as_of_는_한국_날짜로_자른다():
    raw = make_raw()
    t = raw["ohlcv"].index[150]
    # t-1일 16:00 UTC = t일 01:00 KST → t 행까지 나와야 한다
    utc = (t - pd.Timedelta(days=1) + pd.Timedelta(hours=16)).tz_localize("UTC")
    assert ft.build_features(as_of=utc, raw=raw).index.max() == t
