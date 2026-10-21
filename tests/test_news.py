import pandas as pd
import pytest

from src import config
from src.pipeline import collect_news as cn

# 2026-02-02(월) ~ 02-06(금) 거래일. 02-04(수)를 기준 신호일로 쓴다
DAYS = pd.DatetimeIndex(pd.bdate_range("2026-02-02", "2026-02-06"))


@pytest.fixture(autouse=True)
def fixed_cutoff(monkeypatch):
    """테스트는 15:15 기준으로 고정한다."""
    monkeypatch.setattr(config, "SIGNAL_CUTOFF_KST", "15:15")


def kst(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=cn.KST)


def test_15시14분_기사는_그날_신호에_속한다():
    assert cn.assign_signal_date(kst("2026-02-04 15:14"), DAYS) == pd.Timestamp("2026-02-04")


def test_15시16분_기사는_다음_거래일_신호에_속한다():
    assert cn.assign_signal_date(kst("2026-02-04 15:16"), DAYS) == pd.Timestamp("2026-02-05")


def test_금요일_장마감_뒤_기사는_월요일이_아니라_다음_거래일로_간다():
    days = DAYS.append(pd.DatetimeIndex(["2026-02-09"]))
    assert cn.assign_signal_date(kst("2026-02-06 20:00"), days) == pd.Timestamp("2026-02-09")


def test_신호_구간은_전_거래일_15시15분부터_당일_15시15분까지_UTC로():
    start, end = cn.signal_window(pd.Timestamp("2026-02-04"), DAYS)
    assert start == pd.Timestamp("2026-02-03 06:15", tz="UTC")
    assert end == pd.Timestamp("2026-02-04 06:15", tz="UTC")


def test_UTC_시각도_KST로_바꿔_배정한다():
    # 2026-02-04 06:16 UTC = 15:16 KST → 다음 거래일
    assert cn.assign_signal_date(pd.Timestamp("2026-02-04 06:16", tz="UTC"), DAYS) == pd.Timestamp("2026-02-05")


def test_RSS_날짜만_있는_기사는_미국_서부_그날_끝을_발행_시각으로_본다():
    # 실제 RSS pubDate: 2026-02-03 08:00 UTC (= 2월 3일 미국 서부 자정)
    effective = cn.rss_effective_time(pd.Timestamp("2026-02-03 08:00", tz="UTC"))
    assert effective == pd.Timestamp("2026-02-04 08:00", tz="UTC")  # = 2월 4일 17:00 KST
    assert cn.assign_signal_date(effective, DAYS) == pd.Timestamp("2026-02-05")  # 2월 4일 신호에는 못 쓴다


def test_15시15분_정각_기사는_다음_거래일_신호에_속한다():
    assert cn.assign_signal_date(kst("2026-02-04 15:15"), DAYS) == pd.Timestamp("2026-02-05")


def test_15시15분_정각_GDELT_배치는_그날에서_빼고_다음날에_넣는다():
    start, end = cn.signal_window(pd.Timestamp("2026-02-04"), DAYS)
    df = pd.DataFrame({
        "signal_date": [pd.Timestamp("2026-02-04")] * 3,
        "published_utc": [start, end - pd.Timedelta(minutes=15), end],
    })
    kept = cn.within_window(df, DAYS)
    assert kept["published_utc"].tolist() == [start, end - pd.Timedelta(minutes=15)]


def test_구간_확인은_빈_표에서도_걸러진_결과를_돌려준다():
    df = pd.DataFrame({"signal_date": [None], "published_utc": [pd.Timestamp("2026-02-04", tz="UTC")]})
    assert cn.within_window(df, DAYS).empty


@pytest.mark.parametrize("pub_utc, expected_kst, signal", [
    ("2026-03-07 08:00", "2026-03-08 17:00", "2026-03-09"),  # 서머타임 전(PST)
    ("2026-03-08 08:00", "2026-03-09 16:00", "2026-03-10"),  # 서머타임 시작일(PDT)
])
def test_RSS_서머타임_전환일(pub_utc, expected_kst, signal):
    days = pd.DatetimeIndex(pd.bdate_range("2026-03-02", "2026-03-13"))
    effective = cn.rss_effective_time(pd.Timestamp(pub_utc, tz="UTC"))
    assert effective.tz_convert(cn.KST) == kst(expected_kst)
    assert cn.assign_signal_date(effective, days) == pd.Timestamp(signal)


def test_샘플_신호일은_매주_수요일():
    days = pd.DatetimeIndex(pd.bdate_range(config.NEWS_START, config.NEWS_END))
    picked = cn.sample_signal_dates(days)
    assert all(d.weekday() == 2 for d in picked)
    assert len(picked) == 13


def test_수요일이_휴장이면_다음_거래일():
    days = pd.DatetimeIndex(pd.bdate_range(config.NEWS_START, config.NEWS_END)).drop(pd.Timestamp("2026-02-18"))
    assert pd.Timestamp("2026-02-19") in cn.sample_signal_dates(days)
