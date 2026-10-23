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


def test_샘플_신호일은_매주_수요일(monkeypatch):
    monkeypatch.setattr(config, "NEWS_START", "2026-02-01")
    monkeypatch.setattr(config, "NEWS_END", "2026-04-30")
    days = pd.DatetimeIndex(pd.bdate_range("2026-02-01", "2026-04-30"))
    picked = cn.sample_signal_dates(days)
    assert all(d.weekday() == 2 for d in picked)
    assert len(picked) == 13


def test_수요일이_휴장이면_다음_거래일(monkeypatch):
    monkeypatch.setattr(config, "NEWS_START", "2026-02-01")
    monkeypatch.setattr(config, "NEWS_END", "2026-04-30")
    days = pd.DatetimeIndex(pd.bdate_range("2026-02-01", "2026-04-30")).drop(pd.Timestamp("2026-02-18"))
    assert pd.Timestamp("2026-02-19") in cn.sample_signal_dates(days)


def test_실시간_채점일_기사는_그날_행만_수집본에_덧붙이고_원본은_덮어쓰지_않는다(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path))
    days = pd.bdate_range("2026-10-19", "2026-10-30")
    t = pd.Timestamp("2026-10-23")
    old = pd.DataFrame({"signal_date": [pd.Timestamp("2026-10-20")], "provider": ["gdelt_gkg"], "published_utc": ["2026-10-20T01:00:00Z"],
                        "title": ["옛 기사"], "url": ["u0"], "source": ["s"]})
    old.to_parquet(tmp_path / "news.parquet")
    gdelt = pd.DataFrame({"provider": "gdelt_gkg", "published_utc": ["2026-10-23T01:00:00Z", "2026-10-23T07:00:00Z"],  # 10:00 KST(구간 안), 16:00 KST(구간 밖)
                          "title": ["안", "늦음"], "url": ["u1", "u2"], "source": "s"})
    rss = pd.DataFrame({"provider": "google_rss", "published_utc": ["2026-10-23T02:00:00Z", "2026-10-23T07:00:00Z"],
                        "title": ["RSS 안", "RSS 다음날"], "url": ["r1", "r2"], "source": "s",
                        "signal_date": [t, pd.Timestamp("2026-10-26")]})
    day = cn.save_day(t, gdelt, rss, days)
    assert sorted(day["title"]) == ["RSS 안", "안"]
    news = pd.read_parquet(tmp_path / "news.parquet")
    assert len(news) == 3 and "옛 기사" in set(news["title"])
    assert (tmp_path / "raw" / "news" / "2026-10-23.parquet").exists() and (tmp_path / "raw" / "news" / "rss_2026-10-23.parquet").exists()
    cn.save_day(t, gdelt.assign(title="다시"), rss, days)  # 같은 날 다시 실행해도 원본·수집본을 바꾸지 않는다
    assert len(pd.read_parquet(tmp_path / "news.parquet")) == 3
    assert set(pd.read_parquet(tmp_path / "raw" / "news" / "2026-10-23.parquet")["title"]) == {"안", "늦음"}


def test_과거_구간을_다시_모아도_실시간_채점일_행은_남는다(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path))
    days = pd.bdate_range("2026-10-19", "2026-10-30")
    monkeypatch.setattr(cn, "trading_days", lambda: days)
    live_day = pd.DataFrame({"signal_date": [pd.Timestamp("2026-10-23")], "provider": ["gdelt_gkg"], "published_utc": ["2026-10-23T01:00:00Z"],
                             "title": ["실시간"], "url": ["u9"], "source": ["s"]})
    live_day.to_parquet(tmp_path / "news.parquet")
    (tmp_path / "raw" / "news").mkdir(parents=True)
    pd.DataFrame({"provider": ["gdelt_gkg"], "published_utc": ["2026-10-20T01:00:00Z"], "title": ["과거"], "url": ["u1"], "source": ["s"],
                  "signal_date": [pd.Timestamp("2026-10-20")]}).to_parquet(tmp_path / "raw" / "news" / "2026-10-20.parquet")
    pd.DataFrame(columns=["provider", "published_utc", "title", "url", "source", "signal_date"]).to_parquet(tmp_path / "raw" / "news" / "rss_2026-10-20.parquet")
    cn.main(["2026-10-20"])
    assert set(pd.read_parquet(tmp_path / "news.parquet")["title"]) == {"실시간", "과거"}
