import pandas as pd
import pytest

from src import config
from src.pipeline import select_news as sn

DAYS = pd.DatetimeIndex(pd.bdate_range("2026-02-02", "2026-02-06"))
T = pd.Timestamp("2026-02-04")
END = pd.Timestamp("2026-02-04 06:15", tz="UTC")  # 15:15 KST


@pytest.fixture(autouse=True)
def small_quota(monkeypatch):
    """테스트는 총 5건, 삼성 2 / 반도체 2 / 거시 1, 매체당 2건으로 고정한다."""
    monkeypatch.setattr(config, "SELECT_TOTAL", 5)
    monkeypatch.setattr(config, "SELECT_QUOTAS", {"samsung": 2, "semis": 2, "macro": 1})
    monkeypatch.setattr(config, "SELECT_MAX_PER_SOURCE", 2)
    monkeypatch.setattr(config, "SIGNAL_CUTOFF_KST", "15:15")


def make_news(rows):
    """(제목, 매체, 15:15 몇 분 전) 목록으로 신호일 T 의 기사 표를 만든다."""
    return pd.DataFrame({
        "signal_date": [T] * len(rows),
        "title": [r[0] for r in rows],
        "source": [r[1] for r in rows],
        "published_utc": [END - pd.Timedelta(minutes=r[2]) for r in rows],
    })


def test_분류별_할당대로_뽑고_15시15분에_가까운_기사부터():
    news = make_news([
        ("삼성전자 신고가", "a", 10), ("Samsung HBM deal", "b", 20), ("삼성 실적 전망", "c", 30),
        ("반도체 수출 증가", "d", 10), ("Nvidia earnings", "e", 20), ("엔비디아 신제품", "f", 30),
        ("유가 급등", "g", 10), ("연준 금리 동결", "h", 20),
    ])
    picked = sn.select_articles(news, T, DAYS)
    counts = picked["category"].value_counts().to_dict()
    assert counts == {"samsung": 2, "semis": 2, "macro": 1}
    assert "삼성 실적 전망" not in picked["title"].tolist()  # 가장 오래된 삼성 기사는 빠진다


def test_할당을_못_채우면_앞_분류부터_채운다():
    news = make_news([("삼성 1", "a", 1), ("삼성 2", "b", 2), ("삼성 3", "c", 3), ("삼성 4", "d", 4), ("유가 1", "e", 5)])
    picked = sn.select_articles(news, T, DAYS)
    assert len(picked) == 5
    assert (picked["category"] == "samsung").sum() == 4


def test_매체당_상한과_비슷한_제목_제거():
    news = make_news([
        ("삼성전자 신고가 경신", "a", 1), ("삼성전자 신고가 경신!!", "b", 2),  # 비슷한 제목
        ("삼성 A", "x", 3), ("삼성 B", "x", 4), ("삼성 C", "x", 5),  # 같은 매체 3건
    ])
    picked = sn.select_articles(news, T, DAYS)
    assert picked["title"].str.startswith("삼성전자 신고가").sum() == 1
    assert (picked["source"] == "x").sum() == 2


def test_분류에_안_걸리는_기사와_빈_제목은_뺀다():
    news = make_news([("날씨 맑음", "a", 1), ("", "b", 2), ("삼성 X", "c", 3)])
    assert sn.select_articles(news, T, DAYS)["title"].tolist() == ["삼성 X"]


@pytest.mark.parametrize("bad_time", [END, pd.NaT])  # 15:15 정각은 다음 신호일 기사, 시각 없음
def test_신호_구간_밖이거나_시각_없는_기사가_있으면_멈춘다(bad_time):
    news = make_news([("삼성 X", "a", 1)])
    news.loc[0, "published_utc"] = bad_time
    with pytest.raises(ValueError):
        sn.select_articles(news, T, DAYS)


def test_기사가_없는_날은_빈_표():
    assert sn.select_articles(make_news([]), T, DAYS).empty
