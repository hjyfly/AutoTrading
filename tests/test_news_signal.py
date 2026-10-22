import pandas as pd
import pytest

from src import config
from src.pipeline import news_signal as ns
from src.pipeline.labels import DOWN, FLAT, UP


@pytest.fixture(autouse=True)
def rule(monkeypatch):
    """기준값 ±3, 확신도 0.6 으로 고정한다."""
    monkeypatch.setattr(config, "NEWS_ADJUST_SCORE", 3)
    monkeypatch.setattr(config, "NEWS_ADJUST_MIN_CONF", 0.6)


@pytest.mark.parametrize("signal, consensus, conf, expected", [
    (FLAT, 3, 0.6, UP),      # 경계값 포함
    (DOWN, 4, 0.8, FLAT),    # 한 단계만 올린다
    (UP, 5, 0.9, UP),        # 맨 위는 그대로
    (FLAT, -3, 0.7, DOWN),
    (DOWN, -5, 0.9, DOWN),   # 맨 아래는 그대로
    (FLAT, 2, 0.9, FLAT),    # 기준 미달
    (FLAT, 4, 0.59, FLAT),   # 확신도 미달
    (FLAT, float("nan"), float("nan"), FLAT),  # 채점 안 한 날
])
def test_보정_규칙(signal, consensus, conf, expected):
    assert ns.adjust(signal, consensus, conf) == expected


def test_뉴스_점수는_같은_날짜에_붙이고_빈_날은_채우지_않는다():
    days = pd.bdate_range("2026-03-09", periods=3)
    numeric = pd.DataFrame({"signal": [FLAT, FLAT, FLAT], "confidence": [0.5] * 3}, index=days)
    scores = pd.DataFrame({"signal_date": [days[1]], "news_consensus": [4], "news_consensus_conf": [0.8], "no_news": [False]})
    out = ns.news_signals(numeric, scores)
    assert out["signal"].tolist() == [FLAT, UP, FLAT]
    assert out["news_scored"].tolist() == [False, True, False]


def test_최근_k일_평균은_과거_쪽으로만_굴린다():
    days = pd.bdate_range("2026-03-09", periods=3)
    numeric = pd.DataFrame({"signal": [FLAT] * 3, "confidence": [0.5] * 3}, index=days)
    scores = pd.DataFrame({"signal_date": days, "news_consensus": [0, 4, 4], "news_consensus_conf": [0.8] * 3, "no_news": [False] * 3})
    out = ns.news_signals(numeric, scores, window=2)
    assert out["news_used"].tolist() == [0, 2, 4]  # 첫날 값에 다음 날 점수가 섞이지 않는다
    assert out["signal"].tolist() == [FLAT, FLAT, UP]


def test_분류별_점수와_하락_위험_거부():
    days = pd.bdate_range("2026-03-09", periods=2)
    numeric = pd.DataFrame({"signal": [UP, FLAT], "confidence": [0.5] * 2}, index=days)
    scores = pd.DataFrame({"signal_date": days, "news_consensus": [0, 0], "news_consensus_conf": [0.8] * 2, "no_news": [False] * 2,
                           "news_semis_score": [0, 4], "news_downside_risk": [4, 0]})
    out = ns.news_signals(numeric, scores, field="news_semis_score", risk_veto=4)
    assert out["signal"].tolist() == [FLAT, UP]


def test_버전_1_점수표로_분류별_보정을_하면_막는다():
    days = pd.bdate_range("2026-03-09", periods=1)
    numeric = pd.DataFrame({"signal": [FLAT], "confidence": [0.5]}, index=days)
    scores = pd.DataFrame({"signal_date": days, "news_consensus": [4], "news_consensus_conf": [0.8], "no_news": [False]})
    with pytest.raises(ValueError):
        ns.news_signals(numeric, scores, field="news_macro_score")
