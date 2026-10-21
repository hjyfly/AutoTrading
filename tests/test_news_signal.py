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
