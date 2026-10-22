import json
import threading

import pandas as pd
import pytest

from src import config
from src.pipeline import scoring as sc

T = pd.Timestamp("2026-03-11")


class StubLLM:
    """with_structured_output 을 흉내 내는 가짜 LLM. 호출된 스키마와 메시지를 기록한다."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def with_structured_output(self, schema):
        stub = self

        class Runner:
            def invoke(self, messages):
                with stub.lock:
                    stub.calls.append((schema.__name__, messages))
                base = {"score": 2, "confidence": 0.6, "rationale": "근거", "key_articles": [1],
                        "samsung_score": 3, "semis_score": 1, "macro_score": -1, "downside_risk": 2}
                if schema is sc.DebateOpinion:
                    base["change_reason"] = "유지"
                if schema is sc.Consensus:
                    base["disagreement"] = "없음"
                return schema(**base)

        return Runner()


@pytest.fixture(autouse=True)
def tmp_paths(tmp_path, monkeypatch):
    """판단 기록과 점수 표를 임시 폴더에 쓴다."""
    monkeypatch.setattr(config, "TRACES_DIR", str(tmp_path / "traces"))
    monkeypatch.setattr(config, "SCORES_PATH", str(tmp_path / "scores.parquet"))


def make_articles(n=3):
    """신호일 T 의 가짜 선별 기사."""
    return pd.DataFrame({
        "title": [f"삼성전자 기사 {i}" for i in range(n)],
        "source": ["a"] * n,
        "url": [f"http://x/{i}" for i in range(n)],
        "category": ["samsung"] * n,
        "published_utc": [pd.Timestamp("2026-03-11 05:00", tz="UTC") - pd.Timedelta(minutes=i) for i in range(n)],
    })


REPORT = pd.Series({"report_nm": "사업보고서 (2025.12)", "rcept_dt": pd.Timestamp("2026-03-10"), "overview": "반도체와 스마트폰 사업"})


def test_채점자_3명_토론_3명_중재자_순서로_7번_부른다():
    stub = StubLLM()
    trace = sc.score_day(T, model=stub, articles=make_articles(), report=REPORT)
    names = [c[0] for c in stub.calls]
    assert names[:3].count("Opinion") == 3
    assert names[3:6].count("DebateOpinion") == 3
    assert names[6] == "Consensus" and len(names) == 7
    assert trace["final"]["score"] == 2


def test_모든_프롬프트에_기준_시각_조건이_들어간다():
    stub = StubLLM()
    sc.score_day(T, model=stub, articles=make_articles(), report=REPORT)
    for _, messages in stub.calls:
        assert all("2026-03-11 15:15" in m.content and "이후" in m.content for m in messages)


def test_토론_단계에는_다른_채점자_1차_의견이_들어간다():
    stub = StubLLM()
    sc.score_day(T, model=stub, articles=make_articles(), report=REPORT)
    debate = [m for name, m in stub.calls if name == "DebateOpinion"][0]
    assert all(p in debate[-1].content for p in config.SCORING_PERSONAS)


def test_뉴스가_없으면_LLM_을_부르지_않고_0점():
    stub = StubLLM()
    trace = sc.score_day(T, model=stub, articles=make_articles(0), report=REPORT)
    assert stub.calls == []
    assert trace["no_news"] and trace["final"]["score"] == 0


def test_판단_기록과_점수_표():
    sc.score_day(T, model=StubLLM(), articles=make_articles(), report=REPORT)
    saved = json.loads((sc.Path(config.TRACES_DIR) / "2026-03-11.json").read_text(encoding="utf-8"))
    assert saved["company"]["report_nm"] == "사업보고서 (2025.12)"
    assert len(saved["articles"]) == 3
    table = sc.build_scores_table()
    assert table.loc[0, "news_consensus"] == 2
    assert {f"news_{p}" for p in config.SCORING_PERSONAS} <= set(table.columns)


def test_점수_범위를_벗어나면_거부한다():
    with pytest.raises(ValueError):
        sc.Opinion(score=config.SCORE_MAX + 1, confidence=0.5, rationale="x", key_articles=[])


@pytest.mark.parametrize("bad_time", [pd.Timestamp("2026-03-11 06:15", tz="UTC"), pd.NaT])  # 15:15 정각, 시각 없음
def test_구간_밖이거나_시각_없는_기사는_어느_경로로든_막는다(monkeypatch, bad_time):
    monkeypatch.setattr(sc, "trading_days", lambda: pd.DatetimeIndex(pd.bdate_range("2026-03-09", "2026-03-13")))
    bad = make_articles(2)
    bad.loc[0, "published_utc"] = bad_time
    stub = StubLLM()
    with pytest.raises(ValueError):
        sc.score_day(T, model=stub, articles=bad, report=REPORT)  # 기사를 직접 넘겨도 검사한다
    assert stub.calls == []


def test_당일_접수_공시는_막는다(monkeypatch):
    monkeypatch.setattr(sc, "trading_days", lambda: pd.DatetimeIndex(pd.bdate_range("2026-03-09", "2026-03-13")))
    same_day = REPORT.copy()
    same_day["rcept_dt"] = T
    with pytest.raises(ValueError):
        sc.score_day(T, model=StubLLM(), articles=make_articles(), report=same_day)


def test_다른_모델_점수가_섞이면_막는다(monkeypatch):
    monkeypatch.setattr(sc, "trading_days", lambda: pd.DatetimeIndex(pd.bdate_range("2026-03-09", "2026-03-13")))
    sc.score_day(T, model=StubLLM(), articles=make_articles(), report=REPORT)
    monkeypatch.setattr(sc.llm, "scoring_model_id", lambda: "other-model")
    sc.score_day(pd.Timestamp("2026-03-12"), model=StubLLM(), articles=make_articles(0), report=REPORT)
    with pytest.raises(ValueError):
        sc.build_scores_table()


def test_상세_점수와_버전이_점수_표에_들어간다():
    sc.score_day(T, model=StubLLM(), articles=make_articles(), report=REPORT)
    table = sc.build_scores_table()
    row = table.iloc[0]
    assert row["scoring_version"] == config.SCORING_VERSION
    assert (row["news_samsung_score"], row["news_semis_score"], row["news_macro_score"], row["news_downside_risk"]) == (3, 1, -1, 2)
    assert row["news_downside_risk_max"] == 2


def test_채점_형식_버전이_섞이면_막는다(monkeypatch):
    sc.score_day(T, model=StubLLM(), articles=make_articles(), report=REPORT)
    monkeypatch.setattr(config, "SCORING_VERSION", 99)
    sc.score_day(pd.Timestamp("2026-03-12"), model=StubLLM(), articles=make_articles(0), report=REPORT)
    with pytest.raises(ValueError, match="버전"):
        sc.build_scores_table()
