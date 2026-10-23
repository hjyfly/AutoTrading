import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src import agent, server

TODAY = pd.Timestamp("2026-10-21")


@pytest.mark.parametrize("question, expected", [
    ("2026-03-11 삼성전자 신호는?", "2026-03-11"),
    ("2026년 3월 11일 신호 이유", "2026-03-11"),
    ("3월 11일에 왜 보합이었어?", "2026-03-11"),
    ("12월 3일 신호는?", "2025-12-03"),  # 연도가 없고 올해 날짜가 미래면 작년
    ("오늘 신호는?", None),
    ("3/11 신호는?", "2026-03-11"),
    ("3.11 신호는?", "2026-03-11"),
    ("20260311 신호", "2026-03-11"),
    ("2026.3.11 신호", "2026-03-11"),
    ("3월 둘째 주에 하락 신호가 나온 이유는?", "2026-03-06"),  # 3/1(일)이 든 주(2/23~3/1)가 첫째 주 → 둘째 주(3/2~3/8) 금요일
    ("3월 4일 신호를 2026-03-12 결과와 비교", "2026-03-04"),  # 표기가 섞여도 가장 이른 날짜
    ("2026-03-12 결과와 3월 4일 신호 비교", "2026-03-04"),
])
def test_질문에서_시점을_찾는다(question, expected):
    assert agent.extract_as_of(question, today=TODAY) == expected


def test_주_표기는_해석한_구간을_에이전트에_알린다():
    assert agent.period_notes("3월 둘째 주에 하락 신호가 나온 이유는?", today=TODAY) == [
        "'3월 둘째 주' = 2026-03-02 ~ 2026-03-06 (월~일 주, 1일이 든 주가 첫째 주)"]
    assert agent.period_notes("4월 첫째 주", today=TODAY)[0].startswith("'4월 첫째 주' = 2026-04-01 ~ 2026-04-03")  # 그 달 안만
    assert agent.period_notes("2026-03-11 신호", today=TODAY) == []
    late = agent.period_notes("3월 5일과 3월 다섯째 주 비교", today=TODAY, as_of="2026-03-05")
    assert late == ["'3월 다섯째 주' = 2026-03-23 ~ 2026-03-27 (월~일 주, 1일이 든 주가 첫째 주) — 이 가운데 2026-03-05 뒤 날짜는 이 시점에 알 수 없다"]


@pytest.mark.parametrize("question", ["2월 30일 신호는?", "2026-02-30 신호", "2월 29일 신호"])
def test_없는_날짜는_오류로_알린다(question):
    with pytest.raises(agent.InvalidDateError):
        agent.extract_as_of(question, today=TODAY)


def test_없는_날짜_질문은_LLM_없이_답한다():
    r = agent.answer("2월 30일 신호는?", model=object())
    assert "없는 날짜" in r["answer"] and r["contexts"] == []


def test_도구_결과와_호출_순서를_뽑는다():
    messages = [
        HumanMessage("3월 11일 신호"),
        AIMessage("", tool_calls=[{"id": "c1", "name": "get_signal", "args": {"date": "2026-03-11"}}]),
        ToolMessage(json.dumps({"date": "2026-03-11", "numeric": {"signal": "보합"}}), tool_call_id="c1"),
        AIMessage("보합입니다."),
    ]
    contexts, trace = agent.collect_contexts(messages)
    assert contexts == [{"tool": "get_signal", "args": {"date": "2026-03-11"},
                         "result": {"date": "2026-03-11", "numeric": {"signal": "보합"}}}]
    assert trace == [{"step": "tool_call", "tool": "get_signal", "args": {"date": "2026-03-11"}}]


def test_POST_query_는_세_키로_답한다(monkeypatch):
    monkeypatch.setattr(server.agent, "answer", lambda q: {"answer": f"답: {q}", "contexts": [], "trace": []})
    resp = TestClient(server.app).post("/query", json={"question": "오늘 신호는?"})
    assert resp.status_code == 200
    assert set(resp.json()) == {"answer", "contexts", "trace"}


def test_검색_도구도_요청_시점을_넘지_않는다(monkeypatch):
    seen = {}
    monkeypatch.setattr(agent.retriever, "search", lambda q, as_of, k, types: seen.update(as_of=as_of) or [])
    with agent.tools.request_as_of("2026-03-05"):
        out = agent.search_documents.invoke({"query": "근거", "as_of": "2026-03-20"})
    assert seen["as_of"] == "2026-03-05" and "as_of_note" in out


def test_에이전트_도구_목록에_검색_도구가_있다():
    assert "search_documents" in {t.name for t in agent.AGENT_TOOLS}
