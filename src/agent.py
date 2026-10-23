"""/query 질의응답 에이전트. 질문에서 시점(as_of)을 정하고, 그 시점에 알 수 있던 정보만으로 답한다."""
import json
import re

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src import config, llm, retriever, tools

FULL_DATE = re.compile(r"(?<!\d)(\d{4})[-./년]\s*(\d{1,2})[-./월]\s*(\d{1,2})일?(?!\d)")
COMPACT_DATE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")
MONTH_DAY = re.compile(r"(?<![\d.\-/])(\d{1,2})\s*월\s*(\d{1,2})\s*일")
SLASH_DAY = re.compile(r"(?<![\d.\-/])(\d{1,2})[/.](\d{1,2})(?![\d.\-/])")
MONTH_WEEK = re.compile(r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*월\s*(첫째|둘째|셋째|넷째|다섯째)\s*주")  # 그룹: 연도(없을 수 있음), 월, 몇째
WEEK_ORDINAL = {"첫째": 1, "둘째": 2, "셋째": 3, "넷째": 4, "다섯째": 5}
ORDINAL_WORD = {v: k for k, v in WEEK_ORDINAL.items()}


class InvalidDateError(ValueError):
    """질문에 있는 날짜가 달력에 없는 날짜다."""

SYSTEM_PROMPT = """너는 {horizon}거래일 방향 신호 시스템의 질의응답 에이전트다. 한국어로 답한다.

종목:
- 운영 종목은 {samsung}이고, 매매 대상은 {trade}뿐이다. {others}도 같은 정의의 정식 신호를 낸다(성적표로 판정).
- 질문에 종목이 없으면 운영 종목으로 본다. 다른 종목이면 get_signal·get_price·get_investor_flows·get_indicators 에 code(이름 또는 코드)를 넘긴다.
- 뉴스·판단 기록·공시·문서 검색은 {news}만, 과거 성과 요약은 운영 종목만 있다. 다른 종목에 대해 이것을 물으면 없다고 밝히고, 그 종목 신호·지표로 답할 수 있는 만큼 답한다.
- 이 종목들 밖의 종목은 신호를 내지 않는다고 밝힌다.

지금 질문의 시점은 {as_of} (한국 시각 {cutoff} 신호 시점)이다.
- 이 시점 이후의 가격·뉴스·공시·신호·성과는 모른다. 도구가 그보다 뒤를 조회하지 않게 되어 있고, 도구가 거부하면 거부했다고 답에 적는다.
- 질문 일부가 이 시점 이후에 관한 것이어도, 이 시점에 알 수 있는 부분(그날 신호·근거 등)은 빠짐없이 답한다.
- 학습 중에 알게 된 이후의 일도 쓰지 않는다. 도구 결과에 없는 사실은 지어내지 않는다.
{period_note}
예측 정의:
- 운영·매매 기준 신호는 {horizon}거래일 뒤 종가 방향(±{threshold:.0%} 기준 상승·보합·하락)이다.
- {samsung}는 참고용 기간별 신호도 있다: {horizons}. get_signal 에 horizon(거래일 수)을 넘겨 조회한다. 매매 기준은 {horizon}거래일 신호뿐이다.
- 질문이 '내일'(1거래일)·'2주 뒤'(10거래일)·'한 달 뒤'(20거래일)처럼 기간을 물으면 그 기간 신호를 조회해 답하고, 그것이 참고 신호이며 매매 기준은 {horizon}거래일이라고 밝힌다. 기간을 말하지 않으면 {horizon}거래일 신호로 답한다.
- 위 기간에 없는 기간을 물으면 답 첫머리에 지원하는 기간을 밝혀 바로잡은 뒤 가장 가까운 기간의 신호를 알려 준다. 다른 종목은 {horizon}거래일 신호만 있다.
- 예측은 틀릴 수 있고 매매는 사용자가 직접 판단해야 한다고 적는다.

도구 사용:
- 특정 날짜의 신호·이유를 물으면 get_signal 과 get_news_judgement 를 먼저 쓴다.
- 지표·가격·수급·거시 상황은 get_indicators, get_price, get_investor_flows, get_macro 로 확인한다.
- 이유·맥락·회사 정보는 search_documents(판단 기록·공시·보고서 검색)와 get_company_report 를 쓴다.
- 모델 성과 질문에만 get_backtest_summary 를 쓴다.

답 형식:
- 신호를 물으면 방향(상승·보합·하락), 확신도, 근거를 먼저 쓴다.
- 근거마다 어느 도구·날짜의 자료인지 밝힌다.
- 가격·수급·지표 같은 숫자 자료는 시점 전날 종가까지만 쓴다. 이런 자료를 답할 때는 마지막 날짜를 '전날(YYYY-MM-DD) 종가 기준'처럼 밝힌다.
- 투자 권유가 아니라 모델 판단의 설명이라는 점을 마지막에 한 줄 적는다.
"""


def _make_date(year: int, month: int, day: int) -> pd.Timestamp:
    """달력에 있는 날짜만 만든다. 없으면 InvalidDateError."""
    try:
        return pd.Timestamp(year, month, day)
    except ValueError as e:
        raise InvalidDateError(f"{year}-{month:02d}-{day:02d} 는 없는 날짜다") from e


def _with_year(month: int, day: int, today: pd.Timestamp) -> pd.Timestamp:
    """연도가 없는 날짜는 오늘 이전의 가장 가까운 해로 본다."""
    year = today.year if (month, day) <= (today.month, today.day) else today.year - 1
    return _make_date(year, month, day)


def _week_span(month: int, ordinal: int, today: pd.Timestamp, year: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """'[YYYY년] M월 N째 주'의 평일 구간(그 달 안의 월요일 ~ 금요일). 주는 월~일이고 1일이 든 주가 첫째 주다.
    연도가 없으면 오늘 이전의 가장 가까운 해. 1일이 토·일요일이면 첫째 주에 그 달 평일이 없어 InvalidDateError 로 다시 묻는다
    (전달 날짜로 넘어가지 않게)."""
    first = _make_date(year, month, 1) if year else _with_year(month, 1, today)
    week_start = first - pd.Timedelta(days=first.weekday()) + pd.Timedelta(weeks=ordinal - 1)
    friday = week_start + pd.Timedelta(days=4)
    month_end = first + pd.offsets.MonthEnd(0)
    if week_start > month_end:
        raise InvalidDateError(f"{month}월에는 {ordinal}째 주가 없다")
    if friday < first:
        raise InvalidDateError(f"{first.year}년 {month}월 1일이 주말이라 {ORDINAL_WORD[ordinal]} 주(1일이 든 주)에는 그 달 평일이 없다. 둘째 주나 날짜로 물어 달라")
    return max(week_start, first), min(friday, month_end)


def _week_end(month: int, ordinal: int, today: pd.Timestamp, year: int | None = None) -> pd.Timestamp:
    """'M월 N째 주'를 그 주의 금요일(그 달 안)로 본다. 그 주의 신호를 설명하려면 주말까지 알아야 한다."""
    return _week_span(month, ordinal, today, year)[1]


def period_notes(question: str, today: pd.Timestamp | None = None, as_of: str | None = None) -> list[str]:
    """질문의 'M월 N째 주' 표기를 어떤 날짜 구간으로 읽었는지 문장으로 돌려준다(에이전트가 같은 해석을 답에 쓰도록).
    구간 끝이 as_of 보다 뒤면 그 뒤는 이 시점에 알 수 없다고 함께 적는다."""
    today = today or tools.resolve_as_of(None)
    notes = []
    for m in MONTH_WEEK.finditer(question):
        start, end = _week_span(int(m[2]), WEEK_ORDINAL[m[3]], today, int(m[1]) if m[1] else None)
        note = f"'{m[0]}' = {start.date()} ~ {end.date()} (월~일 주, 1일이 든 주가 첫째 주)"
        if as_of and end > pd.Timestamp(as_of):
            note += f" — 이 가운데 {as_of} 뒤 날짜는 이 시점에 알 수 없다"
        notes.append(note)
    return notes


def extract_as_of(question: str, today: pd.Timestamp | None = None) -> str | None:
    """질문에 있는 날짜를 모두 찾아 가장 이른 날짜를 as_of 로 쓴다(보수적). 없으면 None(오늘).

    알아듣는 표기: 2026-03-11, 2026.3.11, 2026년 3월 11일, 20260311, 3월 11일, 3/11, 3.11, [2026년] 3월 둘째 주(그 주 금요일).
    연도가 없으면 오늘 이전의 가장 가까운 해. 달력에 없는 날짜면 InvalidDateError.
    """
    today = today or tools.resolve_as_of(None)
    found, taken = [], []

    def free(m):  # 이미 더 긴 표기로 읽은 자리는 다시 읽지 않는다
        return all(m.end() <= s or m.start() >= e for s, e in taken)

    for pattern, build in [
        (FULL_DATE, lambda m: _make_date(int(m[1]), int(m[2]), int(m[3]))),
        (COMPACT_DATE, lambda m: _make_date(int(m[1]), int(m[2]), int(m[3]))),
        (MONTH_WEEK, lambda m: _week_end(int(m[2]), WEEK_ORDINAL[m[3]], today, int(m[1]) if m[1] else None)),
        (MONTH_DAY, lambda m: _with_year(int(m[1]), int(m[2]), today)),
        (SLASH_DAY, lambda m: _with_year(int(m[1]), int(m[2]), today)),
    ]:
        for m in pattern.finditer(question):
            if free(m):
                found.append(build(m))
                taken.append((m.start(), m.end()))
    return str(min(found).date()) if found else None


@tool
def search_documents(query: str, as_of: str = "", types: str = "") -> dict:
    """판단 기록(judgement)·DART 사업 개요(dart)·보고서(report)에서 질문과 가까운 문서를 찾는다. 질문 시점 이후 문서는 나오지 않는다.
    types 는 쉼표로 구분한 문서 종류(비우면 전체)."""
    t, note = tools.capped(as_of or None)
    found = retriever.search(query, as_of=str(t.date()), k=config.RAG_TOP_K, types=[x.strip() for x in types.split(",") if x.strip()] or None)
    return tools._with_note({"as_of": str(t.date()), "documents": found}, note)


AGENT_TOOLS = [*tools.TOOLS, search_documents]


def collect_contexts(messages: list) -> tuple[list[dict], list[dict]]:
    """에이전트 메시지에서 도구 결과(contexts)와 도구 호출 순서(trace)를 뽑는다."""
    calls, contexts, trace = {}, [], []
    for m in messages:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                calls[c["id"]] = c
                trace.append({"step": "tool_call", "tool": c["name"], "args": c["args"]})
        elif isinstance(m, ToolMessage):
            call = calls.get(m.tool_call_id, {})
            try:
                result = json.loads(m.content)
            except (TypeError, ValueError):
                result = m.content
            contexts.append({"tool": call.get("name", m.name), "args": call.get("args", {}), "result": result})
    return contexts, trace


def answer(question: str, model=None) -> dict:
    """질문 하나에 답한다. 돌려주는 값은 제출 형식(answer, contexts, trace)이다.

    질문에 날짜가 여러 개면 가장 이른 날짜를 시점으로 쓴다(보수적). 시스템 프롬프트의 시점은 질문마다 채운다.
    """
    try:
        as_of = extract_as_of(question)
    except InvalidDateError as e:
        return {"answer": f"질문의 날짜를 알아들을 수 없다: {e}. 날짜를 다시 확인해 달라.", "contexts": [],
                "trace": [{"step": "answer", "as_of": None, "as_of_from_question": True, "cost_usd": 0.0, "error": str(e)}]}
    with tools.request_as_of(as_of) as cap:
        notes = period_notes(question, as_of=str(cap.date()))  # 상한으로 낮춘 시점 기준으로 '알 수 없는 날'을 알린다
        agent = create_agent(model or llm.get_agent_llm(), AGENT_TOOLS, system_prompt=SYSTEM_PROMPT.format(
            as_of=cap.date(), horizon=config.OPERATIONAL_HORIZON, threshold=config.HORIZONS[config.OPERATIONAL_HORIZON], cutoff=config.SIGNAL_CUTOFF_KST,
            horizons=", ".join(f"{config.HORIZON_LABELS[n]}={n}거래일 ±{config.HORIZONS[n]:.1%}" for n in config.HORIZONS if n != config.OPERATIONAL_HORIZON),
            samsung=f"{tools.ticker_name(config.MAIN_TICKER)}({config.MAIN_TICKER})",
            others="·".join(f"{v[0]}({c})" for c, v in config.MULTI_TICKERS.items() if c != config.MAIN_TICKER),
            trade=tools.role_names(config.TRADE_TICKERS), news=tools.role_names(config.NEWS_TICKERS),
            period_note="".join(f"- 질문의 기간 표기는 이렇게 해석했다: {n}. 답 첫머리에 이 해석을 밝히고, 다른 날짜로 다시 해석하지 않는다.\n" for n in notes)))
        spent_before = llm.total_cost()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    contexts, trace = collect_contexts(messages)
    final = messages[-1]
    text = final.content if isinstance(final.content, str) else "".join(
        part.get("text", "") for part in final.content if isinstance(part, dict))
    trace.append({"step": "answer", "as_of": str(cap.date()), "as_of_from_question": as_of is not None,
                  "cost_usd": round(llm.total_cost() - spent_before, 6)})
    return {"answer": text, "contexts": contexts, "trace": trace}
