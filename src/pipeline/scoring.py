import json
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src import config, llm
from src.pipeline.collect_dart import dart_as_of
from src.pipeline.collect_news import signal_window, trading_days

KST = config.KST


class Opinion(BaseModel):
    """채점자 1차 의견."""

    score: int = Field(ge=config.SCORE_MIN, le=config.SCORE_MAX, description=f"앞으로 {config.LABEL_HORIZON_DAYS}거래일 삼성전자 주가에 대한 뉴스 영향. -5 매우 부정, 0 영향 없음 또는 상쇄, +5 매우 긍정")
    confidence: float = Field(ge=0, le=1, description="이 점수에 대한 확신도 0~1")
    rationale: str = Field(description="판단 근거. 한국어 3문장 이내")
    key_articles: list[int] = Field(description="판단에 가장 큰 영향을 준 기사 번호. 최대 5개")


class DebateOpinion(Opinion):
    """다른 채점자 의견을 본 뒤의 의견."""

    change_reason: str = Field(description="1차 의견에서 점수나 확신도를 바꿨다면 그 이유, 유지했다면 유지한 이유. 한국어 2문장 이내")


class Consensus(BaseModel):
    """중재자의 최종 정리."""

    score: int = Field(ge=config.SCORE_MIN, le=config.SCORE_MAX, description="세 채점자 토론을 종합한 최종 점수")
    confidence: float = Field(ge=0, le=1, description="최종 점수에 대한 확신도 0~1")
    rationale: str = Field(description="최종 판단 근거. 한국어 3문장 이내")
    key_articles: list[int] = Field(description="핵심 기사 번호. 최대 5개")
    disagreement: str = Field(description="채점자 사이에 남은 이견. 없으면 '없음'")


class ScoringState(TypedDict):
    signal_date: str
    articles_text: str
    company_text: str
    round1: Annotated[dict, operator.or_]
    round2: Annotated[dict, operator.or_]
    final: dict


def time_header(signal_date: str) -> str:
    """모든 프롬프트 첫머리에 넣는 기준 시각 조건."""
    return (f"지금은 {signal_date} {config.SIGNAL_CUTOFF_KST} (한국 시각)이다. 너는 이 시각 이후에 일어난 사건, 가격, 공시를 전혀 모른다. "
            "학습 중에 알게 된 이후의 일이 떠올라도 쓰지 말고, 아래 자료만 근거로 판단하라.")


def format_articles(articles: pd.DataFrame) -> str:
    """기사를 '[번호] (발행 시각 KST, 매체) 제목' 줄로 만든다."""
    lines = []
    for i, row in enumerate(articles.itertuples(), start=1):
        when = pd.Timestamp(row.published_utc).tz_convert(KST).strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{i}] ({when} KST, {row.source}) {row.title}")
    return "\n".join(lines)


def format_company(report: pd.Series | None) -> str:
    """그 시점에 공개된 가장 최근 정기보고서의 사업 개요를 회사 정보로 만든다."""
    if report is None:
        return "(이 시점에 공개된 정기보고서 없음)"
    return f"[{report['report_nm']}, 접수일 {pd.Timestamp(report['rcept_dt']).date()}]\n{report['overview']}"


TASK = (f"삼성전자 주가가 앞으로 {config.LABEL_HORIZON_DAYS}거래일 동안 위 뉴스들의 영향을 어떻게 받을지 {config.SCORE_MIN}~{config.SCORE_MAX} 점수로 매겨라. "
        "기사 제목만 있으니 제목에서 읽을 수 있는 만큼만 판단하고, 삼성전자와 관련이 적은 기사는 무시하라.")


def _persona_messages(persona: str, state: ScoringState) -> list:
    """채점자 1차 의견 요청 메시지."""
    system = f"{time_header(state['signal_date'])}\n너는 {config.SCORING_PERSONAS[persona]}"
    human = (f"{time_header(state['signal_date'])}\n\n## 회사 정보\n{state['company_text']}\n\n"
             f"## 뉴스 (t-1 {config.SIGNAL_CUTOFF_KST} ~ t {config.SIGNAL_CUTOFF_KST})\n{state['articles_text']}\n\n## 할 일\n{TASK}")
    return [SystemMessage(system), HumanMessage(human)]


def _opinions_text(opinions: dict) -> str:
    """채점자 의견들을 토론·정리 단계에 보여 줄 글로 만든다."""
    return "\n".join(f"- {name}: 점수 {o['score']}, 확신도 {o['confidence']}, 근거: {o['rationale']}" for name, o in opinions.items())


def build_graph(model):
    """채점자 3명(병렬) → 토론 3명(병렬) → 중재자 순서의 LangGraph 를 만든다. model 은 with_structured_output 을 지원해야 한다."""
    personas = list(config.SCORING_PERSONAS)

    def round1(persona):
        def node(state: ScoringState):
            """채점자 1차 의견."""
            out = model.with_structured_output(Opinion).invoke(_persona_messages(persona, state))
            return {"round1": {persona: out.model_dump()}}
        return node

    def round2(persona):
        def node(state: ScoringState):
            """다른 채점자 의견을 본 뒤 의견을 유지하거나 바꾼다."""
            messages = _persona_messages(persona, state)
            messages[-1] = HumanMessage(messages[-1].content + "\n\n## 세 채점자의 1차 의견\n" + _opinions_text(state["round1"]) +
                                        f"\n\n너({persona})의 1차 의견을 다른 채점자 의견과 비교해, 판단을 유지하거나 바꿔라.")
            out = model.with_structured_output(DebateOpinion).invoke(messages)
            return {"round2": {persona: out.model_dump()}}
        return node

    def moderator(state: ScoringState):
        """토론 뒤 의견을 종합해 최종 점수를 낸다."""
        system = f"{time_header(state['signal_date'])}\n너는 세 채점자의 토론을 정리하는 중재자다. 다수결이 아니라 근거의 설득력으로 종합하라."
        human = (f"{time_header(state['signal_date'])}\n\n## 뉴스\n{state['articles_text']}\n\n"
                 f"## 토론 뒤 세 채점자 의견\n{_opinions_text(state['round2'])}\n\n## 할 일\n{TASK} 세 의견을 종합해 최종 점수를 내라.")
        out = model.with_structured_output(Consensus).invoke([SystemMessage(system), HumanMessage(human)])
        return {"final": out.model_dump()}

    graph = StateGraph(ScoringState)
    for p in personas:
        graph.add_node(f"r1_{p}", round1(p))
        graph.add_node(f"r2_{p}", round2(p))
        graph.add_edge(START, f"r1_{p}")
    for p in personas:
        graph.add_edge([f"r1_{q}" for q in personas], f"r2_{p}")
    graph.add_node("moderator", moderator)
    graph.add_edge([f"r2_{p}" for p in personas], "moderator")
    graph.add_edge("moderator", END)
    return graph.compile()


def validate_inputs(signal_date, articles: pd.DataFrame, report: pd.Series | None):
    """채점 입력이 t 15:15 에 알 수 있던 것뿐인지 확인한다. 어떤 경로로 들어온 입력이든 score_day 가 항상 부른다.

    기사: 발행 시각이 있고 [t-1 15:15, t 15:15) KST 안. 공시: t 보다 먼저 접수.
    """
    t = pd.Timestamp(signal_date)
    if not articles.empty:
        start, end = signal_window(t, trading_days())
        published = pd.to_datetime(articles["published_utc"], utc=True)
        if (published.isna() | (published < start) | (published >= end)).any():
            raise ValueError(f"{t.date()} 신호 구간 밖이거나 발행 시각이 없는 기사가 채점 입력에 있다")
    if report is not None and not pd.Timestamp(report["rcept_dt"]) < t.normalize():
        raise ValueError(f"{t.date()} 이후(당일 포함) 접수된 공시가 채점 입력에 있다")


def load_inputs(signal_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series | None]:
    """신호일 t 의 선별 기사와 t 이전에 접수된 가장 최근 정기보고서를 읽는다."""
    t = pd.Timestamp(signal_date)
    selected = pd.read_parquet(Path(config.PROCESSED_DIR) / "selected_news.parquet")
    articles = selected[selected["signal_date"] == t].sort_values("published_utc").reset_index(drop=True)
    reports = dart_as_of(pd.read_parquet(Path(config.PROCESSED_DIR) / "dart_index.parquet"), t)
    report = reports.sort_values("rcept_dt").iloc[-1] if len(reports) else None
    return articles, report


def score_day(signal_date, model=None, articles: pd.DataFrame | None = None, report: pd.Series | None = None) -> dict:
    """신호일 하나를 채점하고 판단 기록 JSON 을 남긴다. 기사가 없으면 LLM 을 부르지 않고 점수 0 을 기록한다."""
    t = pd.Timestamp(signal_date)
    if articles is None:
        articles, report = load_inputs(t)
    validate_inputs(t, articles, report)
    trace = {
        "signal_date": str(t.date()),
        "as_of": f"{t.date()} {config.SIGNAL_CUTOFF_KST} KST",
        "model": llm.scoring_model_id(),
        "company": None if report is None else {"report_nm": report["report_nm"], "rcept_dt": str(pd.Timestamp(report["rcept_dt"]).date())},
        "articles": [{"no": i, "title": r.title, "source": r.source, "url": r.url, "category": r.category,
                      "published_kst": pd.Timestamp(r.published_utc).tz_convert(KST).strftime("%Y-%m-%d %H:%M")}
                     for i, r in enumerate(articles.itertuples(), start=1)],
        "no_news": articles.empty,
    }
    if articles.empty:
        zero = {"score": 0, "confidence": 0.0, "rationale": "뉴스 없음", "key_articles": []}
        trace.update(round1={}, round2={}, final={**zero, "disagreement": "없음"}, cost_usd=0.0)
    else:
        model = model or llm.get_scoring_llm()
        spent_before = llm.total_cost()
        result = build_graph(model).invoke({
            "signal_date": str(t.date()),
            "articles_text": format_articles(articles),
            "company_text": format_company(report),
            "round1": {}, "round2": {}, "final": {},
        })
        trace.update(round1=result["round1"], round2=result["round2"], final=result["final"],
                     cost_usd=round(llm.total_cost() - spent_before, 6))
    path = Path(config.TRACES_DIR) / f"{t.date()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace


def trace_to_row(trace: dict) -> dict:
    """판단 기록을 점수 표 한 줄로 바꾼다. 모델 입력은 채점자 3명의 토론 뒤 점수와 합의 점수다."""
    row = {"signal_date": pd.Timestamp(trace["signal_date"]), "model": trace["model"],
           "n_articles": len(trace["articles"]), "no_news": trace["no_news"]}
    for p in config.SCORING_PERSONAS:
        row[f"news_{p}_r1"] = trace["round1"].get(p, {}).get("score", 0)
        row[f"news_{p}"] = trace["round2"].get(p, {}).get("score", 0)
    row["news_consensus"] = trace["final"]["score"]
    row["news_consensus_conf"] = trace["final"]["confidence"]
    row["cost_usd"] = trace["cost_usd"]
    return row


def build_scores_table() -> pd.DataFrame:
    """판단 기록 JSON 중 뉴스 구간(NEWS_START ~ NEWS_END) 것으로 점수 표를 다시 만들어 저장한다."""
    traces = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(config.TRACES_DIR).glob("*.json"))]
    traces = [t for t in traces if config.NEWS_START <= t["signal_date"] <= config.NEWS_END]  # 실시간 채점 기록은 빼고 뉴스 구간만
    table = pd.DataFrame([trace_to_row(t) for t in traces]).sort_values("signal_date").reset_index(drop=True)
    if table["model"].nunique() > 1:  # 모델마다 아는 시점이 달라 섞이면 유출 위험을 구분할 수 없다
        raise ValueError(f"서로 다른 모델의 채점이 섞였다: {sorted(table['model'].unique())}")
    table.to_parquet(config.SCORES_PATH)
    return table


def main(dates: list[str] | None = None):
    """주어진 신호일(없으면 선별 기사가 있는 모든 날)을 채점한다. 이미 판단 기록이 있는 날은 건너뛴다."""
    selected = pd.read_parquet(Path(config.PROCESSED_DIR) / "selected_news.parquet")
    targets = [pd.Timestamp(d) for d in dates] if dates else sorted(selected["signal_date"].unique())
    for t in targets:
        if (Path(config.TRACES_DIR) / f"{pd.Timestamp(t).date()}.json").exists():
            print(pd.Timestamp(t).date(), "이미 채점됨")
            continue
        trace = score_day(t)
        f = trace["final"]
        print(pd.Timestamp(t).date(), f"합의 {f['score']:+d} (확신도 {f['confidence']:.2f}),",
              "채점자", {p: o["score"] for p, o in trace["round2"].items()}, f"비용 ${trace['cost_usd']:.4f}", flush=True)
    table = build_scores_table()
    print(config.SCORES_PATH, len(table), f"누적 비용 ${llm.total_cost():.4f}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
