import json
import operator
import re
import sys
import time
from pathlib import Path
from typing import Annotated, Literal, TypedDict

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
    samsung_score: int = Field(ge=config.SCORE_MIN, le=config.SCORE_MAX, description="삼성전자 직접 뉴스만 봤을 때의 점수. 해당 기사가 없으면 0")
    semis_score: int = Field(ge=config.SCORE_MIN, le=config.SCORE_MAX, description="반도체·AI 업황 뉴스만 봤을 때의 점수. 해당 기사가 없으면 0")
    macro_score: int = Field(ge=config.SCORE_MIN, le=config.SCORE_MAX, description="거시(유가·금리·환율·관세·중국) 뉴스만 봤을 때의 점수. 해당 기사가 없으면 0")
    downside_risk: int = Field(ge=config.RISK_MIN, le=config.RISK_MAX, description="방향과 별개로, 앞으로 크게 떨어질 위험이 얼마나 큰가. 0 거의 없음 ~ 5 매우 큼")


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
    samsung_score: int = Field(ge=config.SCORE_MIN, le=config.SCORE_MAX, description="삼성전자 직접 뉴스 종합 점수")
    semis_score: int = Field(ge=config.SCORE_MIN, le=config.SCORE_MAX, description="반도체·AI 업황 뉴스 종합 점수")
    macro_score: int = Field(ge=config.SCORE_MIN, le=config.SCORE_MAX, description="거시 뉴스 종합 점수")
    downside_risk: int = Field(ge=config.RISK_MIN, le=config.RISK_MAX, description="종합 하락 위험 0~5")


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
        "기사 제목만 있으니 제목에서 읽을 수 있는 만큼만 판단하고, 삼성전자와 관련이 적은 기사는 무시하라. "
        "종합 점수와 함께 분류별 점수(삼성전자 직접, 반도체·AI 업황, 거시)를 따로 매기고, "
        f"방향과 별개로 크게 떨어질 위험을 {config.RISK_MIN}~{config.RISK_MAX}로 매겨라.")


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
        "scoring_version": config.SCORING_VERSION,
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
    row["scoring_version"] = trace.get("scoring_version", 1)
    for key in [*[f"{c}_score" for c in config.NEWS_CATEGORIES], "downside_risk"]:  # 버전 2 상세 점수(버전 1 기록에는 없음)
        row[f"news_{key}"] = trace["final"].get(key)
    risks = [o.get("downside_risk") for o in trace["round2"].values() if o.get("downside_risk") is not None]
    row["news_downside_risk_max"] = max(risks) if risks else None  # 채점자 중 가장 높은 하락 위험
    row["cost_usd"] = trace["cost_usd"]
    return row


def build_scores_table() -> pd.DataFrame:
    """판단 기록 JSON 중 뉴스 구간(NEWS_START ~ NEWS_END) 것으로 점수 표를 다시 만들어 저장한다."""
    traces = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(config.TRACES_DIR).glob("*.json"))]
    traces = [t for t in traces if config.NEWS_START <= t["signal_date"] <= config.NEWS_END]  # 실시간 채점 기록은 빼고 뉴스 구간만
    table = pd.DataFrame([trace_to_row(t) for t in traces]).sort_values("signal_date").reset_index(drop=True)
    if table["scoring_version"].nunique() > 1:  # 형식이 다른 채점이 섞이면 상세 점수가 일부만 비게 된다
        raise ValueError(f"채점 형식 버전이 섞였다: {sorted(table['scoring_version'].unique())}")
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
        trace = None
        for attempt in range(config.THROTTLE_RETRIES):
            try:
                trace = score_day(t)
                break
            except Exception as e:  # Bedrock 요청 제한이면 기다렸다 다시, 그 밖의 오류는 그대로 알린다
                if "Throttling" not in type(e).__name__ and "Throttling" not in str(e):
                    raise
                print(pd.Timestamp(t).date(), f"요청 제한, {config.THROTTLE_WAIT_SEC}초 뒤 다시 ({attempt + 1}/{config.THROTTLE_RETRIES})", flush=True)
                time.sleep(config.THROTTLE_WAIT_SEC)
        if trace is None:
            print("요청 제한이 풀리지 않아 멈춘다. 다시 실행하면 이어서 채점한다", flush=True)
            break
        f = trace["final"]
        print(pd.Timestamp(t).date(), f"합의 {f['score']:+d} (확신도 {f['confidence']:.2f}),",
              "채점자", {p: o["score"] for p, o in trace["round2"].items()}, f"비용 ${trace['cost_usd']:.4f}", flush=True)
    table = build_scores_table()
    print(config.SCORES_PATH, len(table), f"누적 비용 ${llm.total_cost():.4f}")


# ---- 9일차: 기사별 구조화 추출 (DAY9_PLAN.md 3.1). 하루 1회 호출, 회사명 익명화, 결과는 날짜별로 캐시한다 ----

class ArticleFeature(BaseModel):
    """기사 하나의 구조화 피처. 근거를 먼저 쓴다."""

    no: int = Field(description="기사 번호([번호])")
    reasoning: str = Field(description="이 기사를 아래처럼 판단한 근거. 한국어 1~2문장")
    relevance: float = Field(ge=0, le=1, description=f"{config.ANONYMIZED_NAME} 주가와의 관련도 0~1")
    category: Literal[tuple(config.ARTICLE_CATEGORIES)] = Field(
        description=f"samsung_direct={config.ANONYMIZED_NAME} 직접, semi_ai=반도체·AI 업황, macro=거시(금리·환율·유가·관세·중국), irrelevant=무관")
    event_type: Literal[tuple(config.ARTICLE_EVENTS)] = Field(description="사건 종류")
    direction: int = Field(ge=-1, le=1, description=f"{config.ANONYMIZED_NAME} 주가에 -1 부정, 0 중립, +1 긍정")
    magnitude: int = Field(ge=1, le=3, description="영향 크기 1~3(기준표)")
    novelty: Literal["new", "follow_up", "rehash"] = Field(description="new=처음 나온 사실, follow_up=앞선 보도의 후속, rehash=반복·재탕")
    downside_risk: int = Field(ge=0, le=3, description="방향과 별개로 급락 위험 기여도 0~3(기준표)")
    one_line_summary: str = Field(description="한 줄 요약(한국어, 기사 제목에 있는 사실만)")


class DayFeatures(BaseModel):
    """그날 기사 전체의 피처."""

    articles: list[ArticleFeature]


FEATURE_TASK = (
    f"위 기사마다 {config.ANONYMIZED_NAME}(한국 대형 반도체·전자 기업) 주가에 대한 피처를 뽑아라. 기사 제목만 있으니 제목에서 읽을 수 있는 만큼만 판단하고, "
    "근거(reasoning)를 먼저 쓴 뒤 값을 정하라. 모든 기사 번호에 대해 하나씩 답하라.\n"
    "크기 기준: 3 = 실적·가이던스의 큰 서프라이즈, 핵심 고객 공급 확정·이탈, 새 수출 규제 / 2 = 의미 있는 업황 변화, 경쟁사 주요 사건, 정책 방향 변화 / "
    "1 = 예상된 내용, 업황 일반론, 반복 보도.\n"
    "하락 위험 기준: 3 = 수출 규제·제재, 실적 쇼크, 대형 소송·리콜, 지정학 급변 / 2 = 의미 있는 악재 가능성 / 1 = 작은 악재 / 0 = 없음.\n"
    "새로움은 아래 '직전 신호일 기사'와 비교해 정하라. 직전에 이미 나온 사실이면 follow_up 이나 rehash 다.")


def anonymize(text: str) -> str:
    """회사명을 익명으로 바꾼다(학습 데이터 안 구간에서 모델이 기억한 이후 사건을 떠올리지 않게)."""
    for term in sorted(config.ANONYMIZE_TERMS, key=len, reverse=True):
        text = re.sub(re.escape(term), config.ANONYMIZED_NAME, text, flags=re.IGNORECASE)
    return text


def feature_context(t: pd.Timestamp, selected: pd.DataFrame, close: pd.Series) -> str:
    """새로움 판단용 맥락: 직전 신호일들의 기사 제목(익명화)과 t-1 종가까지의 5·20일 수익률. 모두 t 전에 알 수 있던 것이다."""
    before = sorted(d for d in selected["signal_date"].unique() if d < t)[-config.FEATURE_CONTEXT_DAYS:]
    lines = []
    for d in before:
        day = selected[selected["signal_date"] == d].sort_values("published_utc").tail(config.FEATURE_CONTEXT_TITLES)
        lines.append(f"[{pd.Timestamp(d).date()}] " + " / ".join(anonymize(x) for x in day["title"]))
    close = close[close.index < t]
    rets = {n: close.iloc[-1] / close.iloc[-1 - n] - 1 for n in (5, 20) if len(close) > n}
    ret_text = ", ".join(f"{n}일 {r:+.1%}" for n, r in rets.items()) or "없음"
    return ("## 직전 신호일 기사(새로움 판단용)\n" + ("\n".join(lines) or "(없음)") +
            f"\n\n## {config.ANONYMIZED_NAME} 최근 수익률(전날 종가 기준)\n{ret_text}")


def _day_news(t: pd.Timestamp, raw_news: pd.DataFrame | None) -> pd.Series:
    """그날 모은 전체 기사 제목. 실시간이면 넘겨받은 기사, 과거면 news.parquet 에서 읽는다."""
    if raw_news is not None:
        return raw_news["title"].fillna("")
    path = Path(config.PROCESSED_DIR) / "news.parquet"
    if not path.exists():
        return pd.Series(dtype=str)
    news = pd.read_parquet(path, columns=["signal_date", "title"])
    return news.loc[news["signal_date"] == t, "title"].fillna("")


def _cluster_sizes(articles: pd.DataFrame, day_titles: pd.Series) -> list[int]:
    """기사마다 그날 모은 전체 기사 중 같은 제목 키를 가진 기사 수(보도량)."""
    from src.pipeline.select_news import _dedup_key
    keys = day_titles.map(_dedup_key).value_counts()
    return [int(keys.get(_dedup_key(x), 1)) for x in articles["title"].fillna("")]


def context_articles() -> pd.DataFrame:
    """직전 신호일 맥락용 기사: 선별 기사 파일에 판단 기록(실시간 채점일)의 기사를 이어 붙인다. 모두 그날 15:15 전 발행분이다."""
    selected = pd.read_parquet(Path(config.PROCESSED_DIR) / "selected_news.parquet")
    last = selected["signal_date"].max()
    rows = []
    for path in sorted(Path(config.TRACES_DIR).glob("*.json")):
        if pd.Timestamp(path.stem) <= last:
            continue
        trace = json.loads(path.read_text(encoding="utf-8"))
        rows += [{"signal_date": pd.Timestamp(trace["signal_date"]), "title": a["title"],
                  "published_utc": pd.Timestamp(a["published_kst"]).tz_localize(KST).tz_convert("UTC")} for a in trace["articles"]]
    return pd.concat([selected, pd.DataFrame(rows)], ignore_index=True) if rows else selected


def feature_path(t: pd.Timestamp) -> Path:
    """그날 기사 피처 캐시 파일."""
    return Path(config.FEATURES_DIR) / f"{pd.Timestamp(t).date()}.json"


def extract_day(signal_date, model=None, articles: pd.DataFrame | None = None, close: pd.Series | None = None,
                selected: pd.DataFrame | None = None, raw_news: pd.DataFrame | None = None) -> dict:
    """신호일 하나의 기사별 피처를 뽑아 캐시한다. 같은 프롬프트 버전·모델의 캐시가 있으면 LLM 을 부르지 않는다.
    입력은 t 15:15 전 기사(validate_inputs), 직전 신호일 제목, t-1 종가까지의 수익률뿐이다.
    실시간 채점에서는 선별 기사(articles)와 그날 모은 전체 기사(raw_news, 보도량용)를 넘긴다."""
    from src import tools
    t = pd.Timestamp(signal_date)
    path = feature_path(t)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("prompt_version") == config.FEATURE_PROMPT_VERSION and cached.get("model") == llm.scoring_model_id():
            return cached
    selected = context_articles() if selected is None else selected
    if articles is None:
        articles = selected[selected["signal_date"] == t].sort_values("published_utc").reset_index(drop=True)
    validate_inputs(t, articles, None)
    close = tools.series_until("prices", "samsung_ohlcv", t)[0]["Close"].astype(float) if close is None else close
    record = {"signal_date": str(t.date()), "as_of": f"{t.date()} {config.SIGNAL_CUTOFF_KST} KST", "model": llm.scoring_model_id(),
              "prompt_version": config.FEATURE_PROMPT_VERSION, "anonymized": True, "articles": [], "missing": [], "cost_usd": 0.0}
    day_titles = _day_news(t, raw_news)
    record["n_collected"] = int(len(day_titles))  # 보도량(그날 15:15 까지 모은 기사 수)
    if not articles.empty:
        model = model or llm.get_scoring_llm()
        header = anonymize(time_header(str(t.date())))
        text = "\n".join(anonymize(line) for line in format_articles(articles).split("\n"))
        messages = [SystemMessage(f"{header}\n너는 뉴스를 기사 단위로 분석해 주가 피처를 뽑는 분석가다."),
                    HumanMessage(f"{header}\n\n## 뉴스 (t-1 {config.SIGNAL_CUTOFF_KST} ~ t {config.SIGNAL_CUTOFF_KST})\n{text}\n\n"
                                 f"{feature_context(t, selected, close)}\n\n## 할 일\n{FEATURE_TASK}")]
        spent_before = llm.total_cost()
        out, n, attempts, error = None, len(articles), 0, None
        while attempts < config.FEATURE_MAX_ATTEMPTS:  # 형식 오류면 다시 묻고, 번호가 빠지면 한 번 더 묻고, 그래도 모자라면 받은 것만 쓴다
            attempts += 1
            try:
                got = model.with_structured_output(DayFeatures).invoke(messages)
            except Exception as e:  # 구조화 출력 검증 실패(허용 밖 값, 목록 대신 문자열 등). 요청 제한은 위로 올린다
                if "Throttling" in type(e).__name__ or "Throttling" in str(e):
                    raise
                error = f"{type(e).__name__}: {str(e)[:200]}"
                continue
            out = got if out is None or len(got.articles) > len(out.articles) else out
            nos = [a.no for a in out.articles if 1 <= a.no <= n]
            if sorted(set(nos)) == list(range(1, n + 1)) or attempts >= 2:
                break
        if out is None:
            raise ValueError(f"{t.date()} 기사 피처 형식 오류가 {attempts}번 이어졌다: {error}")
        by_no = {a.no: a.model_dump() for a in out.articles if 1 <= a.no <= n}
        sizes = _cluster_sizes(articles, day_titles)
        for i, r in enumerate(articles.itertuples(), start=1):
            if i not in by_no:
                record["missing"].append(i)
                continue
            record["articles"].append({**by_no[i], "title": r.title, "source": r.source, "url": r.url,
                                       "published_kst": pd.Timestamp(r.published_utc).tz_convert(KST).strftime("%Y-%m-%d %H:%M"),
                                       "cluster_size": sizes[i - 1]})
        record["cost_usd"] = round(llm.total_cost() - spent_before, 6)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main_features(dates: list[str] | None = None):
    """주어진 신호일(없으면 뉴스 구간의 선별 기사가 있는 모든 날)의 기사 피처를 뽑는다. 캐시가 있는 날은 건너뛴다."""
    selected = pd.read_parquet(Path(config.PROCESSED_DIR) / "selected_news.parquet")
    days = sorted(d for d in selected["signal_date"].unique() if config.NEWS_START <= str(pd.Timestamp(d).date()) <= config.NEWS_END)
    targets = [pd.Timestamp(d) for d in dates] if dates else days
    for t in targets:
        record = None
        for attempt in range(config.THROTTLE_RETRIES):
            try:
                record = extract_day(t, selected=selected)
                break
            except ValueError as e:  # 형식 오류가 이어진 날은 건너뛴다(다시 실행하면 그날만 다시 뽑는다)
                print(pd.Timestamp(t).date(), f"건너뜀: {e}", flush=True)
                record = {}
                break
            except Exception as e:  # Bedrock 요청 제한이면 기다렸다 다시, 그 밖의 오류는 그대로 알린다
                if "Throttling" not in type(e).__name__ and "Throttling" not in str(e):
                    raise
                print(pd.Timestamp(t).date(), f"요청 제한, {config.THROTTLE_WAIT_SEC}초 뒤 다시 ({attempt + 1}/{config.THROTTLE_RETRIES})", flush=True)
                time.sleep(config.THROTTLE_WAIT_SEC)
        if record is None:
            print("요청 제한이 풀리지 않아 멈춘다. 다시 실행하면 이어서 뽑는다", flush=True)
            break
        if not record:
            continue
        print(pd.Timestamp(t).date(), f"기사 {len(record['articles'])}건, 빠짐 {len(record['missing'])}, 비용 ${record['cost_usd']:.4f}", flush=True)
    print(f"누적 비용 ${llm.total_cost():.4f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["features"]:
        main_features(sys.argv[2:] or None)
    else:
        main(sys.argv[1:] or None)
