import datetime
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src import config, llm
from src.pipeline.labels import DOWN, FLAT, UP


class KnownEvent(BaseModel):
    """모델이 알고 있다고 답한 사건."""

    date: str = Field(description="사건 날짜 YYYY-MM-DD. 모르면 빈 문자열")
    description: str = Field(description="사건 내용 한 문장")


class LeakAnswer(BaseModel):
    """유출 테스트 답변."""

    direction: Literal["상승", "보합", "하락"] = Field(description=f"{config.LABEL_HORIZON_DAYS}거래일 뒤 종가 방향")
    confidence: float = Field(ge=0, le=1, description="확신도 0~1")
    events_after: list[KnownEvent] = Field(description="기준 날짜 이후에 일어났다고 알고 있는 사건. 모르면 빈 목록")


def question(date: str, anonymize: bool) -> str:
    """기준 날짜 이후 지식을 묻는 질문. 익명화하면 회사명과 날짜를 숨긴다."""
    pct = round(config.LABEL_THRESHOLD * 100)
    if anonymize:
        target, when = "한국 시가총액 1위 반도체·전자 기업 A사", "기준일 D(연도 비공개)"
    else:
        target, when = f"삼성전자({config.SAMSUNG_CODE})", date
    return (f"{when} 장 마감 기준으로 {target} 종가가 {config.LABEL_HORIZON_DAYS}거래일 뒤 어떻게 될지 골라라. "
            f"+{pct}% 이상이면 상승, -{pct}% 이하면 하락, 그 사이면 보합이다. 반드시 하나를 고른다. "
            f"그리고 {when} 이후에 일어났다고 네가 알고 있는 경제·산업·{target} 관련 사건을 날짜와 함께 모두 적어라. 모르면 빈 목록으로 둔다.")


def pick_dates(labels: pd.DataFrame) -> list[pd.Timestamp]:
    """뉴스 구간에서 정답이 있는 거래일 중 고르게 LEAK_TEST_DAYS 개를 고른다."""
    days = labels.loc[config.CRITERION_B_START:config.CRITERION_B_END].index
    idx = np.linspace(0, len(days) - 1, config.LEAK_TEST_DAYS).round().astype(int)
    return list(days[np.unique(idx)])


def classify_event_date(text: str, as_of: pd.Timestamp) -> str:
    """사건 날짜를 'after'(기준일 당일 포함 이후), 'before', 'unclear'(빈 값·월까지만·형식 불명) 로 나눈다."""
    try:
        when = pd.Timestamp(datetime.date.fromisoformat(text.strip()))
    except ValueError:
        return "unclear"
    return "after" if when >= as_of.normalize() else "before"


def binomial_p_at_least(k: int, n: int, p: float) -> float:
    """이항분포에서 n 번 중 k 번 이상 맞힐 확률(한쪽 검정 p 값)."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def run(anonymize: bool = False, model=None) -> pd.DataFrame:
    """날짜마다 질문하고 정답과 비교한 표를 돌려준다."""
    labels = pd.read_parquet(Path(config.PROCESSED_DIR) / "labels.parquet")
    model = (model or llm.get_scoring_llm()).with_structured_output(LeakAnswer)
    rows = []
    for d in pick_dates(labels):
        ans = model.invoke([HumanMessage(question(str(d.date()), anonymize))])
        kinds = [classify_event_date(e.date, d) for e in ans.events_after]
        rows.append({
            "date": d, "answer": ans.direction, "truth": labels.loc[d, "label"], "confidence": ans.confidence,
            "events_after_dated": kinds.count("after"), "events_undated": kinds.count("unclear"),
            "events": "; ".join(f"{e.date} {e.description}" for e in ans.events_after),
        })
    return pd.DataFrame(rows)


def judge(result: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """통과 기준 ① 적중률이 가장 많은 정답 비율보다 통계적으로 높지 않다 ② 이후(당일 포함)이거나 날짜가 불분명한 사건 언급 0건."""
    share = labels.loc[config.CRITERION_B_START:config.CRITERION_B_END, "label"].value_counts(normalize=True)
    n, k = len(result), int((result["answer"] == result["truth"]).sum())
    p_value = binomial_p_at_least(k, n, share.max())
    events = int(result["events_after_dated"].sum() + result.get("events_undated", pd.Series(dtype=int)).sum())
    return {"n": n, "hits": k, "accuracy": k / n, "baseline_label": share.idxmax(), "baseline": share.max(),
            "p_value": p_value, "pass_accuracy": p_value >= config.LEAK_TEST_ALPHA,
            "events_after": events, "pass_events": events == 0,
            "passed": p_value >= config.LEAK_TEST_ALPHA and events == 0}


def write_report(results: dict[str, tuple[pd.DataFrame, dict]]) -> Path:
    """evaluation/reports/leak_test.md 에 판정과 날짜별 답을 쓴다."""
    today = pd.Timestamp.now(tz=config.KST).date()
    lines = ["# 미래 정보 유출 테스트", "", f"> 작성일: {today} · 모델: {llm.scoring_model_id()} · 기준: DAY2_PLAN.md 3절", ""]
    for name, (result, verdict) in results.items():
        lines += [f"## {name}", "",
                  f"- 판정: **{'통과' if verdict['passed'] else '실패'}**",
                  f"- ① 적중 {verdict['hits']}/{verdict['n']} ({verdict['accuracy']:.0%}). "
                  f"기준선 '{verdict['baseline_label']}' {verdict['baseline']:.1%}, 한쪽 이항검정 p = {verdict['p_value']:.3f} "
                  f"→ {'통과' if verdict['pass_accuracy'] else '실패'} (p ≥ {config.LEAK_TEST_ALPHA})",
                  f"- ② 기준 날짜 당일·이후이거나 날짜가 불분명한 사건 언급 {verdict['events_after']}건 → {'통과' if verdict['pass_events'] else '실패'}",
                  f"- 답 분포: {result['answer'].value_counts().to_dict()}",
                  "- 한계: 20일·기준선 61.7%에서는 17일 이상 맞혀야 ①이 실패한다. 모델이 한 답만 고르면 ①은 가려내는 힘이 약하다. "
                  "지식 기준 시점이 더 늦은 모델로 바꾸면 반드시 다시 돌리고, 실제 채점 프롬프트로도 시험한다.", ""]
        lines += ["| 날짜 | 답 | 정답 | 확신도 | 언급한 사건 |", "|---|---|---|---|---|"]
        for r in result.itertuples():
            lines.append(f"| {r.date.date()} | {r.answer} | {r.truth} | {r.confidence:.2f} | {r.events or '-'} |")
        lines.append("")
    out = Path("evaluation/reports/leak_test.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    """실명으로 시험하고, 실패하면 익명화해서 다시 시험한 뒤 보고서를 쓴다."""
    labels = pd.read_parquet(Path(config.PROCESSED_DIR) / "labels.parquet")
    results = {}
    plain = run(anonymize=False)
    results["실명 시험"] = (plain, judge(plain, labels))
    if not results["실명 시험"][1]["passed"]:
        anon = run(anonymize=True)
        results["익명화 재시험"] = (anon, judge(anon, labels))
    print(write_report(results))
    for name, (_, v) in results.items():
        print(name, "통과" if v["passed"] else "실패", f"적중 {v['hits']}/{v['n']}, p={v['p_value']:.3f}, 이후 사건 {v['events_after']}건")
    print(f"누적 비용 ${llm.total_cost():.4f}")


if __name__ == "__main__":
    main()
