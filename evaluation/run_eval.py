"""test_queries.csv 로 /query 에이전트를 평가한다. 자동 검사(시점·도구·방향·유출)와 judge 채점을 함께 한다.

실행: .venv/Scripts/python -m evaluation.run_eval
"""
import json
import re
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src import agent, config, llm, tools

QUERIES = Path("evaluation/test_queries.csv")
DETAILS = Path("data/scores/eval/query_eval.json")  # 기사 제목이 들어 있어 git 제외
REPORT = Path("evaluation/reports/query_eval.md")
CONTEXT_CHARS = 8000


class JudgeScore(BaseModel):
    """judge 의 채점."""

    faithfulness: int = Field(ge=1, le=5, description="답이 도구 결과(contexts)에만 근거하는가. 5 = 모든 주장이 근거 있음")
    relevance: int = Field(ge=1, le=5, description="질문에 직접 답했는가. 5 = 완전히 답함")
    meets_note: bool = Field(description="기대 사항(note)을 지켰는가")
    temporal_violation: bool = Field(description="질문 시점 이후의 가격·결과·사건을 답에 썼는가")
    comment: str = Field(description="판단 이유. 한국어 2문장 이내")


JUDGE_SYSTEM = ("너는 주식 신호 질의응답 시스템의 평가자다. 답이 도구 결과에 근거하는지, 질문에 답했는지, 기대 사항을 지켰는지, "
                "질문 시점 이후 정보를 썼는지 엄격하게 채점한다. 투자 의견의 옳고 그름은 평가하지 않는다.")


DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# 값이 질문 시점(t)과 같아도 되는 키: 시점 자체, 신호일, 신호일에 만든 판단 기록
SAME_DAY_OK = {"as_of", "signal_row_date", "date", "model_block_start", "window_kst"}


def violations(result, as_of: str) -> list[str]:
    """도구 결과 전체를 훑어 시점 규칙을 어긴 날짜를 모은다. 거부된 결과는 검사하지 않는다.

    - 데이터 날짜(rows 의 date, rcept_dt, last_date, train_label_last_date 등)는 t 보다 앞서야 한다.
    - 시점·신호일 키(as_of, date, signal_row_date)는 t 와 같아도 된다. 뒤면 위반.
    - 뉴스 published_kst 는 t 15:15 보다 앞서야 한다. 문서 available_from 은 t 이하.
    - 결과가 문자열이면 그 안의 날짜가 t 보다 뒤면 위반으로 본다.
    """
    cutoff = f"{as_of} {config.SIGNAL_CUTOFF_KST}"
    found = []

    def walk(node, key=None, depth=0):
        if isinstance(node, dict):
            if "refused" in node:
                return
            is_document = "available_from" in node  # 검색 문서는 공개 시점(available_from)으로만 판단한다
            for k, v in node.items():
                if is_document and k in ("date", "text"):
                    continue
                walk(v, k, depth + 1)
        elif isinstance(node, list):
            for v in node:
                walk(v, key, depth + 1)
        elif key == "available_from" and isinstance(node, int):
            if pd.Timestamp(str(node)).strftime("%Y-%m-%d") > as_of:
                found.append(f"available_from={node}")
        elif isinstance(node, str):
            if key == "published_kst":
                if node[:16] >= cutoff:
                    found.append(f"published_kst={node}")
                return
            for d in DATE_RE.findall(node):
                same_day_ok = key in SAME_DAY_OK and depth <= 2  # 최상위 시점 키만 같은 날을 허용
                if d > as_of or (d == as_of and not same_day_ok and key is not None):
                    found.append(f"{key}={d}")

    walk(result)
    return found


def _dates_in(result) -> list[str]:
    """(하위 호환) 결과에서 날짜 문자열을 모두 모은다."""
    return DATE_RE.findall(json.dumps(result, ensure_ascii=False, default=str)) if not (isinstance(result, dict) and "refused" in result) else []


def auto_checks(row: pd.Series, result: dict) -> dict:
    """규칙 검사: 시점을 맞게 잡았나, 기대 도구를 불렀나, 기대 방향을 말했나, 시점 이후 자료가 들어왔나."""
    final = result["trace"][-1]
    expected_as_of = row["as_of"] if isinstance(row["as_of"], str) and row["as_of"] else str(tools.resolve_as_of(None).date())
    called = {s["tool"] for s in result["trace"] if s["step"] == "tool_call"}
    expected_tools = [t for t in str(row["expected_sources"]).split(";") if t and t != "nan"]
    direction = row["expected_direction"] if isinstance(row["expected_direction"], str) and row["expected_direction"] else None
    late = [f"{c['tool']}: {v}" for c in result["contexts"] for v in violations(c["result"], expected_as_of)]
    return {
        "as_of_ok": final["as_of"] == expected_as_of,
        "sources_ok": set(expected_tools) <= called,
        "direction_ok": None if direction is None else direction in result["answer"],
        "leak_ok": not late,
        "late_dates": late,
        "tools_called": sorted(called),
        "cost_usd": final["cost_usd"],
    }


def judge(row: pd.Series, result: dict, model) -> JudgeScore:
    """judge 모델로 답을 채점한다. contexts 는 길이를 잘라 넣는다."""
    contexts = json.dumps(result["contexts"], ensure_ascii=False, default=str)[:CONTEXT_CHARS]
    human = (f"## 질문\n{row['question']}\n\n## 질문 시점\n{result['trace'][-1]['as_of']} {config.SIGNAL_CUTOFF_KST} KST\n\n"
             f"## 기대 사항\n{row['note']}\n\n## 도구 결과(contexts, 일부)\n{contexts}\n\n## 답\n{result['answer']}")
    return model.with_structured_output(JudgeScore).invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(human)])


def write_report(rows: list[dict]) -> None:
    """평가 결과 요약과 질문별 표를 evaluation/reports/query_eval.md 에 쓴다."""
    df = pd.DataFrame(rows)
    rate = lambda col: f"{df[col].dropna().astype(bool).mean():.0%} ({int(df[col].dropna().astype(bool).sum())}/{df[col].notna().sum()})"
    same_model = config.JUDGE_MODEL == llm.scoring_model_id()
    lines = [
        "# /query 에이전트 평가", "",
        f"> 질문 {len(df)}개 · 에이전트 모델 {llm.scoring_model_id()} · judge {config.JUDGE_MODEL}"
        + (" · **같은 모델 평가(편향 가능)**" if same_model else ""), "",
        "## 요약", "",
        "| 항목 | 결과 |", "|---|---|",
        f"| 시점(as_of)을 맞게 잡음 | {rate('as_of_ok')} |",
        f"| 기대 도구를 부름 | {rate('sources_ok')} |",
        f"| 기대 방향을 답에 씀 | {rate('direction_ok')} |",
        f"| 도구 결과에 시점 이후 자료 없음 (자동 검사) | {rate('leak_ok')} |",
        f"| judge: 시점 위반 없음 | {(~df['temporal_violation']).mean():.0%} |",
        f"| judge: 기대 사항 충족 | {df['meets_note'].mean():.0%} |",
        f"| judge: 근거 충실도 평균 (1~5) | {df['faithfulness'].mean():.2f} |",
        f"| judge: 질문 적합도 평균 (1~5) | {df['relevance'].mean():.2f} |",
        f"| 비용 (에이전트) | ${df['cost_usd'].sum():.3f} |", "",
        "## 질문별", "",
        "| # | 질문 | 시점 | 도구 | 방향 | 유출 검사 | 충실 | 적합 | 기대 | 시점 위반 | judge 의견 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    mark = lambda v: "-" if v is None or (isinstance(v, float) and pd.isna(v)) else ("O" if v else "X")
    for i, r in enumerate(rows, start=1):
        lines.append(f"| {i} | {r['question']} | {mark(r['as_of_ok'])} | {mark(r['sources_ok'])} | {mark(r['direction_ok'])} | "
                     f"{mark(r['leak_ok'])} | {r['faithfulness']} | {r['relevance']} | {mark(r['meets_note'])} | "
                     f"{'X' if r['temporal_violation'] else 'O'} | {r['comment']} |")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """질문마다 에이전트로 답하고 자동 검사와 judge 채점을 한 뒤 보고서를 쓴다."""
    queries = pd.read_csv(QUERIES, dtype=str).fillna("")
    judge_model = llm.get_judge_llm()
    rows, details = [], []
    for _, row in queries.iterrows():
        try:
            result = agent.answer(row["question"])
        except Exception as e:  # 한 질문의 실패가 평가 전체를 멈추지 않게 한다
            result = {"answer": f"오류: {type(e).__name__}: {e}", "contexts": [],
                      "trace": [{"step": "answer", "as_of": None, "cost_usd": 0.0, "error": str(e)}]}
        checks = auto_checks(row, result)
        score = judge(row, result, judge_model)
        rows.append({"question": row["question"], **checks, **score.model_dump()})
        details.append({"question": row["question"], "result": result, "checks": checks, "judge": score.model_dump()})
        print(f"{row['question'][:30]:<30} 시점 {checks['as_of_ok']} 도구 {checks['sources_ok']} 방향 {checks['direction_ok']} "
              f"유출검사 {checks['leak_ok']} | judge 충실 {score.faithfulness} 적합 {score.relevance} 기대 {score.meets_note} "
              f"시점위반 {score.temporal_violation}", flush=True)
    DETAILS.parent.mkdir(parents=True, exist_ok=True)
    DETAILS.write_text(json.dumps(details, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(rows)
    print(REPORT, f"누적 비용 ${llm.total_cost():.3f}")


if __name__ == "__main__":
    main()
