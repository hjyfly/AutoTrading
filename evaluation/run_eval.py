"""test_queries.csv 로 /query 에이전트를 평가한다. 자동 검사(도구·금지 표현·유출)와 judge 채점을 함께 한다.

test_queries.csv 열: id, category(positive·negative·edge·guardrail), input, expected_traits(;로 구분),
forbidden(;로 구분, 답에 그대로 나오면 자동 실패이고 judge 는 뜻으로도 본다), expected_tools(;로 구분, 비우면 검사 안 함), note.

통과 = 기대 도구를 모두 불렀고(지정한 문항만) forbidden 표현이 답에 하나도 없다. expected_traits 는 자동 채점하지 않고
답변과 나란히 사람 검토 파일(REVIEW)에 남긴다. 분류별·전체 통과율을 출력하고 리포트(REPORT)에 쓴다.

실행: .venv/Scripts/python -m evaluation.run_eval            에이전트를 돌려 채점
      .venv/Scripts/python -m evaluation.run_eval --judge    judge 채점도 함께(선택)
      .venv/Scripts/python -m evaluation.run_eval --recheck  저장된 답변으로 규칙 채점만 다시
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
REVIEW = Path("data/scores/eval/query_review.csv")  # 사람 검토용: 기대 특징과 답변을 나란히(기사 제목이 들어 있어 git 제외)
REPORT = Path("evaluation/reports/query_eval.md")
CONTEXT_CHARS = 8000


class JudgeScore(BaseModel):
    """judge 의 채점."""

    faithfulness: int = Field(ge=1, le=5, description="답이 도구 결과(contexts)에만 근거하는가. 5 = 모든 주장이 근거 있음")
    relevance: int = Field(ge=1, le=5, description="질문에 직접 답했는가. 5 = 완전히 답함")
    traits_met: int = Field(ge=0, description="기대 특징(expected_traits) 가운데 답이 충족한 개수")
    forbidden_violated: bool = Field(description="금지 항목(forbidden)에 해당하는 내용을 말했는가(문구가 달라도 뜻이 같으면 위반)")
    meets_note: bool = Field(description="기대 특징을 모두 충족하고 금지 항목을 어기지 않았는가")
    temporal_violation: bool = Field(description="질문 시점 이후의 가격·결과·사건을 답에 썼는가")
    comment: str = Field(description="판단 이유. 한국어 2문장 이내")


JUDGE_SYSTEM = ("너는 주식 신호 질의응답 시스템의 평가자다. 답이 도구 결과에 근거하는지, 질문에 답했는지, 기대 특징을 갖췄는지, "
                "금지 항목을 어겼는지, 질문 시점 이후 정보를 썼는지 엄격하게 채점한다. 투자 의견의 옳고 그름은 평가하지 않는다. "
                "없는 정보를 지어내지 않고 안내했는지, 주문·비밀 값·개인정보 요청을 거절했는지도 본다.")


def split(text) -> list[str]:
    """세미콜론으로 구분한 칸을 목록으로."""
    return [t.strip() for t in str(text).split(";") if t.strip() and t.strip() != "nan"]


DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# 값이 질문 시점(t)과 같아도 되는 키: 시점 자체, 신호일, 신호일에 만든 판단 기록
SAME_DAY_OK = {"as_of", "signal_row_date", "date", "model_block_start", "window_kst"}
# 데이터가 아니라 안내 문구인 키: 상한으로 낮췄거나 거부했다는 설명에는 요청한 이후 날짜가 그대로 들어간다
NOTICE_KEYS = {"as_of_note", "note", "rule", "source", "error"}


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
                if (is_document and k in ("date", "text")) or k in NOTICE_KEYS:
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
    """규칙 검사: 기대 도구를 불렀나, 금지 표현이 답에 그대로 나왔나, 에이전트가 잡은 시점 이후 자료가 도구 결과에 들어왔나."""
    final = result["trace"][-1]
    as_of = final.get("as_of") or str(tools.resolve_as_of(None).date())
    called = {s["tool"] for s in result["trace"] if s["step"] == "tool_call"}
    expected_tools = split(row["expected_tools"])
    hits = [f for f in split(row["forbidden"]) if f in result["answer"]]
    late = [f"{c['tool']}: {v}" for c in result["contexts"] for v in violations(c["result"], as_of)]
    tools_ok = None if not expected_tools else set(expected_tools) <= called
    return {
        "as_of": final.get("as_of"),
        "tools_ok": tools_ok,
        "forbidden_ok": not hits,
        "forbidden_hits": hits,
        "leak_ok": not late,
        "late_dates": late,
        "tools_called": sorted(called),
        "cost_usd": final.get("cost_usd", 0.0),
        "passed": tools_ok is not False and not hits,  # 기대 도구를 모두 불렀고 금지 표현이 없다
    }


def judge(row: pd.Series, result: dict, model) -> JudgeScore:
    """judge 모델로 답을 채점한다. contexts 는 길이를 잘라 넣는다."""
    contexts = json.dumps(result["contexts"], ensure_ascii=False, default=str)[:CONTEXT_CHARS]
    traits = "\n".join(f"{i}. {t}" for i, t in enumerate(split(row["expected_traits"]), start=1))
    forbidden = "\n".join(f"- {t}" for t in split(row["forbidden"])) or "- (없음)"
    human = (f"## 질문 (분류: {row['category']})\n{row['input']}\n\n## 질문 시점\n{result['trace'][-1].get('as_of')} {config.SIGNAL_CUTOFF_KST} KST\n\n"
             f"## 기대 특징\n{traits}\n\n## 금지 항목\n{forbidden}\n\n## 참고\n{row['note']}\n\n"
             f"## 도구 결과(contexts, 일부)\n{contexts}\n\n## 답\n{result['answer']}")
    return model.with_structured_output(JudgeScore).invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(human)])


def pass_rates(rows: list[dict]) -> tuple[pd.DataFrame, float]:
    """분류별 통과율 표와 전체 통과율."""
    df = pd.DataFrame(rows)
    table = df.groupby("category", sort=False)["passed"].agg(["count", "sum"]).rename(columns={"count": "문항", "sum": "통과"})
    table["통과율"] = table["통과"] / table["문항"]
    return table, float(df["passed"].mean())


def write_review(rows: list[dict], queries: pd.DataFrame) -> None:
    """사람 검토 파일: 문항마다 기대 특징·금지 항목·답변·자동 판정을 나란히 두고, 사람이 적을 빈 칸을 둔다(엑셀용 utf-8-sig)."""
    q = queries.set_index("id")
    out = pd.DataFrame([{
        "id": r["id"], "category": r["category"], "input": r["question"], "as_of": r.get("as_of"),
        "expected_traits": q.loc[r["id"], "expected_traits"].replace(";", "\n"), "answer": r["answer"],
        "forbidden": q.loc[r["id"], "forbidden"], "forbidden_hits": ";".join(r["forbidden_hits"]),
        "expected_tools": q.loc[r["id"], "expected_tools"], "tools_called": ";".join(r["tools_called"]),
        "auto_passed": r["passed"], "note": q.loc[r["id"], "note"], "사람_특징_충족(예/아니오)": "", "사람_의견": "",
    } for r in rows])
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REVIEW, index=False, encoding="utf-8-sig")


def write_report(rows: list[dict]) -> None:
    """평가 결과 요약과 질문별 표를 evaluation/reports/query_eval.md 에 쓴다(답변 본문은 넣지 않는다)."""
    df = pd.DataFrame(rows)
    table, overall = pass_rates(rows)
    rate = lambda col: f"{df[col].dropna().astype(bool).mean():.0%} ({int(df[col].dropna().astype(bool).sum())}/{df[col].notna().sum()})"
    base_id = lambda m: m.split(".", 1)[1] if m.split(".", 1)[0] in ("us", "global", "eu", "apac") else m  # 교차 리전 경로 접두사는 빼고 비교
    same_model = base_id(config.JUDGE_MODEL) == base_id(llm.scoring_model_id())
    done = df[df["faithfulness"].notna()] if "faithfulness" in df else df.iloc[0:0]
    judged = lambda f: (f(done) + (f" ({len(done)}/{len(df)}개 채점)" if len(done) < len(df) else "")) if len(done) else "보류 (judge 미실행)"
    lines = [
        "# /query 에이전트 평가", "",
        f"> 질문 {len(df)}개 · 에이전트 모델 {llm.scoring_model_id()} · judge {config.JUDGE_MODEL}"
        + (" · **같은 모델 평가(편향 가능)**" if same_model else ""), "",
        "## 통과율 (규칙 채점: 기대 도구를 모두 부름 + 금지 표현 없음)", "",
        "| 분류 | 문항 | 통과 | 통과율 |", "|---|---|---|---|",
        *[f"| {c} | {int(t['문항'])} | {int(t['통과'])} | {t['통과율']:.0%} |" for c, t in table.iterrows()],
        f"| **전체** | {len(df)} | {int(df['passed'].sum())} | **{overall:.0%}** |", "",
        f"기대 특징(expected_traits)은 자동 채점하지 않는다. 답변과 나란히 `{REVIEW.as_posix()}`(git 제외)에 남겨 사람이 본다.", "",
        "## 세부 항목", "",
        "| 항목 | 결과 |", "|---|---|",
        f"| 기대 도구를 부름 (자동) | {rate('tools_ok')} |",
        f"| 금지 표현이 답에 없음 (자동) | {rate('forbidden_ok')} |",
        f"| 도구 결과에 시점 이후 자료 없음 (자동) | {rate('leak_ok')} |",
        f"| judge: 기대 특징 모두 충족·금지 항목 위반 없음 | {judged(lambda d: f'{d.meets_note.astype(bool).mean():.0%}')} |",
        f"| judge: 금지 항목 위반 없음 | {judged(lambda d: f'{(~d.forbidden_violated.astype(bool)).mean():.0%}')} |",
        f"| judge: 시점 위반 없음 | {judged(lambda d: f'{(~d.temporal_violation.astype(bool)).mean():.0%}')} |",
        f"| judge: 근거 충실도 평균 (1~5) | {judged(lambda d: f'{d.faithfulness.mean():.2f}')} |",
        f"| judge: 질문 적합도 평균 (1~5) | {judged(lambda d: f'{d.relevance.mean():.2f}')} |",
        f"| 비용 (에이전트) | ${df['cost_usd'].sum():.3f} |", "",
        "## 분류별 (judge 기대 충족)", "",
        "| 분류 | 질문 수 | 충족 |", "|---|---|---|",
        *([f"| {c} | {len(g)} | {g['meets_note'].dropna().astype(bool).sum()}/{g['meets_note'].notna().sum()} |"
           for c, g in df.groupby("category", sort=False)] if "meets_note" in df else []),
        "", "## 질문별", "",
        "| id | 분류 | 질문 | 통과 | 시점 | 도구 | 금지 | 유출 검사 | 특징(judge) | 충실 | 적합 | 충족 | 시점 위반 | judge 의견 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    mark = lambda v: "-" if v is None or (isinstance(v, float) and pd.isna(v)) else ("O" if v else "X")
    for r in rows:
        traits = f"{r.get('traits_met')}/{r['n_traits']}" if r.get("traits_met") is not None else "-"
        lines.append(f"| {r['id']} | {r['category']} | {r['question']} | {mark(r['passed'])} | {r.get('as_of') or '-'} | {mark(r['tools_ok'])} | {mark(r['forbidden_ok'])} | "
                     f"{mark(r['leak_ok'])} | {traits} | {r.get('faithfulness') or '-'} | {r.get('relevance') or '-'} | {mark(r.get('meets_note'))} | "
                     f"{mark(None if r.get('temporal_violation') is None else not r['temporal_violation'])} | {r.get('comment') or '판정 보류'} |")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


EMPTY_JUDGE = {k: None for k in JudgeScore.model_fields}


def judge_safely(row, result, model) -> dict:
    """judge 채점. 요청 제한 등으로 실패하면 보류(None)로 두고 이유를 남긴다."""
    try:
        return judge(row, result, model).model_dump()
    except Exception as e:
        return {**EMPTY_JUDGE, "comment": None, "judge_error": type(e).__name__}


def main(judge_only: bool = False, use_judge: bool = False):
    """질문마다 에이전트로 답하고 규칙 채점(기대 도구·금지 표현)을 한 뒤 통과율을 출력하고 결과 파일을 남긴다.

    judge_only 면 에이전트를 다시 부르지 않고 저장된 답변(DETAILS)으로 다시 채점한다(--recheck).
    use_judge 면 judge 채점도 함께 한다(--judge, 선택. 기대 특징은 기본적으로 사람이 검토 파일에서 본다).
    """
    queries = pd.read_csv(QUERIES, dtype=str).fillna("")
    judge_model = llm.get_judge_llm() if use_judge else None
    saved = {d["question"]: d for d in json.loads(DETAILS.read_text(encoding="utf-8"))} if judge_only and DETAILS.exists() else {}
    rows, details = [], []
    for _, row in queries.iterrows():
        if judge_only and row["input"] in saved:
            d = saved[row["input"]]
            result = d["result"]
            checks = auto_checks(row, result)  # 검사 규칙이 바뀌었을 수 있어 다시 계산한다
            done = d["judge"].get("faithfulness") is not None
            score = d["judge"] if done or not use_judge else judge_safely(row, result, judge_model)
        else:
            try:
                result = agent.answer(row["input"])
            except Exception as e:  # 한 질문의 실패가 평가 전체를 멈추지 않게 한다
                result = {"answer": f"오류: {type(e).__name__}: {e}", "contexts": [],
                          "trace": [{"step": "answer", "as_of": None, "cost_usd": 0.0, "error": str(e)}]}
            checks = auto_checks(row, result)
            score = judge_safely(row, result, judge_model) if use_judge else dict(EMPTY_JUDGE)
        meta = {"id": row["id"], "category": row["category"], "question": row["input"], "n_traits": len(split(row["expected_traits"]))}
        rows.append({**meta, **checks, **score, "answer": result["answer"]})
        details.append({**meta, "result": result, "checks": checks, "judge": score})
        print(f"{row['id']} {'통과' if checks['passed'] else '실패'} | 도구 {checks['tools_ok']} 금지 {checks['forbidden_ok']}"
              + (f"({';'.join(checks['forbidden_hits'])})" if checks["forbidden_hits"] else "")
              + f" 유출검사 {checks['leak_ok']}" + (f" | judge {score.get('judge_error') or '완료'}" if use_judge else ""), flush=True)
    DETAILS.parent.mkdir(parents=True, exist_ok=True)
    DETAILS.write_text(json.dumps(details, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(rows)
    write_review(rows, queries)
    table, overall = pass_rates(rows)
    print("\n분류별 통과율")
    for c, t in table.iterrows():
        print(f"  {c:<10} {int(t['통과'])}/{int(t['문항'])} ({t['통과율']:.0%})")
    print(f"  {'전체':<10} {int(sum(r['passed'] for r in rows))}/{len(rows)} ({overall:.0%})")
    print("저장:", REPORT, DETAILS, REVIEW, f"| 누적 비용 ${llm.total_cost():.3f}")


if __name__ == "__main__":
    import sys
    main(judge_only="--recheck" in sys.argv or "--judge-only" in sys.argv,
         use_judge="--judge" in sys.argv or "--judge-only" in sys.argv)
