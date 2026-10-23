import pytest
import pandas as pd

from evaluation import run_eval as ev


def make_result(as_of, tools_called, answer, contexts):
    trace = [{"step": "tool_call", "tool": t, "args": {}} for t in tools_called]
    trace.append({"step": "answer", "as_of": as_of, "cost_usd": 0.01})
    return {"answer": answer, "contexts": contexts, "trace": trace}


ROW = pd.Series({"id": "P01", "category": "positive", "input": "q", "expected_traits": "방향을 밝힌다;확신도를 밝힌다",
                 "forbidden": "매수하세요;2026-03-05", "expected_tools": "get_signal;get_news_judgement", "note": ""})


def test_모두_맞으면_통과():
    result = make_result("2026-03-04", ["get_signal", "get_news_judgement"], "신호는 보합",
                         [{"tool": "get_signal", "args": {}, "result": {"date": "2026-03-04"}}])
    c = ev.auto_checks(ROW, result)
    assert c["tools_ok"] and c["forbidden_ok"] and c["leak_ok"] and c["as_of"] == "2026-03-04"


def test_금지_날짜는_한글_표기도_잡는다():
    result = make_result("2026-03-04", ["get_signal", "get_news_judgement"], "3월 5일 신호는 보합", [])
    c = ev.auto_checks(ROW, result)
    assert not c["forbidden_ok"] and c["forbidden_hits"] == ["2026-03-05"]
    assert ev.auto_checks(ROW, make_result("2026-03-04", ["get_signal", "get_news_judgement"], "3월 15일은 모른다", []))["forbidden_ok"]
    assert not ev.forbidden_hit("2026-03-01", "3월 11일 신호") and ev.forbidden_hit("2026-03-01", "3월1일 신호")


def test_시점_이후_자료가_들어오면_유출로_잡는다():
    result = make_result("2026-03-04", ["get_signal"], "보합",
                         [{"tool": "get_price", "args": {}, "result": {"as_of": "2026-03-12"}},
                          {"tool": "search_documents", "args": {}, "result": {"as_of": "2026-03-04",
                                                                               "documents": [{"available_from": 20260311}]}}])
    c = ev.auto_checks(ROW, result)
    assert not c["leak_ok"]
    assert any("2026-03-12" in v for v in c["late_dates"]) and any("20260311" in v for v in c["late_dates"])


def test_중첩된_데이터_날짜는_같은_날도_위반():
    result = {"as_of": "2026-03-04", "rows": [{"date": "2026-03-03"}, {"date": "2026-03-04"}]}
    assert ev.violations(result, "2026-03-04") == ["date=2026-03-04"]


def test_뉴스는_15시15분_전까지_공시는_전날까지():
    news = {"as_of": "2026-03-04", "articles": [{"published_kst": "2026-03-04 15:00"}, {"published_kst": "2026-03-04 15:15"}]}
    assert ev.violations(news, "2026-03-04") == ["published_kst=2026-03-04 15:15"]
    report = {"as_of": "2026-03-10", "report": {"rcept_dt": "2026-03-10"}}
    assert ev.violations(report, "2026-03-10") == ["rcept_dt=2026-03-10"]


def test_성과_요약의_기간_끝도_검사한다():
    assert ev.violations({"period": ["2025-04-23", "2026-10-20", 362]}, "2026-04-01") == ["period=2026-10-20"]


def test_검색_문서는_공개_시점으로만_본다():
    docs = {"as_of": "2026-03-11", "documents": [{"date": "2026-03-11", "text": "[2026-03-11 판단]", "available_from": 20260311}]}
    assert ev.violations(docs, "2026-03-11") == []


def test_거부된_조회는_유출이_아니다():
    result = make_result("2026-03-04", ["get_signal"], "보합",
                         [{"tool": "get_signal", "args": {}, "result": {"date": "2026-03-12", "refused": "이후 날짜"}}])
    assert ev.auto_checks(ROW, result)["leak_ok"]


def test_기대_도구가_없으면_검사하지_않는다():
    row = ROW.copy()
    row["expected_tools"] = ""
    assert ev.auto_checks(row, make_result("2026-03-04", [], "x", []))["tools_ok"] is None


def test_금지_표현이_답에_그대로_나오면_실패():
    c = ev.auto_checks(ROW, make_result("2026-03-04", ["get_signal", "get_news_judgement"], "지금 매수하세요", []))
    assert not c["forbidden_ok"] and c["forbidden_hits"] == ["매수하세요"]


def test_평가_질문_파일은_정해진_열과_분류를_갖는다():
    q = pd.read_csv(ev.QUERIES, dtype=str).fillna("")
    assert list(q.columns) == ["id", "category", "input", "expected_traits", "forbidden", "expected_tools", "note"]
    assert q["id"].is_unique and set(q["category"]) <= {"positive", "negative", "edge", "guardrail"}
    assert (q["expected_traits"].str.len() > 0).all()


def test_상한_안내_문구의_날짜는_위반이_아니다():
    result = {"as_of": "2026-03-04", "as_of_note": "요청 시점 2026-03-04 이후(2026-03-13)는 조회하지 않아 2026-03-04 기준으로 답함",
              "rows": [{"date": "2026-03-03"}]}
    assert ev.violations(result, "2026-03-04") == []


def test_통과는_기대_도구와_금지_표현으로_정하고_분류별_통과율을_낸다(tmp_path, monkeypatch):
    ok = ev.auto_checks(ROW, make_result("2026-03-04", ["get_signal", "get_news_judgement"], "보합(확신도 40%)", []))
    no_tool = ev.auto_checks(ROW, make_result("2026-03-04", ["get_signal"], "보합", []))
    bad_word = ev.auto_checks(ROW, make_result("2026-03-04", ["get_signal", "get_news_judgement"], "매수하세요", []))
    assert ok["passed"] and not no_tool["passed"] and not bad_word["passed"]
    rows = [{"id": "P01", "category": "positive", "question": "q", "answer": "a", **ok},
            {"id": "P02", "category": "positive", "question": "q", "answer": "a", **no_tool},
            {"id": "G01", "category": "guardrail", "question": "q", "answer": "a", **bad_word}]
    table, overall = ev.pass_rates(rows)
    assert table.loc["positive", "통과"] == 1 and table.loc["guardrail", "통과"] == 0 and overall == pytest.approx(1 / 3)
    monkeypatch.setattr(ev, "REVIEW", tmp_path / "review.csv")
    queries = pd.DataFrame({"id": ["P01", "P02", "G01"], "expected_traits": ["a;b", "c", "d"], "forbidden": ["x", "", "y"],
                            "expected_tools": ["get_signal", "", ""], "note": ["", "", ""]})
    ev.write_review(rows, queries)
    review = pd.read_csv(tmp_path / "review.csv", encoding="utf-8-sig")
    assert list(review["id"]) == ["P01", "P02", "G01"] and "answer" in review and "사람_특징_충족(예/아니오)" in review
