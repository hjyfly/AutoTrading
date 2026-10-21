import pandas as pd

from evaluation import run_eval as ev


def make_result(as_of, tools_called, answer, contexts):
    trace = [{"step": "tool_call", "tool": t, "args": {}} for t in tools_called]
    trace.append({"step": "answer", "as_of": as_of, "cost_usd": 0.01})
    return {"answer": answer, "contexts": contexts, "trace": trace}


ROW = pd.Series({"question": "q", "as_of": "2026-03-04", "expected_direction": "보합",
                 "expected_sources": "get_signal;get_news_judgement", "note": ""})


def test_모두_맞으면_통과():
    result = make_result("2026-03-04", ["get_signal", "get_news_judgement"], "신호는 보합",
                         [{"tool": "get_signal", "args": {}, "result": {"date": "2026-03-04"}}])
    c = ev.auto_checks(ROW, result)
    assert c["as_of_ok"] and c["sources_ok"] and c["direction_ok"] and c["leak_ok"]


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


def test_방향_기대가_없으면_검사하지_않는다():
    row = ROW.copy()
    row["expected_direction"] = ""
    assert ev.auto_checks(row, make_result("2026-03-04", [], "x", []))["direction_ok"] is None


def test_상한_안내_문구의_날짜는_위반이_아니다():
    result = {"as_of": "2026-03-04", "as_of_note": "요청 시점 2026-03-04 이후(2026-03-13)는 조회하지 않아 2026-03-04 기준으로 답함",
              "rows": [{"date": "2026-03-03"}]}
    assert ev.violations(result, "2026-03-04") == []
