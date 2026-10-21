"""Streamlit 화면. 실행: .venv\\Scripts\\streamlit run src/ui.py

왼쪽은 질의응답(에이전트), 오른쪽은 선택한 날짜 기준의 신호·차트·지표·판단 기록·뉴스다.
오른쪽 데이터는 모두 도구 함수(src/tools.py)로 읽으므로, 선택한 날짜 시점에 알 수 있던 정보만 보인다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # streamlit run 으로 실행해도 src 를 찾게 한다

import altair as alt
import pandas as pd
import streamlit as st

from src import agent, config, llm, tools
from src.pipeline import live_signal

st.set_page_config(page_title=f"삼성전자 {config.LABEL_HORIZON_DAYS}거래일 신호", layout="wide")


@st.cache_data(ttl=600)
def load_signals() -> pd.DataFrame:
    """검증 구간 신호(뉴스 보정 반영)와 실시간으로 만든 신호를 합친다."""
    numeric = pd.read_parquet(config.SIGNALS_NUMERIC_PATH)[["signal", "confidence"]]
    news_path = Path(config.PROCESSED_DIR) / "signals_news.parquet"
    if news_path.exists():
        news = pd.read_parquet(news_path)
        numeric.loc[news.index, "signal"] = news["signal"]
    live_path = Path(config.PROCESSED_DIR) / "signals_live.parquet"
    if live_path.exists():
        live = pd.read_parquet(live_path)[["signal", "confidence"]]
        numeric = pd.concat([numeric[~numeric.index.isin(live.index)], live]).sort_index()
    return numeric


@st.cache_data(ttl=600)
def load_prices() -> pd.Series:
    """저장된 삼성전자 종가."""
    return pd.read_parquet(Path(config.RAW_DIR) / "prices" / "samsung_ohlcv.parquet")["Close"].astype(float)


@st.cache_data(ttl=600)
def load_equity() -> pd.DataFrame:
    """검증 구간 모의 매매 자산 곡선(모델 vs 들고 있기)."""
    return pd.read_parquet(Path(config.PROCESSED_DIR) / "equity_numeric.parquet")


def price_chart(as_of: pd.Timestamp, days: int) -> alt.Chart:
    """선택한 날짜 전날까지의 종가와 그 기간의 신호 표시."""
    close = load_prices()
    close = close[close.index < as_of].tail(days)
    signals = load_signals().reindex(close.index).dropna()
    base = pd.DataFrame({"date": close.index, "close": close.values})
    marks = pd.DataFrame({"date": signals.index, "close": close.reindex(signals.index).values, "signal": signals["signal"].values})
    line = alt.Chart(base).mark_line().encode(x=alt.X("date:T", title=None), y=alt.Y("close:Q", title="종가", scale=alt.Scale(zero=False)))
    dots = alt.Chart(marks).mark_point(filled=True, size=60).encode(
        x="date:T", y="close:Q",
        color=alt.Color("signal:N", title="신호", scale=alt.Scale(domain=["상승", "보합", "하락"], range=["#d62728", "#7f7f7f", "#1f77b4"])),
        tooltip=["date:T", "signal:N", "close:Q"])
    return (line + dots).properties(height=320)


def equity_chart(as_of: pd.Timestamp) -> alt.Chart | None:
    """검증 구간 자산 곡선. 선택한 날짜 이후는 그리지 않는다(사후 정보)."""
    eq = load_equity()
    eq = eq[eq.index < as_of]
    if eq.empty:
        return None
    long = eq.reset_index(names="date").melt("date", var_name="전략", value_name="자산")
    long["전략"] = long["전략"].map({"model": "숫자 모델", "hold": "그냥 들고 있기"})
    return alt.Chart(long).mark_line().encode(x=alt.X("date:T", title=None), y=alt.Y("자산:Q", title="자산(시작 1.0)"),
                                              color="전략:N").properties(height=260)


# ---- 사이드바: 기준 날짜, 오늘 신호 만들기, 비용 ----
with st.sidebar:
    st.header("기준 날짜")
    picked = st.date_input(f"이 날짜 {config.SIGNAL_CUTOFF_KST} 시점으로 봄", value=tools.resolve_as_of(None).date())
    as_of = pd.Timestamp(picked)
    use_date_in_chat = st.checkbox("질문에 날짜가 없으면 이 날짜로 묻기", value=True)
    st.divider()
    st.header("신호 만들기")
    st.caption(f"저장된 신호는 {config.DATA_END} 까지. 그 뒤 날짜는 실시간 데이터로 만든다.")
    if st.button("숫자 모델 신호 만들기"):
        with st.spinner("실시간 데이터로 학습·예측 중"):
            try:
                live_signal.make_signal(str(as_of.date()))
                st.cache_data.clear()
                st.success("완료")
            except Exception as e:
                st.error(f"실패: {e}")
    if st.button("뉴스 보정까지 (약 $0.15, 2~3분)"):
        with st.spinner("뉴스 수집·채점 중"):
            try:
                out = live_signal.make_signal(str(as_of.date()), with_news=True)
                st.cache_data.clear()
                st.success(out["news_status"])
            except Exception as e:
                st.error(f"실패: {e}")
    st.divider()
    spent = llm.total_cost()
    st.metric("LLM 누적 비용", f"${spent:.2f}", help=f"한도 ${config.COST_LIMIT_POC_USD:.0f}")
    st.progress(min(spent / config.COST_LIMIT_POC_USD, 1.0))
    st.caption("투자 권유가 아니라 모델 판단의 설명이다.")

st.title(f"삼성전자 {config.LABEL_HORIZON_DAYS}거래일 방향 신호 · {as_of.date()} {config.SIGNAL_CUTOFF_KST} 기준")
chat_col, data_col = st.columns([5, 7])

# ---- 오른쪽: 데이터 ----
with data_col:
    sig = tools.signal_for(str(as_of.date()))
    c1, c2, c3 = st.columns(3)
    final = sig.get("with_news", {}).get("signal") or sig.get("numeric", {}).get("signal")
    c1.metric("최종 신호", final or "없음")
    if "numeric" in sig:
        c2.metric("숫자 모델", sig["numeric"]["signal"], f"확신도 {sig['numeric']['confidence']:.0%}", delta_color="off")
    if "with_news" in sig:
        c3.metric("뉴스 합의", f"{sig['with_news']['news_consensus']:+d}", "보정됨" if sig["with_news"]["adjusted"] else "보정 없음", delta_color="off")
    if "candidate" in sig:
        st.caption(f"비교용 후보 모델(A+B+D 지표 추가): **{sig['candidate']['signal']}** (확신도 {sig['candidate']['confidence']:.0%}) "
                   "— 최종 신호가 아니며 앞으로의 성과로 기존 모델과 비교 중")
    if "note" in sig:
        st.info(sig["note"])

    tab_chart, tab_ind, tab_judge, tab_news, tab_perf = st.tabs(["가격·신호", "지표", "판단 기록", "뉴스", "성과"])
    with tab_chart:
        days = st.slider("기간(거래일)", 20, 250, 120)
        st.altair_chart(price_chart(as_of, days), width="stretch")
    with tab_ind:
        try:
            ind = tools.indicators(str(as_of.date()))
            left, right = st.columns(2)
            left.subheader("차트·수급 (전날 종가 기준)")
            left.dataframe(pd.Series(ind["chart_and_flow"], name="값"), width="stretch")
            right.subheader("해외·거시 (현지 날짜 < 기준일)")
            right.dataframe(pd.Series(ind["foreign"], name="값"), width="stretch")
            with st.expander("데이터 출처"):
                st.json(ind["data_sources"])
        except Exception as e:
            st.warning(f"지표를 계산하지 못했다: {e}")
    with tab_judge:
        j = tools.news_judgement(str(as_of.date()))
        if "final" not in j:
            st.info(j.get("note", "판단 기록 없음"))
        else:
            st.subheader(f"합의 {j['final']['score']:+d} (확신도 {j['final']['confidence']:.2f})")
            st.write(j["final"]["rationale"])
            st.caption(f"남은 이견: {j['final']['disagreement']}")
            st.markdown("**핵심 기사**\n" + "\n".join(f"- {t}" for t in j["final"]["key_articles"] if t))
            for name, o in j["personas"].items():
                with st.expander(f"{name}: {o['score']:+d} (확신도 {o['confidence']:.2f})"):
                    st.write(o["rationale"])
                    st.caption(f"토론 뒤: {o['change_reason']}")
    with tab_news:
        n = tools.news_for_signal(str(as_of.date()), limit=30)
        st.caption(n["source"])
        if n["articles"]:
            st.dataframe(pd.DataFrame(n["articles"])[["published_kst", "source", "title"]], width="stretch", hide_index=True)
    with tab_perf:
        chart = equity_chart(as_of)
        if chart is None:
            st.info(f"검증 구간({config.VALID_START}~) 이전 날짜다.")
        else:
            st.caption("선택한 날짜 전까지의 모의 매매 자산 곡선. 전체 성과 판정은 evaluation/reports/day3_backtest.md")
            st.altair_chart(chart, width="stretch")

# ---- 왼쪽: 질의응답 ----
with chat_col:
    st.subheader("질의응답")
    if "history" not in st.session_state:
        st.session_state.history = []
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("result"):
                with st.expander(f"근거 {len(turn['result']['contexts'])}건 · 비용 ${turn['result']['trace'][-1].get('cost_usd', 0):.3f}"):
                    st.json({"trace": turn["result"]["trace"], "contexts": turn["result"]["contexts"]}, expanded=False)
    question = st.chat_input("예: 이날 신호와 이유는? / 외국인 수급은? / 모델 성과는?")
    if question:
        asked = question
        if use_date_in_chat and agent.extract_as_of(question) is None:
            asked = f"{as_of.date()} 기준: {question}"
        st.session_state.history.append({"role": "user", "content": asked})
        with st.spinner("답을 만드는 중 (20~40초)"):
            result = agent.answer(asked)
        st.session_state.history.append({"role": "assistant", "content": result["answer"], "result": result})
        st.rerun()
