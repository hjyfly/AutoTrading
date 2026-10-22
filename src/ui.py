"""Streamlit 화면. 실행: .venv\\Scripts\\streamlit run src/ui.py

왼쪽은 질의응답(에이전트), 오른쪽은 선택한 날짜 기준의 신호·차트·지표·판단 기록·뉴스다.
오른쪽 데이터는 모두 도구 함수(src/tools.py)로 읽으므로, 선택한 날짜 시점에 알 수 있던 정보만 보인다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # streamlit run 으로 실행해도 src 를 찾게 한다

import json

import altair as alt
import pandas as pd
import streamlit as st

from src import agent, config, llm, tools
from src.pipeline import experiment, live_signal, news_signal, scoreboard

st.set_page_config(page_title=f"삼성전자 {config.LABEL_HORIZON_DAYS}거래일 신호", layout="wide")


@st.cache_data(ttl=600)
def load_signals() -> pd.DataFrame:
    """검증 구간 신호(뉴스 보정 반영)와 실시간으로 만든 신호를 합친다."""
    numeric = pd.read_parquet(config.SIGNALS_NUMERIC_PATH)[["signal", "confidence"]]
    news = news_signal.load_matching()  # 지금 숫자 모델을 보정한 날만(지표 묶음을 바꾼 뒤 다시 만들기 전 기록은 뺀다)
    if len(news):
        numeric.loc[news.index, "signal"] = news["signal"]
    live_path = Path(config.PROCESSED_DIR) / "signals_live.parquet"
    if live_path.exists():
        live = pd.read_parquet(live_path)
        if "model" in live.columns:  # 차트에는 최종 신호(기존 모델)만 쓴다
            live = live[live["model"] == "base"]
        live = live[["signal", "confidence"]]
        numeric = pd.concat([numeric[~numeric.index.isin(live.index)], live]).sort_index()
    return numeric


@st.cache_data(ttl=600)
def load_prices() -> pd.Series:
    """저장된 삼성전자 종가."""
    return pd.read_parquet(Path(config.RAW_DIR) / "prices" / "samsung_ohlcv.parquet")["Close"].astype(float)


@st.cache_resource
def experiment_data() -> experiment.Data:
    """조건 비교 실험용 원본 표와 학습 결과 보관소(같은 학습 조건은 한 번만 학습한다)."""
    return experiment.Data()


def experiment_tab():
    """모니터링 리포트 보기와 조건 조회를 하위 탭으로 나눠 보여 준다. 결과는 모두 탐색이다."""
    sub_report, sub_query, sub_features = st.tabs(["모니터링 리포트", "조건 조회", "지표 그룹 진단"])
    with sub_report:
        monitor_report_view()
    with sub_query:
        condition_query_view()
    with sub_features:
        feature_study_view()


def monitor_report_view():
    """사전 등록한 조건을 하나씩 바꿔 본 민감도 분석 리포트를 보여 주고, 버튼으로 다시 계산한다."""
    st.caption("사전 등록한 조건을 하나씩 바꿔 본 민감도 분석과 해석. 평일 자동 실행이 매주 금요일에 다시 만든다. "
               "고른 날짜와 상관없이 저장된 전체 데이터로 계산한 **탐색 결과**다.")
    report_path = Path(config.MONITOR_REPORT_PATH)
    if st.button("모니터링 다시 계산"):
        with st.spinner("계산 중(조건마다 다시 학습해 몇 분 걸린다)"):
            st.session_state.monitor = experiment.monitor(experiment_data())
    if "monitor" in st.session_state:
        for group, curves in st.session_state.monitor["curves"].items():
            st.markdown(f"**{group} 자산 곡선**")
            st.line_chart(pd.DataFrame(curves))
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("아직 리포트가 없다. 위 버튼으로 만든다.")


def condition_query_view():
    """조건 조합을 골라 '조회' 를 누르면 다시 계산해 비교 표와 자산 곡선을 보여 준다."""
    st.caption("선택한 날짜와 상관없이 저장된 전체 데이터로 다시 계산한다. 검증 구간을 본 뒤의 비교라 **탐색 결과**이며 공식 판정(기준 A·B)을 바꾸지 않는다. "
               "신호 시각(15:15)과 기사 수집 구간은 뉴스를 다시 채점해야 바뀌므로 여기서는 고를 수 없다.")
    base = config.EXPERIMENT_BASE
    names = {**config.RETRAIN_FREQS, **config.EXECUTIONS, **config.DOWN_ACTION_LABELS, **config.NEWS_SCORE_FIELDS,
             "none": "숫자 모델만", "adjust": "뉴스 보정", "base": f"운영 지표({len(config.FEATURE_SET_BASE)}개)", "base_v1": f"1~3일차 지표({len(config.FEATURE_SET_BASE_V1)}개)", "candidate": "후보 지표(A+B+D)", None: "쓰지 않음"}
    options = {**{k: config.EXPERIMENT_UI_OPTIONS[k] for k in ("horizon_days", "threshold")}, **experiment.CHOICES,
               **{k: v for k, v in config.EXPERIMENT_UI_OPTIONS.items() if k not in ("horizon_days", "threshold")}}
    # 폼 안의 선택은 '조회' 를 누를 때만 반영된다(조건을 바꿀 때마다 화면 전체를 다시 계산하지 않게)
    with st.form("experiment_form"):
        vary = {}
        cols = st.columns(3)
        for i, (key, values) in enumerate(options.items()):
            if base[key] not in values:
                values = [base[key], *values]
            vary[key] = cols[i % 3].multiselect(experiment.LABELS[key], values, default=[base[key]], key=f"exp_{key}",
                                                format_func=lambda v: names.get(v, str(v)))
        c1, c2 = st.columns(2)
        start = c1.text_input("시작일(비우면 자동)", "", key="exp_start")
        end = c2.text_input("끝일(비우면 자동)", "", key="exp_end")
        st.caption(f"조합은 최대 {config.EXPERIMENT_MAX_COMBOS}개. 정답 정의·지표 묶음·재학습 주기가 다르면 조합마다 다시 학습해 시간이 걸린다.")
        submitted = st.form_submit_button("조회")
    if submitted:
        vary = {k: v for k, v in vary.items() if v}
        cfgs = experiment.expand({**base, "start": start or None, "end": end or None}, vary)
        if len(cfgs) > config.EXPERIMENT_MAX_COMBOS:
            st.warning(f"조합 {len(cfgs)}개는 너무 많다(최대 {config.EXPERIMENT_MAX_COMBOS}개). 값을 줄여 다시 조회한다.")
        else:
            try:
                with st.spinner(f"조합 {len(cfgs)}개 계산 중"):
                    table, curves = experiment.compare(cfgs, experiment_data())
                st.session_state.experiment = (table, curves)
            except ValueError as e:
                st.warning(str(e))
    if "experiment" in st.session_state:
        table, curves = st.session_state.experiment
        st.markdown(experiment.to_markdown(table))
        st.line_chart(pd.DataFrame(curves))
        if st.button("보고서로 저장"):
            stamp = pd.Timestamp.now(tz=config.KST).strftime("%Y%m%d_%H%M")
            path = Path(config.EXPERIMENT_REPORT_DIR) / f"experiments_{stamp}_ui.md"
            path.write_text(f"# 조건 비교 실험 {stamp}\n\n" + experiment.to_markdown(table), encoding="utf-8", newline="\n")
            st.success(f"저장: {path}")


def feature_study_view():
    """지표 그룹을 빼고 더해 본 진단·추천 리포트와, 그룹을 직접 골라 기존 지표와 비교하는 폼."""
    st.caption("기존 지표를 그룹 단위로 하나씩 빼고 후보 그룹을 하나씩 더해 다시 학습한 **탐색 결과**다. 추천 묶음도 운영 신호를 바꾸지 않고, "
               "config.LIVE_EXTRA_FEATURE_SETS 에 넣으면 실시간 성적표에서 앞으로의 날로 함께 채점된다.")
    report_path = Path(config.FEATURE_STUDY_REPORT_PATH)
    if st.button("지표 그룹 진단 다시 계산"):
        with st.spinner("계산 중(그룹마다 다시 학습해 몇 분 걸린다)"):
            st.session_state.feature_study = experiment.feature_study(experiment_data())
    if "feature_study" in st.session_state:
        st.line_chart(pd.DataFrame(st.session_state.feature_study["curves"]))
    with st.form("feature_group_form"):
        groups = st.multiselect("직접 고를 지표 그룹", list(config.FEATURE_GROUPS), default=config.FEATURE_GROUPS_BASE,
                                format_func=lambda g: f"{config.FEATURE_GROUP_LABELS[g]} ({len(config.FEATURE_GROUPS[g])}개)")
        submitted = st.form_submit_button("기존 지표와 비교")
    if submitted:
        if not groups:
            st.warning("그룹을 하나 이상 고른다.")
        else:
            name = experiment.train.group_set_name(groups)
            with st.spinner("계산 중"):
                table, curves = experiment.compare(experiment.expand(dict(config.EXPERIMENT_BASE), {"feature_set": ["base", name]}),
                                                   experiment_data())
            st.markdown(experiment.to_markdown(table))
            st.line_chart(pd.DataFrame(curves))
            st.caption(f"이 묶음 이름: `{name}`")
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("아직 진단 리포트가 없다. 위 버튼으로 만든다.")


@st.cache_data(ttl=600)
def load_equity() -> pd.DataFrame:
    """검증 구간 모의 매매 자산 곡선(모델 vs 들고 있기)."""
    return pd.read_parquet(Path(config.PROCESSED_DIR) / f"equity_numeric_{config.DOWN_ACTION}.parquet")


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
    st.caption(f"모의 매매 규칙: 하락 신호 = {config.DOWN_ACTION_LABELS[config.DOWN_ACTION]}")
    st.caption("투자 권유가 아니라 모델 판단의 설명이다.")

st.title(f"삼성전자 {config.LABEL_HORIZON_DAYS}거래일 방향 신호 · {as_of.date()} {config.SIGNAL_CUTOFF_KST} 기준")
tab_ask, tab_data, tab_report = st.tabs(["질의응답", "신호·데이터", "리포트·조건 비교"])

# ---- 리포트·조건 비교 (고른 날짜와 상관없는 탐색) ----
with tab_report:
    experiment_tab()

# ---- 신호·데이터 (고른 날짜 시점) ----
with tab_data:
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

    tab_chart, tab_ind, tab_judge, tab_news, tab_perf, tab_live = st.tabs(
        ["가격·신호", "지표", "판단 기록", "뉴스", "성과", "실시간 성적"])
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
            if "downside_risk" in j["final"]:
                f = j["final"]
                st.caption(f"분류별: 삼성 직접 {f['samsung_score']:+d} · 반도체 {f['semis_score']:+d} · 거시 {f['macro_score']:+d}"
                           f" · 하락 위험 {f['downside_risk']}/{config.RISK_MAX}")
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
            st.caption(f"선택한 날짜 전까지의 모의 매매 자산 곡선(운영 모델 {len(config.FEATURE_SET_BASE)}개 지표·재학습 "
                       f"{config.RETRAIN_FREQS[config.RETRAIN_FREQ]}, 하락={config.DOWN_ACTION_LABELS[config.DOWN_ACTION]} 규칙). "
                       "과거 구간은 검증 구간을 본 뒤 재계산한 탐색 결과다. "
                       f"공식 판정은 1~3일차 {len(config.FEATURE_SET_BASE_V1)}개 지표·매월 재학습·하락=인버스 모델이다(day3_backtest.md).")
            st.altair_chart(chart, width="stretch")
    with tab_live:
        st.caption(f"평일 자동 실행으로 장 마감 전에 기록한 신호를 {config.LABEL_HORIZON_DAYS}거래일 뒤 정답이 확정되는 대로 채점한다(②안). "
                   "기존 모델과 후보 모델을 비교하고, 장 마감 뒤에 만든 사후 기록은 제외한다.")
        if st.button("성적표 새로 계산"):
            with st.spinner("계산 중"):
                scoreboard.build()
        board_path = Path(config.SCOREBOARD_PATH)
        if not board_path.exists():
            st.info("아직 성적표가 없다. 위 버튼으로 만든다.")
        else:
            board = json.loads(board_path.read_text(encoding="utf-8"))
            st.caption(f"계산 시각 {board['updated_at']} · 가격 마지막 날 {board.get('price_last_date', '-')}")
            for line in scoreboard.summary_lines(board):
                (st.warning if "경고" in line else st.write)(line)
            curves = {s["title"]: pd.Series(s["equity"]) for s in board.get("models", {}).values() if "equity" in s}
            if curves:
                st.line_chart(pd.DataFrame(curves))

# ---- 질의응답 ----
with tab_ask:
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
