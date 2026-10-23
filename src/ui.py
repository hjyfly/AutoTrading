"""Streamlit 화면. 실행: .venv\\Scripts\\streamlit run src/ui.py

사이드바에서 종목(config.MULTI_TICKERS 종목 레지스트리)과 기준 날짜를 고른다.
탭은 질의응답(에이전트), 고른 종목·날짜의 신호·차트·지표(뉴스·판단 기록은 config.NEWS_TICKERS 종목만), 고른 종목의 리포트·조건 비교다.
데이터는 모두 도구 함수(src/tools.py)로 읽으므로, 선택한 날짜 시점에 알 수 있던 정보만 보인다.
매매·주문 대상은 config.TRADE_TICKERS 뿐이고, 나머지 종목은 성적표로 판정하는 신호만 보여 준다(SERVICE.md 5.6).
"운영 종목인가"·"뉴스가 있는 종목인가"·"매매 종목인가"는 config 의 역할 설정(MAIN_TICKER·NEWS_TICKERS·TRADE_TICKERS)으로만 판단한다.
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

st.set_page_config(page_title=f"AutoTrading 종목별 {config.LABEL_HORIZON_DAYS}거래일 신호", layout="wide")
MAIN = config.MAIN_TICKER                       # 운영 모델 종목
MAIN_NAME = config.MULTI_TICKERS[MAIN][0]
NEWS_NAMES = "·".join(config.MULTI_TICKERS[c][0] for c in config.NEWS_TICKERS)
TRADE_NAMES = "·".join(config.MULTI_TICKERS[c][0] for c in config.TRADE_TICKERS)


@st.cache_data(ttl=600)
def load_signals(code: str = config.MAIN_TICKER) -> pd.DataFrame:
    """검증 구간 신호와 실시간으로 만든 신호를 합친다(뉴스 보정은 config.NEWS_ADJUST_IN_SIGNAL 일 때만 반영).
    다른 종목은 그 종목 알고리즘의 워크포워드 신호(8일차)에 실시간 기록을 덮어 쓴다."""
    if code != config.MAIN_TICKER:
        path = Path(config.SECTOR_DIR) / f"signals_{tools.ticker_algorithm(code)}_{code}.parquet"
        stored = pd.read_parquet(path)[["signal", "confidence"]] if path.exists() else pd.DataFrame(columns=["signal", "confidence"])
        if Path(config.MULTI_LIVE_PATH).exists():
            live = pd.read_parquet(config.MULTI_LIVE_PATH)
            live = live[live["code"] == code][["signal", "confidence"]]
            stored = pd.concat([stored[~stored.index.isin(live.index)], live]).sort_index()
        return stored
    numeric = pd.read_parquet(config.SIGNALS_NUMERIC_PATH)[["signal", "confidence"]]
    news = news_signal.load_matching() if config.NEWS_ADJUST_IN_SIGNAL else pd.DataFrame()  # 뉴스 보정을 끄면 숫자 신호 그대로
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
def load_prices(code: str = config.MAIN_TICKER) -> pd.Series:
    """저장된 종가(운영 종목은 기존 원본, 다른 종목은 data/raw/tickers)."""
    path = (Path(config.RAW_DIR) / "prices" / "samsung_ohlcv.parquet" if code == config.MAIN_TICKER
            else Path(config.TICKER_RAW_DIR) / f"{code}_ohlcv.parquet")
    return pd.read_parquet(path)["Close"].astype(float)


@st.cache_resource
def _experiment_data(extended: bool, code: str) -> experiment.Data:
    """조건 비교 실험용 원본 표와 학습 결과 보관소(같은 학습 조건은 한 번만 학습한다). 구간·종목마다 따로 둔다."""
    return experiment.Data(extended=extended, code=code)


def experiment_data(code: str) -> experiment.Data:
    """화면의 조건 비교 실험 데이터(고른 종목). 기존 검증 구간 하나로만 계산한다(확장 검증은 리포트 파일로만 남긴다)."""
    return _experiment_data(False, code)


def experiment_tab(code: str):
    """고른 종목의 모니터링 리포트·조건 조회·지표 그룹 진단과 전 종목 비교를 하위 탭으로 보여 준다. 결과는 모두 탐색이다."""
    name = config.MULTI_TICKERS[code][0]
    st.caption(f"**{name}** 기준(사이드바에서 종목을 바꾼다)."
               + ("" if code == MAIN else f" 운영 종목과 같은 설정(S0 단독, 지표 {len(config.FEATURE_SET_BASE)}개)으로 계산하며, 분야 알고리즘의 추가 지표·뉴스 조건·인버스 규칙은 이 종목에 없다."))
    sub_report, sub_query, sub_features, sub_tickers = st.tabs([f"모니터링 리포트 · {name}", f"조건 조회 · {name}", f"지표 그룹 진단 · {name}", "종목 비교(전체)"])
    with sub_tickers:
        path = Path(config.TICKER_ANALYSIS_REPORT_PATH)
        if st.button("종목 비교 다시 계산 (1~2분, LLM 없음)"):
            with st.spinner(f"{len(config.MULTI_TICKERS)}종목 분석 중"):
                from src.pipeline import multi_ticker
                multi_ticker.analysis()
                st.cache_data.clear()
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8").replace("../../", ""))
        else:
            st.info("아직 종목 비교 리포트가 없다. 위 버튼으로 만든다.")
    with sub_report:
        monitor_report_view(code)
    with sub_query:
        condition_query_view(code)
    with sub_features:
        feature_study_view(code)


def monitor_report_view(code: str):
    """고른 종목의 사전 등록 조건을 하나씩 바꿔 본 민감도 분석 리포트를 보여 주고, 버튼으로 다시 계산한다."""
    st.caption("사전 등록한 조건을 하나씩 바꿔 본 민감도 분석과 해석. 평일 자동 실행이 매주 금요일에 다시 만든다. "
               "고른 날짜와 상관없이 저장된 전체 데이터로 계산한 **탐색 결과**다.")
    report_path = experiment.ticker_path(config.MONITOR_REPORT_PATH, code)
    key = f"monitor_{code}"
    if st.button("모니터링 다시 계산", key=f"btn_{key}"):
        with st.spinner("계산 중(조건마다 다시 학습해 종목당 4~5분 걸린다)"):
            st.session_state[key] = experiment.monitor(experiment_data(code))
    if key in st.session_state:
        for group, curves in st.session_state[key]["curves"].items():
            st.markdown(f"**{group} 자산 곡선**")
            st.line_chart(pd.DataFrame(curves))
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("아직 리포트가 없다. 위 버튼으로 만든다.")


def condition_query_view(code: str):
    """고른 종목으로 조건 조합을 골라 '조회' 를 누르면 다시 계산해 비교 표와 자산 곡선을 보여 준다."""
    st.caption("선택한 날짜와 상관없이 저장된 전체 데이터로 다시 계산한다. 검증 구간을 본 뒤의 비교라 **탐색 결과**이며 공식 판정(기준 A·B)을 바꾸지 않는다. "
               "신호 시각(15:15)과 기사 수집 구간은 뉴스를 다시 채점해야 바뀌므로 여기서는 고를 수 없다.")
    base = config.EXPERIMENT_BASE
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
                                                format_func=lambda v, key=key: experiment.value_label(key, v))
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
                    table, curves = experiment.compare(cfgs, experiment_data(code))
                st.session_state[f"experiment_{code}"] = (table, curves)
            except ValueError as e:
                st.warning(str(e))
    if f"experiment_{code}" in st.session_state:
        table, curves = st.session_state[f"experiment_{code}"]
        st.markdown(experiment.to_markdown(table))
        st.line_chart(pd.DataFrame(curves))
        if st.button("보고서로 저장"):
            stamp = pd.Timestamp.now(tz=config.KST).strftime("%Y%m%d_%H%M")
            path = Path(config.EXPERIMENT_REPORT_DIR) / f"experiments_{stamp}_ui{'' if code == MAIN else '_' + code}.md"
            path.write_text(f"# 조건 비교 실험 {stamp} — {config.MULTI_TICKERS[code][0]}\n\n" + experiment.to_markdown(table), encoding="utf-8", newline="\n")
            st.success(f"저장: {path}")


def feature_study_view(code: str):
    """고른 종목으로 지표 그룹을 빼고 더해 본 진단·추천 리포트와, 그룹을 직접 골라 기존 지표와 비교하는 폼."""
    st.caption("기존 지표를 그룹 단위로 하나씩 빼고 후보 그룹을 하나씩 더해 다시 학습한 **탐색 결과**다. 추천 묶음도 운영 신호를 바꾸지 않고, "
               "config.LIVE_EXTRA_FEATURE_SETS 에 넣으면 실시간 성적표에서 앞으로의 날로 함께 채점된다.")
    report_path = experiment.ticker_path(config.FEATURE_STUDY_REPORT_PATH, code)
    key = f"feature_study_{code}"
    if st.button("지표 그룹 진단 다시 계산", key=f"btn_{key}"):
        with st.spinner("계산 중(그룹마다 다시 학습해 몇 분 걸린다)"):
            st.session_state[key] = experiment.feature_study(experiment_data(code))
    if key in st.session_state:
        st.line_chart(pd.DataFrame(st.session_state[key]["curves"]))
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
                                                   experiment_data(code))
            st.markdown(experiment.to_markdown(table))
            st.line_chart(pd.DataFrame(curves))
            st.caption(f"이 묶음 이름: `{name}`")
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("아직 진단 리포트가 없다. 위 버튼으로 만든다.")


@st.cache_data(ttl=600)
def load_ticker_analysis(name: str) -> pd.DataFrame:
    """종목별 분석 표(multi_ticker.analysis 결과). 없으면 빈 표."""
    path = Path(config.TICKER_ANALYSIS_DIR) / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def ticker_performance_view(code: str, as_of: pd.Timestamp):
    """다른 종목 성과: 검증 구간 자산 곡선(고른 날짜 전까지), 구간별·연도별 성적, 5일 수익률과 관련 큰 지표. 모두 탐색이다."""
    periods = load_ticker_analysis("periods")
    if periods.empty:
        st.info("아직 종목 분석이 없다. '리포트·조건 비교 → 종목 비교'에서 계산한다.")
        return
    mine = periods[periods["code"] == code]
    st.caption(f"선정 알고리즘 {mine['algorithm'].iloc[0]} 의 워크포워드 신호로 운영 종목({MAIN_NAME})과 같은 규칙(하락 = 현금, 수수료 {config.FEE_RATE:.1%})에서 계산했다. "
               "7·8일차에 이미 본 구간이라 모두 탐색이고, 판정은 실시간 성적표로만 한다. 뉴스는 쓰지 않았다. "
               "자산 곡선은 고른 날짜 전까지만 그리고, 아래 표는 고른 날짜와 관계없이 데이터 끝까지의 사후 성적이다.")
    eq = load_ticker_analysis("equity")
    eq = eq[(eq["code"] == code) & (eq.index < as_of)]
    if eq.empty:
        st.info(f"검증 구간({config.VALID_START}~) 이전 날짜라 자산 곡선이 없다.")
    else:
        long = eq[["model", "hold"]].reset_index(names="date").melt("date", var_name="전략", value_name="자산")
        long["전략"] = long["전략"].map({"model": "모델(하락 = 현금)", "hold": "그냥 들고 있기"})
        st.altair_chart(alt.Chart(long).mark_line().encode(x=alt.X("date:T", title=None), y=alt.Y("자산:Q", title="자산(시작 1.0)"),
                                                            color="전략:N").properties(height=260), width="stretch")
    pct = lambda v: "-" if pd.isna(v) else f"{v:.1%}"
    mark = lambda v: "-" if v is None or pd.isna(v) else ("통과" if v else "실패")
    st.dataframe(pd.DataFrame([{"구간": r["period"], "일수": r["days"], "모델 수익": pct(r["return"]), "모델 MDD": pct(r["mdd"]),
                                "들고 있기": pct(r["hold_return"]), "들고 있기 MDD": pct(r["hold_mdd"]), "정확도": pct(r["accuracy"]),
                                "늘 최빈 정답": pct(r["most_common"]), "하락 신호 적중": pct(r["hit_하락"]), "보유일": pct(r["hold_days"]),
                                "A": mark(r["A"]), "B": mark(r["B"]), "C": mark(r["C"])} for _, r in mine.iterrows()]),
                 width="stretch", hide_index=True)
    years = load_ticker_analysis("yearly")
    years = years[years["code"] == code]
    if len(years):
        st.markdown("**연도별 장세 (각 연도 현금에서 시작)**")
        st.dataframe(pd.DataFrame([{"연도": int(y["year"]), "장세": y["regime"], "모델 수익": pct(y["return"]), "모델 MDD": pct(y["mdd"]),
                                    "들고 있기": pct(y["hold_return"]), "들고 있기 MDD": pct(y["hold_mdd"]), "정확도": pct(y["accuracy"])}
                                   for _, y in years.iterrows()]), width="stretch", hide_index=True)
    ic = load_ticker_analysis("feature_ic")
    ic = ic[ic["code"] == code].assign(abs_ic=lambda d: d["ic"].abs()).sort_values("abs_ic", ascending=False).head(config.TICKER_ANALYSIS_TOP_FEATURES)
    if len(ic):
        st.markdown("**5일 수익률과 관련이 큰 지표 (IC, 겹침 보정 t)**")
        st.dataframe(ic[["feature", "group", "ic", "nw_t"]].rename(columns={"feature": "지표", "group": "그룹", "ic": "IC", "nw_t": "NW t"}).round(3),
                     width="stretch", hide_index=True)


@st.cache_data(ttl=600)
def load_equity() -> pd.DataFrame:
    """검증 구간 모의 매매 자산 곡선(모델 vs 들고 있기)."""
    return pd.read_parquet(Path(config.PROCESSED_DIR) / f"equity_numeric_{config.DOWN_ACTION}.parquet")


def price_chart(as_of: pd.Timestamp, days: int, code: str = config.MAIN_TICKER) -> alt.Chart:
    """선택한 날짜 전날까지의 종가와 그 기간의 신호 표시."""
    close = load_prices(code)
    close = close[close.index < as_of].tail(days)
    signals = load_signals(code).reindex(close.index).dropna()
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


def _mark(v) -> str:
    """성공 기준 표시."""
    return "-" if v is None else ("통과" if v else "실패")


def samsung_board_view(board: dict):
    """운영 종목의 운영·비교 모델 성적표(요약 줄, 기준 표, 자산 곡선)."""
    st.markdown(f"**{MAIN_NAME} 운영·비교 모델**")
    for line in scoreboard.summary_lines(board):
        (st.warning if "경고" in line else st.write)(line)
    rows = [{"모델": s["title"], "판정 상태": s["criteria"]["status"], "A 위험 대비 수익": _mark(s["criteria"]["A"]["passed"]),
             "B 하락 신호 적중": _mark(s["criteria"]["B"]["passed"]), "C 반등 지키기": _mark(s["criteria"]["C"]["passed"]),
             "승격": ("검토 대상" if s["promotion"]["eligible"] else "아님") if "promotion" in s else "운영"}
            for s in board.get("models", {}).values() if s.get("criteria")]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    curves = {s["title"]: pd.Series(s["equity"]) for s in board.get("models", {}).values() if "equity" in s}
    if curves:
        st.line_chart(pd.DataFrame(curves))


def tickers_board_view(board: dict, only: str | None = None):
    """다른 종목 최근 신호와 성적표. only 를 주면 그 종목만 보여 준다."""
    st.caption("종목마다 쓰는 알고리즘(S0 단독·S4 분야 공동 학습)으로 매일 기록한 신호를 그 종목 들고 있기와 같은 기준 A·B·C 로 채점한다.")
    if Path(config.MULTI_LIVE_PATH).exists():
        recent = scoreboard.load_ticker_live()
        if only:
            recent = recent[recent["code"] == only]
        if len(recent):
            recent = recent[recent.index == recent.index.max()]
            in_time = scoreboard.realtime_mask(recent)
            st.markdown(f"**최근 신호 ({recent.index.max().date()})**")
            st.dataframe(pd.DataFrame({
                "종목": [f"{config.MULTI_TICKERS.get(c, (c,))[0]}({c})" for c in recent["code"]],
                "신호": recent["signal"].values, "확신도": [f"{v:.0%}" for v in recent["confidence"]],
                "알고리즘": recent["model"].values,
                "기록": ["실시간" if ok else "사후(성적 제외)" for ok in in_time]}), width="stretch", hide_index=True)
    entries = {k: v for k, v in board.get("tickers", {}).items() if only is None or v.get("code") == only}
    if not entries:
        st.info("아직 이 종목의 실시간 성적이 없다. 평일 15:20 자동 실행 기록부터 쌓인다.")
        return
    for s in entries.values():
        line = scoreboard.entry_line(s)
        (st.warning if "경고" in line else st.write)(line)
    rows = [{"종목·알고리즘": s["title"], "정답 확정": s["n_matured"],
             "정확도": "-" if s["accuracy"] is None else f"{s['accuracy']:.0%}",
             "판정 상태": s["criteria"]["status"] if "criteria" in s else "-",
             "A": _mark(s["criteria"]["A"]["passed"]) if "criteria" in s else "-",
             "B": _mark(s["criteria"]["B"]["passed"]) if "criteria" in s else "-",
             "C": _mark(s["criteria"]["C"]["passed"]) if "criteria" in s else "-"}
            for s in entries.values()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    curves = {s["title"]: pd.Series(s["equity"]) for s in entries.values() if "equity" in s}
    if curves:
        st.line_chart(pd.DataFrame(curves))


# ---- 사이드바: 종목, 기준 날짜, 오늘 신호 만들기, 비용 ----
with st.sidebar:
    st.header("종목")
    code = st.selectbox("신호를 볼 종목", list(config.MULTI_TICKERS),
                        format_func=lambda c: f"{config.MULTI_TICKERS[c][0]} ({c})" + "".join(
                            f" · {label}" for label, on in (("운영", c == MAIN), ("매매", c in config.TRADE_TICKERS)) if on))
    name = tools.ticker_name(code)
    is_main = code == MAIN                       # 운영 모델(숫자 모델 운영 신호·비교 줄·과거 공식 판정)이 있는 종목
    has_news = code in config.NEWS_TICKERS       # 뉴스 채점·판단 기록·공시가 있는 종목
    st.header("기준 날짜")
    picked = st.date_input(f"이 날짜 {config.SIGNAL_CUTOFF_KST} 시점으로 봄", value=tools.resolve_as_of(None).date())
    as_of = pd.Timestamp(picked)
    use_date_in_chat = st.checkbox("질문에 날짜가 없으면 이 날짜로 묻기", value=True)
    st.divider()
    st.header("신호 만들기")
    st.caption(f"{MAIN_NAME} 저장 신호는 {config.DATA_END} 까지. 그 뒤 날짜는 실시간 데이터로 만든다. "
               f"평일 15:20 자동 실행이 {len(config.MULTI_TICKERS)}개 종목 신호를 모두 기록하므로 보통은 누르지 않아도 된다.")
    if not is_main and st.button(f"다른 종목 신호 만들기 ({len(config.MULTI_TICKERS) - 1}종목)"):
        with st.spinner("다른 종목 원본을 덧붙이고 신호를 만드는 중"):
            try:
                from src.pipeline import multi_ticker
                multi_ticker.refresh(as_of)
                st.write("\n".join(multi_ticker.summary_lines(multi_ticker.live_signals(str(as_of.date())))))
                st.cache_data.clear()
            except Exception as e:
                st.error(f"실패: {e}")
    if is_main and st.button("숫자 모델 신호 만들기"):
        with st.spinner("실시간 데이터로 학습·예측 중"):
            try:
                live_signal.make_signal(str(as_of.date()))
                st.cache_data.clear()
                st.success("완료")
            except Exception as e:
                st.error(f"실패: {e}")
    if is_main and has_news and st.button("뉴스 채점까지, 참고용 (약 $0.27, 3~4분)"):  # 뉴스 채점·추출은 지금 운영 종목 파이프라인 전용
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
    st.caption(f"모의 매매 규칙: 하락 신호 = {config.DOWN_ACTION_LABELS[config.DOWN_ACTION]}. 주문은 {TRADE_NAMES}만, 사람이 직접 한다.")
    st.caption("투자 권유가 아니라 모델 판단의 설명이다.")

st.title(f"{name} {config.LABEL_HORIZON_DAYS}거래일 방향 신호 · {as_of.date()} {config.SIGNAL_CUTOFF_KST} 기준")
if not is_main:
    st.caption(f"정식 신호(실시간 성적표로 판정) · {'매매 종목' if code in config.TRADE_TICKERS else '매매·주문 대상 아님'} · 알고리즘 {tools.ticker_algorithm(code)} "
               + ("" if has_news else f"· 뉴스 채점·판단 기록·공시는 {NEWS_NAMES}만 있다"))
tab_ask, tab_data, tab_report = st.tabs(["질의응답", "신호·데이터", "리포트·조건 비교"])

# ---- 리포트·조건 비교 (고른 날짜와 상관없는 탐색) ----
with tab_report:
    experiment_tab(code)

# ---- 신호·데이터 (고른 날짜 시점) ----
with tab_data:
    sig = tools.signal_for(str(as_of.date()), code)
    c1, c2, c3 = st.columns(3)
    final = sig.get("with_news", {}).get("signal") or sig.get("numeric", {}).get("signal")
    c1.metric("최종 신호", final or "없음")
    if not is_main and "numeric" in sig:
        c2.metric("확신도", f"{sig['numeric']['confidence']:.0%}")
        c3.metric("알고리즘", sig["numeric"]["algorithm"])
        st.caption(f"출처: {sig['numeric']['source']}")
    if is_main and "numeric" in sig:
        c2.metric("숫자 모델", sig["numeric"]["signal"], f"확신도 {sig['numeric']['confidence']:.0%}", delta_color="off")
    if "with_news" in sig:
        news_note = ("보정됨" if sig["with_news"]["adjusted"] else "보정 없음") if config.NEWS_ADJUST_IN_SIGNAL else "참고용"
        c3.metric("뉴스 합의", f"{sig['with_news']['news_consensus']:+d}", news_note, delta_color="off")
    if "candidate" in sig:
        st.caption(f"비교용 후보 모델(A+B+D 지표 추가): **{sig['candidate']['signal']}** (확신도 {sig['candidate']['confidence']:.0%}) "
                   "— 최종 신호가 아니며 앞으로의 성과로 기존 모델과 비교 중")
    live_path = Path(config.PROCESSED_DIR) / "signals_live.parquet"
    if is_main and live_path.exists():  # 그날 실시간으로 기록한 비교 줄(최종 신호 아님). 그날 기록이라 고른 날짜 시점 규칙을 지킨다
        live = pd.read_parquet(live_path)
        today_rows = live[(live.index == as_of.normalize()) & live["model"].isin([*config.LIVE_EXTRA_FEATURE_SETS, config.LIVE_RISK_MODEL])]
        for model, r in today_rows.set_index("model").iterrows():
            if model == config.LIVE_RISK_MODEL:
                st.caption(f"비교용 하락 위험 조건 {config.LIVE_RISK_CONDITION}: 목표 비중 **{float(r['weight']):.0%}** "
                           f"(위험 확률 {float(r['p_risk']):.0%}) — 최종 신호가 아니며 앞으로의 성과로 비교 중")
            else:
                st.caption(f"비교용 지표 묶음 `{model}`: **{r['signal']}** (확신도 {float(r['confidence']):.0%}) — 최종 신호가 아니며 앞으로의 성과로 비교 중")
    if "note" in sig and (is_main or "numeric" not in sig):
        st.info(sig["note"])

    tab_chart, tab_ind, tab_judge, tab_news, tab_perf, tab_live = st.tabs(
        ["가격·신호", "지표", "판단 기록", "뉴스", "성과", "실시간 성적"])
    with tab_chart:
        days = st.slider("기간(거래일)", 20, 250, 120)
        st.altair_chart(price_chart(as_of, days, code), width="stretch")
    with tab_ind:
        try:
            ind = tools.indicators(str(as_of.date()), code)
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
        j = tools.news_judgement(str(as_of.date())) if is_main and has_news else {"note": f"뉴스 채점·판단 기록은 {NEWS_NAMES}만 있다. 이 종목은 숫자 모델 신호만 낸다"}
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
        if is_main and has_news:  # 판단 기록·기사 피처는 지금 운영 종목 파이프라인 전용
            feats = tools.news_features(str(as_of.date()))
            st.markdown("**기사별 추출 (9일차, 참고)**")
            if "articles" not in feats:
                st.caption(feats["note"])
            else:
                if feats["alert"]:
                    from src.pipeline import news_signal
                    st.warning(news_signal.render_alert(as_of, feats["alert"], config.DAY9_ALERT_STATUS))
                v1 = feats.get("v1")
                if v1 and "signal" in v1:
                    st.caption(f"뉴스 모델 V1(참고, 장부 V1 확인 통과 · 운영 신호 아님): **{v1['signal']}** "
                               f"(상승−하락 확률 {v1['score']:+.2f}, {v1['retrained_on']} 재학습, 학습 {v1['train_rows']}행)")
                st.caption(f"회사명을 익명(A사)으로 바꿔 기사마다 뽑았다 · 그날 모은 기사 {feats['n_collected']}건 · 프롬프트 버전 {feats['prompt_version']}")
                table = pd.DataFrame(feats["articles"])
                if len(table):
                    table["one_line_summary"] = table["one_line_summary"].str.replace(config.ANONYMIZED_NAME, MAIN_NAME)
                    st.dataframe(table.rename(columns={"published_kst": "발행(KST)", "title": "제목", "category": "분류", "event_type": "사건",
                                                       "direction": "방향", "magnitude": "크기", "novelty": "새로움", "downside_risk": "하락 위험",
                                                       "relevance": "관련도", "one_line_summary": "한 줄 요약"}), width="stretch", hide_index=True)
    with tab_news:
        if not has_news:
            st.info(f"뉴스는 {NEWS_NAMES} 관련 기사만 모은다. 아래는 같은 날 {MAIN_NAME} 뉴스(시장 전반 참고용)다.")
        n = tools.news_for_signal(str(as_of.date()), limit=30)
        st.caption(n["source"])
        if n["articles"]:
            st.dataframe(pd.DataFrame(n["articles"])[["published_kst", "source", "title"]], width="stretch", hide_index=True)
    with tab_perf:
        chart = equity_chart(as_of) if is_main else None
        if not is_main:
            ticker_performance_view(code, as_of)
        elif chart is None:
            st.info(f"검증 구간({config.VALID_START}~) 이전 날짜다.")
        else:
            st.caption(f"선택한 날짜 전까지의 모의 매매 자산 곡선(운영 모델 {len(config.FEATURE_SET_BASE)}개 지표·재학습 "
                       f"{config.RETRAIN_FREQS[config.RETRAIN_FREQ]}, 하락={config.DOWN_ACTION_LABELS[config.DOWN_ACTION]} 규칙). "
                       "과거 구간은 검증 구간을 본 뒤 재계산한 탐색 결과다. "
                       f"공식 판정은 1~3일차 {len(config.FEATURE_SET_BASE_V1)}개 지표·매월 재학습·하락=인버스 모델이다(day3_backtest.md).")
            st.altair_chart(chart, width="stretch")
    with tab_live:
        st.caption(f"평일 자동 실행으로 장 마감 전에 기록한 신호를 {config.LABEL_HORIZON_DAYS}거래일 뒤 정답이 확정되는 대로 채점한다. "
                   f"{MAIN_NAME} 운영·비교 모델과 다른 종목을 SERVICE.md 5절 성공 기준(A 위험 대비 수익, B 하락 신호 적중, C 반등 지키기)으로 판정한다. "
                   f"확정 신호 {config.LIVE_PROMOTION_MIN_SIGNALS}개 전까지는 참고 값이다. 장 마감 뒤에 만든 사후 기록은 제외한다.")
        if st.button("성적표 새로 계산"):
            with st.spinner("계산 중"):
                scoreboard.build()
        board_path = Path(config.SCOREBOARD_PATH)
        if not board_path.exists():
            st.info("아직 성적표가 없다. 위 버튼으로 만든다.")
        else:
            board = json.loads(board_path.read_text(encoding="utf-8"))
            st.caption(f"계산 시각 {board['updated_at']} · 가격 마지막 날 {board.get('price_last_date', '-')}")
            if is_main:
                samsung_board_view(board)
                st.subheader("다른 종목 (SERVICE.md 5.6, 매매·주문 대상 아님)")
                tickers_board_view(board)
            else:
                st.subheader(f"{name} 실시간 성적 (SERVICE.md 5.6, 매매·주문 대상 아님)")
                tickers_board_view(board, code)
                with st.expander(f"{MAIN_NAME}(운영) · 다른 종목 성적 보기"):
                    samsung_board_view(board)
                    tickers_board_view(board)

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
    st.caption(f"{len(config.MULTI_TICKERS)}개 종목 신호·가격·수급·지표를 물을 수 있다. 뉴스·판단 기록·공시는 {NEWS_NAMES}, 과거 성과 요약은 {MAIN_NAME}만 있다. "
               "질문에 종목이 없으면 사이드바에서 고른 종목으로 묻는다.")
    question = st.chat_input("예: 이날 신호와 이유는? / SK하이닉스 외국인 수급은? / 삼성전자 모델 성과는?")
    if question:
        asked = question
        if not is_main and not any(v[0] in question or c in question for c, v in config.MULTI_TICKERS.items()):
            asked = f"{name} {asked}"
        if use_date_in_chat and agent.extract_as_of(question) is None:
            asked = f"{as_of.date()} 기준: {asked}"
        st.session_state.history.append({"role": "user", "content": asked})
        with st.spinner("답을 만드는 중 (20~40초)"):
            result = agent.answer(asked)
        st.session_state.history.append({"role": "assistant", "content": result["answer"], "result": result})
        st.rerun()
