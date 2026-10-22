"""조건 비교 실험: 설정값을 바꿔 가며 지금까지 모은 데이터로 학습·모의 매매를 다시 돌려 한 표로 비교한다.

바꿀 수 있는 조건(config.EXPERIMENT_BASE 의 열쇠):
- horizon_days, threshold: 정답 정의(t+N거래일 ±기준). 바꾸면 모델을 그 정답으로 다시 학습한다
- feature_set: 지표 묶음(base·candidate)
- retrain_freq: 재학습 주기(W·M·Q)
- execution: 체결 시점(t 종가 · t+1 시가)
- down_action: 하락 신호 때 행동(현금·인버스)
- news_mode: none(숫자 모델만) · adjust(뉴스 점수로 한 단계 보정)
- news_field, news_window, news_risk_veto, news_min_score, news_min_conf: 뉴스 보정 방법
- start, end: 비교 구간(비우면 검증 구간, 뉴스를 쓰는 조건이 섞이면 뉴스 채점 구간)

신호 시각(15:15)과 기사 수집 구간 자체는 바꾸지 않는다. 바꾸면 뉴스를 다시 채점해야 해서 저장된 점수로는 비교할 수 없다.
여기서 나온 숫자는 모두 검증 구간을 보고 고른 조건의 '탐색' 결과이고 공식 판정(기준 A·B)을 바꾸지 않는다.
"""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from src import config
from src.pipeline import backtest as bt
from src.pipeline import news_signal, scoreboard, train
from src.pipeline.labels import DOWN, UP, make_labels

TRAIN_KEYS = ("horizon_days", "threshold", "feature_set", "retrain_freq")
LABELS = {  # 보고서·화면에 쓰는 한국어 이름
    "horizon_days": "정답 기간(거래일)", "threshold": "정답 기준(±)", "feature_set": "지표 묶음", "retrain_freq": "재학습 주기",
    "execution": "체결 시점", "down_action": "하락 행동", "news_mode": "뉴스 사용", "news_field": "뉴스 점수",
    "news_window": "뉴스 평균(채점일 수)", "news_risk_veto": "하락 위험 거부", "news_min_score": "보정 점수 기준",
    "news_min_conf": "보정 확신도 기준", "start": "시작", "end": "끝",
}
CHOICES = {  # 고를 수 있는 값이 정해진 조건
    "feature_set": list(config.FEATURE_SETS), "retrain_freq": list(config.RETRAIN_FREQS), "execution": list(config.EXECUTIONS),
    "down_action": list(config.DOWN_ACTIONS), "news_mode": ["none", "adjust"], "news_field": list(config.NEWS_SCORE_FIELDS),
}


class Data:
    """실험마다 다시 읽지 않도록 원본 표와 학습 결과를 한 번만 읽고 보관한다."""

    def __init__(self):
        raw = Path(config.RAW_DIR) / "prices"
        self.samsung = pd.read_parquet(raw / "samsung_ohlcv.parquet")
        self.inverse = pd.read_parquet(raw / f"inverse_{config.INVERSE_ETF_CODE}.parquet")
        self.features = pd.read_parquet(Path(config.PROCESSED_DIR) / "features.parquet")
        self.scores = pd.read_parquet(config.SCORES_PATH) if Path(config.SCORES_PATH).exists() else None
        self._signals = {}

    def labels(self, horizon: int, threshold: float) -> pd.DataFrame:
        """정답 기간·기준을 바꾼 정답 표."""
        return make_labels(self.samsung["Close"], horizon, threshold)

    def signals(self, cfg: dict) -> pd.DataFrame:
        """숫자 모델 워크포워드 신호. 학습에 영향을 주는 조건이 같으면 다시 학습하지 않는다."""
        key = tuple(cfg[k] for k in TRAIN_KEYS)
        if key not in self._signals:
            labels = self.labels(cfg["horizon_days"], cfg["threshold"])
            self._signals[key] = train.walk_forward(self.features, labels, feature_set=cfg["feature_set"], freq=cfg["retrain_freq"])
        return self._signals[key]

    def news_range(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """뉴스 점수 표의 첫날과 끝날. 점수 표가 없으면 None."""
        if self.scores is None or self.scores.empty:
            return None
        return self.scores["signal_date"].min(), self.scores["signal_date"].max()


def trade_prices(ohlcv: pd.DataFrame, execution: str) -> pd.Series:
    """t일 신호를 체결하는 가격을 t 행에 둔다. next_open 은 다음 거래일 시가다.

    다음 날 시가는 체결 가격으로만 쓰고 신호를 만드는 입력에는 넣지 않는다. 자산은 체결 가격끼리의 변화로 계산한다.
    """
    if execution == "close":
        return ohlcv["Close"]
    if execution == "next_open":
        return ohlcv["Open"].shift(-1)
    raise ValueError(f"알 수 없는 체결 시점: {execution}")


def expand(base: dict, vary: dict[str, list]) -> list[dict]:
    """기준 조건에서 vary 의 값들을 모든 조합으로 바꾼 조건 목록."""
    unknown = (set(base) | set(vary)) - set(config.EXPERIMENT_BASE)
    if unknown:
        raise ValueError(f"알 수 없는 조건: {sorted(unknown)}")
    keys = list(vary)
    return [{**base, **dict(zip(keys, values))} for values in itertools.product(*(vary[k] for k in keys))] or [dict(base)]


def default_period(cfgs: list[dict], data: Data) -> tuple[str, str]:
    """조건 목록 전체가 같은 구간에서 비교되도록 기본 구간을 정한다."""
    if any(c["news_mode"] != "none" for c in cfgs):
        rng = data.news_range()
        if rng is None:
            raise ValueError("뉴스를 쓰는 조건이 있는데 뉴스 점수 표가 없다")
        return str(rng[0].date()), str(rng[1].date())
    return config.VALID_START, config.DATA_END


def run_one(cfg: dict, data: Data) -> dict:
    """조건 하나로 신호를 만들고 모의 매매해 핵심 숫자와 자산 곡선을 돌려준다."""
    for key, allowed in CHOICES.items():
        if cfg[key] not in allowed:
            raise ValueError(f"{LABELS[key]} 값 '{cfg[key]}' 은 {allowed} 중 하나여야 한다")
    numeric = data.signals(cfg)
    signals = numeric
    if cfg["news_mode"] == "adjust":
        signals = news_signal.news_signals(numeric, data.scores, field=cfg["news_field"], window=cfg["news_window"],
                                           risk_veto=cfg["news_risk_veto"], min_score=cfg["news_min_score"],
                                           min_conf=cfg["news_min_conf"])
    samsung_px = trade_prices(data.samsung, cfg["execution"])
    inverse_px = trade_prices(data.inverse, cfg["execution"])
    dates = signals.loc[cfg["start"]:cfg["end"]].index
    dates = dates[samsung_px.reindex(dates).notna()]  # 다음 날 시가가 아직 없는 마지막 날은 뺀다
    if len(dates) < 2:
        raise ValueError(f"구간 {cfg['start']}~{cfg['end']} 에 비교할 날이 부족하다")
    sig = signals.loc[dates, "signal"]
    sim = bt.simulate(sig, samsung_px, inverse_px, cfg["down_action"])
    hold = bt.buy_and_hold(samsung_px, dates)
    labels = data.labels(cfg["horizon_days"], cfg["threshold"])["label"]
    m, h = bt.summary(sim["equity"]), bt.summary(hold)
    direction = bt.direction_hit_rate(sig, labels)
    has_label = labels.reindex(dates).notna()
    share = sig.value_counts(normalize=True)
    return {
        **cfg, "start": str(dates.min().date()), "end": str(dates.max().date()), "days": len(dates),
        "total_return": m["total_return"], "mdd": m["mdd"], "calmar": m["calmar"],
        "hold_return": h["total_return"], "hold_mdd": h["mdd"],
        "trades": int((sim["fee"] > 0).sum()), "fee_paid": float(1 - (1 - sim["fee"]).prod()),  # trades = 포지션을 바꾼 날 수
        "hit_rate": direction["hit_rate"], "critical_error": direction["critical_error_rate"],
        "accuracy": float((sig[has_label] == labels.reindex(dates)[has_label]).mean()) if has_label.any() else float("nan"),
        "up_share": float(share.get(UP, 0)), "down_share": float(share.get(DOWN, 0)),
        "judge_a_ref": bt.judge_a(sim["equity"], hold)["passed"],
        "news_adjusted": int(signals.loc[dates, "adjusted"].sum()) if "adjusted" in signals else None,  # 뉴스로 신호를 바꾼 날 수
        "equity": sim["equity"], "hold_equity": hold,
    }


def compare(cfgs: list[dict], data: Data | None = None) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """여러 조건을 같은 구간에서 돌려 결과 표와 자산 곡선(조건 이름 → 곡선)을 돌려준다."""
    data = data or Data()
    start, end = common_period(cfgs, data)
    rows, curves = [], {}
    for i, cfg in enumerate(cfgs, 1):
        cfg = {**cfg, "start": cfg["start"] or start, "end": cfg["end"] or end}
        r = run_one(cfg, data)
        name = f"조건 {i}"
        curves[name] = r.pop("equity")
        curves.setdefault(f"들고 있기({config.EXECUTIONS[cfg['execution']]})", r.pop("hold_equity"))
        rows.append({"name": name, **r})
    return pd.DataFrame(rows), curves


def varied_columns(table: pd.DataFrame) -> list[str]:
    """조건 중 실제로 값이 달라진 열."""
    return [k for k in config.EXPERIMENT_BASE if k in table and table[k].astype(str).nunique() > 1]


def to_markdown(table: pd.DataFrame, note: str = "") -> str:
    """결과 표를 보고서용 마크다운으로 만든다. 탐색 결과·다중 비교 경고를 붙인다."""
    varied = varied_columns(table)
    fixed = {LABELS[k]: table[k].iloc[0] for k in config.EXPERIMENT_BASE if k not in varied and k in table}
    show = table[["name", *varied, "days", "total_return", "mdd", "calmar", "hold_return", "hold_mdd", "trades",
                  "fee_paid", "hit_rate", "critical_error", "accuracy", "judge_a_ref"]].copy()
    for col in ["total_return", "mdd", "hold_return", "hold_mdd", "fee_paid", "hit_rate", "critical_error", "accuracy"]:
        show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}" if col in ("total_return", "hold_return") else f"{v:.1%}")
    show["calmar"] = show["calmar"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    show["judge_a_ref"] = show["judge_a_ref"].map({True: "통과", False: "실패"})
    show = show.rename(columns={**LABELS, "name": "이름", "days": "일수", "total_return": "수익", "mdd": "MDD",
                                "calmar": "칼마", "hold_return": "들고 있기 수익", "hold_mdd": "들고 있기 MDD", "trades": "포지션 변경일", "fee_paid": "수수료 합계",
                                "hit_rate": "방향 적중", "critical_error": "정반대 오류", "accuracy": "3분류 정확도",
                                "judge_a_ref": "(참고) 기준 A"})
    lines = [
        "> **탐색 결과**: 검증 구간을 이미 본 뒤 조건을 바꿔 다시 계산한 숫자다. 공식 판정(기준 A·B)을 바꾸지 않는다.",
        f"> **다중 비교 경고**: 조건 {len(table)}개를 같은 구간에서 비교했다. 가장 좋은 줄은 운이 섞여 실제보다 좋아 보인다. "
        "고른 조건은 실시간 성적표(②안)처럼 앞으로의 날로 다시 확인해야 한다.",
        "", f"- 구간: {table['start'].min()} ~ {table['end'].max()}",
        *[f"- 고정 조건 · {k}: {v}" for k, v in fixed.items()],
        *([f"- 메모: {note}"] if note else []),
        "- 방향 적중·정확도는 그 조건의 정답(t 종가 → t+N 종가)으로 잰다. t+1 시가 체결이면 실제 보유 구간과 하루 어긋난다",
        "- 자산 곡선과 들고 있기는 체결 가격끼리의 변화다. t+1 시가 체결 줄의 수익·MDD 는 시가끼리 계산했고 곡선의 t 행은 t+1 시가 시점 자산이다",
        *([f"- 뉴스 유출 테스트는 {config.CRITERION_B_START}~{config.CRITERION_B_END}(기준 B 구간)에서만 했다. 그 밖 구간의 뉴스 점수는 유출 검증 전이다"]
          if (table["news_mode"] != "none").any() else []),
        "", show.to_markdown(index=False),
    ]
    return "\n".join(lines) + "\n"


# ---- 조건 모니터링: 설정을 하나씩 바꾼 민감도 분석, 기간별 추적, 해석 리포트 ----

AXIS_NOTES = {  # 조건마다 무엇을 뜻하고 바꾸면 보통 어떤 득실이 생기는지
    "threshold": "상승·하락으로 볼 최소 움직임이다. 낮추면 '보합' 이 줄고 방향 신호가 늘어 매매가 잦아지고, 높이면 큰 움직임만 노려 신호가 드물어진다. "
                 "정답 자체가 바뀌므로 방향 적중률은 같은 기준끼리만 비교한다.",
    "horizon_days": "몇 거래일 뒤의 방향을 맞히는지다. 짧으면 단기 잡음이 커지고 신호가 자주 바뀌며, 길면 추세를 보지만 방향 전환을 늦게 따라간다. "
                    "정답이 바뀌므로 방향 적중률은 같은 기간끼리만 비교한다.",
    "retrain_freq": "모델을 다시 학습하는 간격이다. 자주 하면 최근 흐름을 빨리 반영하지만 잡음에 흔들리고, 드물게 하면 안정적이지만 국면 변화에 늦다.",
    "execution": "t 종가 체결은 15:15 신호 뒤 장 마감 동시호가에 사는 가정, t+1 시가는 다음 날 아침에 사는 가정이다. 차이는 밤사이 해외 시장과 장 시작 갭의 몫이다.",
    "down_action": "하락 신호 때 현금으로 나갈지 인버스 ETF 를 살지다. 인버스는 하락이 맞으면 벌지만 틀리면 오르는 장에서 반대로 잃는다.",
    "feature_set": "기존 45개 지표와 사전 등록한 후보 77개 지표(A+B+D)다. 지표가 많으면 표현력은 늘지만 과적합 위험도 커진다.",
    "news_mode": "뉴스 보정을 끄면(숫자 모델만) 뉴스가 실제로 더해 준 몫을 볼 수 있다.",
    "news_min_score": "뉴스 합의가 이 값 이상(이하)일 때만 신호를 한 단계 올린다(내린다). 낮추면 보정이 잦아지고 뉴스 소음도 더 들어온다.",
    "news_min_conf": "중재자 확신도가 이 값 이상일 때만 보정한다. 높이면 확신이 큰 날만 보정한다.",
    "news_window": "최근 k개 채점일 점수의 평균으로 보정한다. 늘리면 하루짜리 소음이 줄지만 새 뉴스에 늦게 반응한다.",
    "news_field": "종합 대신 분류별 점수(삼성 직접·반도체·거시)만으로 보정한다. 어떤 뉴스가 주가와 더 맞았는지 본다. 채점 버전 2 에서만 계산된다.",
    "news_risk_veto": "하락 위험 점수가 이 값 이상이면 상승 신호를 보합으로 낮춘다. 큰 하락을 피하려는 방어 장치다. 채점 버전 2 에서만 계산된다.",
}
TOL = config.MONITOR_TOL  # 수익·MDD 차이를 '거의 같다' 로 보는 폭


def value_label(key: str, value) -> str:
    """조건 값의 한국어 표시."""
    names = {**config.RETRAIN_FREQS, **config.EXECUTIONS, **config.DOWN_ACTION_LABELS, **config.NEWS_SCORE_FIELDS,
             "none": "숫자 모델만", "adjust": "뉴스 보정", "base": "기존 지표", "candidate": "후보 지표"}
    if value is None:
        return "없음"
    if key == "threshold":
        return f"±{value:.0%}" if value * 100 == int(value * 100) else f"±{value:.1%}"
    return names.get(value, str(value))


def common_period(cfgs: list[dict], data: Data) -> tuple[str, str]:
    """조건 목록 전체를 같은 날들로 비교하는 기본 구간. 다음 날 시가 체결이 섞이면 시가가 없는 마지막 날도 함께 뺀다."""
    start, end = default_period(cfgs, data)
    if any(c["execution"] == "next_open" for c in cfgs):
        has_next_open = data.samsung["Open"].shift(-1).dropna().index
        end = str(min(pd.Timestamp(end), has_next_open.max()).date())
    return start, end


def sensitivity(base: dict, axes: dict[str, list], data: Data) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """기준 조건에서 조건을 하나씩만 바꿔 같은 구간에서 돌린다. 계산할 수 없는 조건은 이유를 남기고 넘어간다."""
    cases = [("기준", None, None, dict(base))]
    cases += [(f"{LABELS[k]} {value_label(k, v)}", k, v, {**base, k: v}) for k, values in axes.items() for v in values if v != base[k]]
    start, end = common_period([c for *_, c in cases], data)
    rows, curves = [], {}
    for name, axis, value, cfg in cases:
        cfg = {**cfg, "start": cfg["start"] or start, "end": cfg["end"] or end}
        try:
            r = run_one(cfg, data)
        except ValueError as e:
            rows.append({"name": name, "axis": axis, "value": value, "error": str(e)})
            continue
        curves[name] = r.pop("equity")
        curves.setdefault(f"들고 있기({config.EXECUTIONS[cfg['execution']]})", r.pop("hold_equity"))
        rows.append({"name": name, "axis": axis, "value": value, "error": None, **r})
    return pd.DataFrame(rows), curves


def monthly_returns(curves: dict[str, pd.Series]) -> pd.DataFrame:
    """자산 곡선마다 달별 수익률. 첫 달은 시작 자산 1.0 대비다."""
    out = {}
    for name, eq in curves.items():
        month_end = eq.resample("ME").last()
        out[name] = month_end / month_end.shift(1).fillna(1.0) - 1
    return pd.DataFrame(out)


def recent_stats(curve: pd.Series, days: int) -> tuple[float, float]:
    """마지막 days 거래일의 수익률과 MDD(그 구간 시작 자산을 1.0 으로 본다)."""
    part = curve.iloc[-(days + 1):]
    part = part / part.iloc[0]
    return float(part.iloc[-1] - 1), bt.max_drawdown(part.iloc[1:])


def change_kind(d_ret: float, d_mdd: float) -> str:
    """기준 대비 수익 차이와 MDD 크기 차이(+ 면 더 깊게 빠짐)를 한 마디로."""
    up, down = d_ret > TOL, d_ret < -TOL
    safer, riskier = d_mdd < -TOL, d_mdd > TOL
    if up and safer:
        return "수익과 MDD 가 모두 좋아졌다"
    if up and riskier:
        return "수익은 늘었지만 MDD 도 깊어졌다(위험을 더 지고 얻은 수익)"
    if down and safer:
        return "수익을 내주고 MDD 를 줄였다(방어적)"
    if down and riskier:
        return "수익과 MDD 가 모두 나빠졌다"
    if up:
        return "MDD 는 비슷하고 수익이 늘었다"
    if down:
        return "MDD 는 비슷하고 수익이 줄었다"
    if safer:
        return "수익은 비슷하고 MDD 가 줄었다"
    if riskier:
        return "수익은 비슷하고 MDD 가 깊어졌다"
    return "기준과 거의 같다"


def interpret_row(r: pd.Series, b: pd.Series, monthly: pd.DataFrame) -> str:
    """조건 한 줄을 기준과 비교해 해석한다."""
    d_ret, d_mdd = r["total_return"] - b["total_return"], abs(r["mdd"]) - abs(b["mdd"])
    both = monthly[[r["name"], "기준"]].dropna()
    gap = both[r["name"]] - both["기준"]
    win, lose = int((gap > config.MONITOR_MONTH_TOL).sum()), int((gap < -config.MONITOR_MONTH_TOL).sum())
    same = len(both) - win - lose
    if win + lose == 0:
        spread = "모든 달이 기준과 같았다"
    elif win >= config.MONITOR_STEADY_RATIO * max(lose, 1) and win >= config.MONITOR_STEADY_MIN:
        spread = "나은 달이 꾸준히 많았다"
    elif lose >= config.MONITOR_STEADY_RATIO * max(win, 1) and lose >= config.MONITOR_STEADY_MIN:
        spread = "못한 달이 꾸준히 많았다"
    else:
        best = gap.abs().idxmax()
        spread = f"달마다 엇갈렸고 차이가 가장 큰 달은 {best:%Y-%m}({gap[best] * 100:+.1f}%p)이다"
    text = (f"수익 {r['total_return']:+.1%}(기준 대비 {d_ret * 100:+.1f}%p), MDD {r['mdd']:.1%}(기준 대비 {-d_mdd * 100:+.1f}%p) → **{change_kind(d_ret, d_mdd)}**. "
            f"달별로는 나은 달 {win} · 못한 달 {lose} · 같은 달 {same}개월로 {spread}. "
            f"포지션 변경 {int(r['trades'])}회(기준 {int(b['trades'])}회), 수수료 합계 {r['fee_paid']:.1%}.")
    if pd.notna(r.get("news_adjusted")) and r["news_mode"] == "adjust":
        text += f" 뉴스로 신호를 바꾼 날 {int(r['news_adjusted'])}일(기준 {int(b['news_adjusted'])}일)."
        if r["news_adjusted"] == 0:
            text += " 보정이 한 번도 일어나지 않아 사실상 숫자 모델만 쓴 것과 같다."
    if r["axis"] not in ("threshold", "horizon_days") and pd.notna(r["hit_rate"]) and pd.notna(b["hit_rate"]):
        text += f" 방향 적중 {r['hit_rate']:.1%}(기준 {b['hit_rate']:.1%})."
    elif pd.notna(r["hit_rate"]):
        text += f" 방향 적중 {r['hit_rate']:.1%}(정답이 달라 기준과 직접 비교하지 않는다)."
    return text


def axis_trend(rows: pd.DataFrame, base_row: pd.Series, axis: str, base_cfg: dict) -> str:
    """숫자 조건이면 값을 키울수록 수익·MDD 가 어떻게 움직였는지 한 문장으로."""
    part = pd.concat([rows[rows["axis"] == axis], base_row.to_frame().T.assign(value=base_cfg[axis])])
    part = part[part["error"].isna()] if "error" in part else part
    if len(part) < 3 or not all(isinstance(v, (int, float)) and v is not None for v in part["value"]):
        return ""
    part = part.sort_values("value")
    ret, mdd = part["total_return"].astype(float), part["mdd"].abs().astype(float)
    def direction(s):
        """값이 커질수록 모두 늘었는지·줄었는지."""
        d = s.diff().dropna()
        return "늘었다" if (d > 0).all() else "줄었다" if (d < 0).all() else "한 방향이 아니었다"
    return f"값을 키울수록 수익은 {direction(ret)}, MDD 깊이는 {direction(mdd)}."


def group_section(title: str, table: pd.DataFrame, curves: dict[str, pd.Series], note: str, base_cfg: dict) -> list[str]:
    """한 묶음(숫자 조건 또는 뉴스 조건)의 리포트 본문."""
    ok = table[table["error"].isna()]
    if ok.empty or "기준" not in set(ok["name"]):
        return [f"## {title}", "", "계산할 수 있는 조건이 없다.", ""]
    b = ok[ok["name"] == "기준"].iloc[0]
    monthly = monthly_returns(curves)
    lines = [f"## {title}", "", note, "",
             f"- 구간: {b['start']} ~ {b['end']} ({int(b['days'])}거래일)",
             f"- 기준 조건(지금 운영 설정): 수익 {b['total_return']:+.1%}, MDD {b['mdd']:.1%}, 칼마 {b['calmar']:.2f}, "
             f"포지션 변경 {int(b['trades'])}회 · 들고 있기 {b['hold_return']:+.1%} / MDD {b['hold_mdd']:.1%}", ""]
    view = ok[["name", "total_return", "mdd", "calmar", "trades", "fee_paid", "hit_rate"]].copy()
    view["trades"] = view["trades"].astype(int)
    for days in [d for d in config.MONITOR_RECENT_DAYS if d < b["days"]]:  # 구간보다 긴 최근 구간은 전체와 같아 뺀다
        stats = {n: recent_stats(curves[n], days) for n in view["name"]}
        view[f"최근 {days}일"] = [f"{stats[n][0]:+.1%} / {stats[n][1]:.1%}" for n in view["name"]]
    for col, f in [("total_return", "{:+.1%}"), ("mdd", "{:.1%}"), ("calmar", "{:.2f}"), ("fee_paid", "{:.1%}"), ("hit_rate", "{:.1%}")]:
        view[col] = view[col].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    view = view.rename(columns={"name": "조건", "total_return": "수익", "mdd": "MDD", "calmar": "칼마", "trades": "포지션 변경",
                                "fee_paid": "수수료", "hit_rate": "방향 적중"})
    lines += ["### 결과 표 (최근 N일 칸은 수익 / MDD)", "", view.to_markdown(index=False), ""]
    lines += ["### 달별 수익", "", monthly.rename(index=lambda d: d.strftime("%Y-%m")).map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
              .rename_axis("월").reset_index().to_markdown(index=False), ""]
    lines += ["### 조건별 해석", ""]
    for axis in dict.fromkeys(table["axis"].dropna()):
        lines += [f"#### {LABELS[axis]} (기준 {value_label(axis, base_cfg[axis])})", "", AXIS_NOTES.get(axis, ""), ""]
        for _, r in table[table["axis"] == axis].iterrows():
            lines.append(f"- **{r['name']}**: " + (f"계산하지 못했다 — {r['error']}" if pd.notna(r["error"]) else interpret_row(r, b, monthly)))
        trend = axis_trend(ok, b, axis, base_cfg)
        lines += ([f"- 추세: {trend}"] if trend else []) + [""]
    better = ok[(ok["name"] != "기준") & (ok["total_return"] - b["total_return"] > TOL) & (ok["mdd"].abs() - abs(b["mdd"]) < -TOL)]
    lines += ["### 요약", "",
              f"- 수익과 MDD 가 모두 기준보다 좋았던 조건: {', '.join(better['name']) if len(better) else '없음'}",
              f"- 칼마(수익 ÷ MDD)가 가장 높은 조건: {ok.loc[ok['calmar'].astype(float).idxmax(), 'name']}",
              f"- 들고 있기({b['hold_return']:+.1%})보다 수익이 높은 조건: {', '.join(ok[ok['total_return'] > ok['hold_return']]['name']) or '없음'}", ""]
    return lines


def monitor(data: Data | None = None, save: bool = True) -> dict:
    """사전 등록한 조건 목록(config.MONITOR_AXES·MONITOR_NEWS_AXES)으로 민감도 분석을 돌려 해석 리포트와 이력을 남긴다.

    같은 데이터면 숫자가 같다. 저장 데이터·뉴스 점수가 늘어나면 다시 돌릴 때 숫자가 바뀌고, 리포트에 지난 실행과의 차이가 나온다.
    앞으로의 실제 성과는 실시간 성적표(scoreboard.json)에서 가져와 함께 싣는다.
    """
    data = data or Data()
    run_at = pd.Timestamp.now(tz=config.KST).strftime("%Y-%m-%d %H:%M")
    numeric, numeric_curves = sensitivity(dict(config.EXPERIMENT_BASE), config.MONITOR_AXES, data)
    groups = [("숫자 조건 (검증 구간)", numeric, numeric_curves,
               "숫자 모델의 정답 정의·학습·매매 방식을 하나씩 바꿨다. 나머지 조건은 기준과 같다.", dict(config.EXPERIMENT_BASE))]
    if data.news_range() is not None:
        news_base = {**config.EXPERIMENT_BASE, "news_mode": "adjust"}
        news, news_curves = sensitivity(news_base, config.MONITOR_NEWS_AXES, data)
        versions = sorted(data.scores["scoring_version"].unique()) if "scoring_version" in data.scores else [1]
        groups.append(("뉴스 조건 (뉴스 채점 구간)", news, news_curves,
                       f"뉴스 보정 방법을 하나씩 바꿨다. 기준은 지금 기준 B 규칙(합의 ±{config.NEWS_ADJUST_SCORE}, 확신도 {config.NEWS_ADJUST_MIN_CONF})이다. "
                       f"채점 버전 {versions}, 채점일 {len(data.scores)}일. 뉴스 유출 테스트는 {config.CRITERION_B_START}~{config.CRITERION_B_END}에서만 했다.", news_base))
    all_rows = pd.concat([t.assign(group=g) for g, t, *_ in groups], ignore_index=True)
    history_path = Path(config.MONITOR_HISTORY_PATH)
    previous = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    snapshot = all_rows[all_rows["error"].isna()][["group", "name", "start", "end", "days", "total_return", "mdd", "calmar"]].assign(run_at=run_at)
    lines = ["# 조건 모니터링 리포트", "", f"계산 시각 {run_at} (KST) · 저장 데이터 끝 {config.DATA_END}", "",
             "> **탐색 결과**: 검증 구간을 이미 본 뒤 조건을 하나씩 바꿔 다시 계산한 숫자다. 공식 판정(기준 A·B)과 운영 설정을 바꾸지 않는다.",
             f"> **다중 비교 경고**: 조건 {len(all_rows)}개를 같은 구간에서 비교했다. 가장 좋아 보이는 조건에는 운이 섞여 있다. "
             "운영 설정을 바꾸려면 아래 '앞으로의 성과' 처럼 새로 쌓이는 날들로 다시 확인한다.", "",
             "읽는 법: 수익은 구간 누적 수익률, MDD 는 고점 대비 가장 크게 빠진 비율, 칼마는 연 환산 수익 ÷ |MDD| 다. "
             "기준 대비 %p 는 두 수익률(또는 MDD)의 차이다. 'MDD 기준 대비 +' 는 덜 빠졌다는 뜻이다. "
             f"차이가 ±{TOL * 100:.1f}%p 안이면 '비슷하다' 로 본다.", ""]
    lines += ["## 한눈에 보기", ""] + [f"- {x}" for x in highlights(groups)] + [""]
    for title, table, curves, note, base_cfg in groups:
        lines += group_section(title, table, curves, note, base_cfg)
    lines += ["## 지난 실행과 비교", ""]
    if previous.empty:
        lines += ["첫 실행이라 비교할 이력이 없다.", ""]
    else:
        last = previous[previous["run_at"] == previous["run_at"].max()]
        merged = snapshot.merge(last, on=["group", "name"], suffixes=("", "_prev"))
        changed = merged[(merged["total_return"] - merged["total_return_prev"]).abs() > 1e-9]
        lines += [f"지난 실행 {last['run_at'].iloc[0]} 대비: " + ("데이터가 같아 숫자가 같다." if changed.empty else f"{len(changed)}개 조건의 숫자가 바뀌었다."), ""]
        for _, r in changed.iterrows():
            lines.append(f"- {r['group']} · {r['name']}: 수익 {r['total_return_prev']:+.1%} → {r['total_return']:+.1%}, "
                         f"MDD {r['mdd_prev']:.1%} → {r['mdd']:.1%} (구간 끝 {r['end_prev']} → {r['end']})")
        lines.append("")
    lines += ["## 앞으로의 성과 (실시간 성적표)", "",
              "아래는 매일 장 마감 전에 기록한 실제 신호의 채점 결과다. 위 숫자와 달리 조건을 보고 고를 수 없는 날들이라 가장 믿을 만한 확인 수단이다.", ""]
    board_path = Path(config.SCOREBOARD_PATH)
    if board_path.exists():
        lines += [f"- {x}" for x in scoreboard.summary_lines(json.loads(board_path.read_text(encoding="utf-8")))]
    else:
        lines.append("- 아직 성적표가 없다.")
    report = "\n".join(lines) + "\n"
    if save:
        Path(config.MONITOR_REPORT_PATH).write_text(report, encoding="utf-8", newline="\n")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([previous, snapshot], ignore_index=True).to_csv(history_path, index=False, encoding="utf-8-sig")
    return {"report": report, "tables": {g: t for g, t, *_ in groups}, "curves": {g: c for g, _, c, *_ in groups},
            "summary": monitor_summary(groups)}


def highlights(groups) -> list[str]:
    """리포트 맨 앞 요약: 묶음마다 영향이 컸던 조건과 들고 있기 대비 위치."""
    out = []
    for title, table, *_ in groups:
        ok = table[table["error"].isna()]
        if "기준" not in set(ok["name"]):
            continue
        b = ok[ok["name"] == "기준"].iloc[0]
        others = ok[ok["name"] != "기준"].assign(d_ret=lambda t: t["total_return"] - b["total_return"],
                                                 d_mdd=lambda t: t["mdd"].abs() - abs(b["mdd"]))
        out.append(f"**{title}** 기준 {b['total_return']:+.1%} / MDD {b['mdd']:.1%}, 들고 있기 {b['hold_return']:+.1%} / MDD {b['hold_mdd']:.1%}.")
        big = others.reindex(others["d_ret"].abs().sort_values(ascending=False).index).head(3)
        out.append("영향이 가장 큰 조건: " + "; ".join(f"{r['name']} {r['d_ret'] * 100:+.1f}%p({change_kind(r['d_ret'], r['d_mdd'])})"
                                                for _, r in big.iterrows()))
        better = others[(others["d_ret"] > TOL) & (others["d_mdd"] < -TOL)]
        out.append("수익·MDD 가 모두 나았던 조건: " + (", ".join(better["name"]) if len(better) else "없음")
                   + ". 한 조건만 바꾼 결과이고 같은 구간을 여러 번 본 탐색이라, 운영 설정 변경 근거로 쓰려면 앞으로의 날로 다시 확인해야 한다.")
        if (ok["total_return"] < ok["hold_return"]).all():
            margin = config.MONITOR_MDD_MARGIN
            safer = ok[ok["mdd"].abs() < abs(b["hold_mdd"]) - margin]
            market = ("들고 있기가 크게 오른 구간이라 현금으로 쉬는 날이 있는 전략이 불리했다. " if b["hold_return"] > 0
                      else "들고 있기도 손실이었지만 조건들이 그보다 못했다. ")
            out.append("모든 조건이 들고 있기보다 수익이 낮았다. " + market +
                       f"MDD 를 들고 있기보다 크게 줄였는지도 함께 본다. 들고 있기보다 MDD 가 {margin * 100:.0f}%p 넘게 얕은 조건: "
                       + (", ".join(f"{r['name']}({r['mdd']:.1%})" for _, r in safer.iterrows()) or "없음") + ".")
        n_err = int(table["error"].notna().sum())
        if n_err:
            out.append(f"계산하지 못한 조건 {n_err}개(뉴스 분류별 점수·하락 위험은 채점 버전 2 재채점 뒤에 계산된다).")
    return out


def monitor_summary(groups) -> list[str]:
    """알림 메일용 짧은 요약: 묶음마다 기준 성과와 수익·MDD 가 모두 좋았던 조건."""
    out = []
    for title, table, *_ in groups:
        ok = table[table["error"].isna()]
        if "기준" not in set(ok["name"]):
            continue
        b = ok[ok["name"] == "기준"].iloc[0]
        better = ok[(ok["name"] != "기준") & (ok["total_return"] - b["total_return"] > TOL) & (ok["mdd"].abs() - abs(b["mdd"]) < -TOL)]
        out.append(f"{title}: 기준 {b['total_return']:+.1%} / MDD {b['mdd']:.1%} · 둘 다 나은 조건 "
                   + (", ".join(f"{r['name']}({r['total_return']:+.1%}/{r['mdd']:.1%})" for _, r in better.iterrows()) or "없음"))
    return out


def parse_value(key: str, text: str):
    """명령줄 값을 조건 형식으로 바꾼다."""
    text = text.strip()
    base = config.EXPERIMENT_BASE[key]
    if text.lower() in ("none", ""):
        return None if key in ("news_risk_veto", "start", "end") else base  # 빈 값은 기본 조건으로
    if key in ("horizon_days", "news_window", "news_risk_veto"):
        return int(text)
    if key in ("threshold", "news_min_score", "news_min_conf"):
        return float(text)
    return type(base)(text) if base is not None else text


def main(argv=None):
    """예: python -m src.pipeline.experiment --vary horizon_days=3,5,10 --vary down_action=cash,inverse"""
    parser = argparse.ArgumentParser(description="조건 비교 실험 (탐색)")
    parser.add_argument("--set", action="append", default=[], help="기준 조건 고정값. 예: --set feature_set=candidate")
    parser.add_argument("--vary", action="append", default=[], help="바꿔 볼 값들. 예: --vary threshold=0.02,0.03")
    parser.add_argument("--name", default="", help="보고서 파일 이름에 붙일 짧은 이름")
    parser.add_argument("--note", default="", help="보고서에 남길 메모")
    parser.add_argument("--monitor", action="store_true", help="사전 등록한 조건으로 모니터링 리포트를 만든다")
    args = parser.parse_args(argv)
    if args.monitor:
        result = monitor()
        print("\n".join(result["summary"]))
        print("저장:", config.MONITOR_REPORT_PATH, config.MONITOR_HISTORY_PATH)
        return
    base = dict(config.EXPERIMENT_BASE)
    for item in args.set:
        key, value = item.split("=", 1)
        base[key] = parse_value(key, value)
    vary = {}
    for item in args.vary:
        key, values = item.split("=", 1)
        vary[key] = [parse_value(key, v) for v in values.split(",")]
    table, _ = compare(expand(base, vary))
    stamp = pd.Timestamp.now(tz=config.KST).strftime("%Y%m%d_%H%M")
    out_dir = Path(config.EXPERIMENT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"experiments_{stamp}" + (f"_{args.name}" if args.name else "")
    table.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    report = Path(config.EXPERIMENT_REPORT_DIR) / f"{stem}.md"
    report.write_text(f"# 조건 비교 실험 {stamp}\n\n" + to_markdown(table, args.note), encoding="utf-8", newline="\n")
    print(to_markdown(table, args.note))
    print("저장:", out_dir / f"{stem}.csv", report)


if __name__ == "__main__":
    main()
