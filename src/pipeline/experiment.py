"""조건 비교 실험: 설정값을 바꿔 가며 지금까지 모은 데이터로 학습·모의 매매를 다시 돌려 한 표로 비교한다.

바꿀 수 있는 조건(config.EXPERIMENT_BASE 의 열쇠):
- horizon_days, threshold: 정답 정의(t+N거래일 ±기준). 바꾸면 모델을 그 정답으로 다시 학습한다
- feature_set: 지표 묶음(base·candidate, 또는 지표 그룹을 고른 'groups:price+chart' 형식)
- retrain_freq: 재학습 주기(W·M·Q)
- execution: 체결 시점(t 종가 · t+1 시가)
- down_action: 하락 신호 때 행동(현금·인버스)
- position_steps, position_floor: 분할 매매(신호마다 1/단계씩 비중 조절, 하락 때도 남길 기본 비중). 현금 규칙에서만
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
    "execution": "체결 시점", "down_action": "하락 행동", "position_steps": "분할 단계", "position_floor": "기본 보유 비중", "news_mode": "뉴스 사용", "news_field": "뉴스 점수",
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
        if key == "feature_set" and str(cfg[key]).startswith(config.FEATURE_GROUP_PREFIX):
            train.feature_set_columns(cfg[key])  # 그룹 묶음은 이름만 검사한다(모르는 그룹이면 ValueError)
            continue
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
    split = is_split(cfg)
    if split and cfg["down_action"] != "cash":
        raise ValueError("분할 매매는 하락 행동이 현금일 때만 계산한다")
    sim = (bt.simulate_split(sig, samsung_px, cfg["position_steps"], cfg["position_floor"]) if split
           else bt.simulate(sig, samsung_px, inverse_px, cfg["down_action"]))
    hold = bt.buy_and_hold(samsung_px, dates)
    labels = data.labels(cfg["horizon_days"], cfg["threshold"])["label"]
    m, h = bt.summary(sim["equity"]), bt.summary(hold)
    direction = bt.direction_hit_rate(sig, labels)
    has_label = labels.reindex(dates).notna()
    known = labels.reindex(dates)[has_label]
    share = sig.value_counts(normalize=True)

    def precision(label: str) -> float:
        """그 신호를 낸 날 중 정답도 그 방향이었던 비율."""
        hit = sig[has_label] == label
        return float((known[hit] == label).mean()) if hit.any() else float("nan")
    return {
        **cfg, "start": str(dates.min().date()), "end": str(dates.max().date()), "days": len(dates),
        "total_return": m["total_return"], "mdd": m["mdd"], "calmar": m["calmar"],
        "hold_return": h["total_return"], "hold_mdd": h["mdd"],
        "trades": int((sim["fee"] > 0).sum()), "fee_paid": float(1 - (1 - sim["fee"]).prod()),  # trades = 포지션을 바꾼 날 수
        "hit_rate": direction["hit_rate"], "critical_error": direction["critical_error_rate"],
        "accuracy": float((sig[has_label] == known).mean()) if has_label.any() else float("nan"),
        "always_up_accuracy": float((known == UP).mean()) if has_label.any() else float("nan"),  # 매일 '상승' 이라고만 했을 때의 정확도
        "up_precision": precision(UP), "down_precision": precision(DOWN),
        "actual_down_share": float((known == DOWN).mean()) if has_label.any() else float("nan"),
        "n_features": len(train.feature_set_columns(cfg["feature_set"])),
        "first_train_rows": int(numeric["train_rows"].iloc[0]) if "train_rows" in numeric else None,  # 묶음마다 결측이 달라 첫 학습 행 수가 다를 수 있다
        "up_share": float(share.get(UP, 0)), "down_share": float(share.get(DOWN, 0)),
        "judge_a_ref": bt.judge_a(sim["equity"], hold)["passed"],
        "news_adjusted": int(signals.loc[dates, "adjusted"].sum()) if "adjusted" in signals else None,  # 뉴스로 신호를 바꾼 날 수
        "equity": sim["equity"], "hold_equity": hold,
    }


def is_split(cfg: dict) -> bool:
    """분할 매매 조건인지(1단계·바닥 0 이 아니면 분할)."""
    return cfg["position_steps"] != 1 or cfg["position_floor"] > 0


def compare(cfgs: list[dict], data: Data | None = None) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """여러 조건을 같은 구간에서 돌려 결과 표와 자산 곡선(조건 이름 → 곡선)을 돌려준다."""
    data = data or Data()
    skipped = [c for c in cfgs if is_split(c) and c["down_action"] != "cash"]  # 분할 매매는 현금 규칙에서만 계산한다
    cfgs = [c for c in cfgs if c not in skipped]
    if not cfgs:
        raise ValueError("분할 매매는 하락 행동이 현금일 때만 계산한다")
    start, end = common_period(cfgs, data)
    rows, curves = [], {}
    for i, cfg in enumerate(cfgs, 1):
        cfg = {**cfg, "start": cfg["start"] or start, "end": cfg["end"] or end}
        r = run_one(cfg, data)
        name = f"조건 {i}"
        curves[name] = r.pop("equity")
        curves.setdefault(f"들고 있기({config.EXECUTIONS[cfg['execution']]})", r.pop("hold_equity"))
        rows.append({"name": name, **r})
    table = pd.DataFrame(rows)
    table.attrs["skipped"] = len(skipped)
    return table, curves


def varied_columns(table: pd.DataFrame) -> list[str]:
    """조건 중 실제로 값이 달라진 열."""
    return [k for k in config.EXPERIMENT_BASE if k in table and table[k].astype(str).nunique() > 1]


def to_markdown(table: pd.DataFrame, note: str = "") -> str:
    """결과 표를 보고서용 마크다운으로 만든다. 탐색 결과·다중 비교 경고를 붙인다."""
    varied = varied_columns(table)
    fixed = {LABELS[k]: table[k].iloc[0] for k in config.EXPERIMENT_BASE if k not in varied and k in table}
    show = table[["name", *varied, "days", "total_return", "mdd", "calmar", "hold_return", "hold_mdd", "trades",
                  "fee_paid", "hit_rate", "critical_error", "accuracy", "always_up_accuracy", "down_precision", "judge_a_ref"]].copy()
    for col in ["total_return", "mdd", "hold_return", "hold_mdd", "fee_paid", "hit_rate", "critical_error", "accuracy",
                "always_up_accuracy", "down_precision"]:
        show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}" if col in ("total_return", "hold_return") else f"{v:.1%}")
    show["calmar"] = show["calmar"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    show["judge_a_ref"] = show["judge_a_ref"].map({True: "통과", False: "실패"})
    show = show.rename(columns={**LABELS, "name": "이름", "days": "일수", "total_return": "수익", "mdd": "MDD",
                                "calmar": "칼마", "hold_return": "들고 있기 수익", "hold_mdd": "들고 있기 MDD", "trades": "포지션 변경일", "fee_paid": "수수료 합계",
                                "hit_rate": "방향 적중", "critical_error": "정반대 오류", "accuracy": "3분류 정확도",
                                "always_up_accuracy": "항상 상승 정확도", "down_precision": "하락 신호 적중",
                                "judge_a_ref": "(참고) 기준 A"})
    if LABELS["feature_set"] in show:
        show[LABELS["feature_set"]] = show[LABELS["feature_set"]].map(lambda v: value_label("feature_set", v))
    lines = [
        "> **탐색 결과**: 검증 구간을 이미 본 뒤 조건을 바꿔 다시 계산한 숫자다. 공식 판정(기준 A·B)을 바꾸지 않는다.",
        f"> **다중 비교 경고**: 조건 {len(table)}개를 같은 구간에서 비교했다. 가장 좋은 줄은 운이 섞여 실제보다 좋아 보인다. "
        "고른 조건은 실시간 성적표(②안)처럼 앞으로의 날로 다시 확인해야 한다.",
        "", f"- 구간: {table['start'].min()} ~ {table['end'].max()}",
        *[f"- 고정 조건 · {k}: {v}" for k, v in fixed.items()],
        *([f"- 메모: {note}"] if note else []),
        *([f"- 하락 행동이 인버스인 분할 매매 조합 {table.attrs['skipped']}개는 계산하지 않았다(분할 매매는 현금 규칙에서만)"]
          if table.attrs.get("skipped") else []),
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
    "position_steps": "신호 한 번에 비중을 얼마나 바꿀지다. 단계를 늘리면 틀린 신호 하나의 손실이 줄지만 맞는 신호의 수익도 늦게 따라간다.",
    "position_floor": "하락 신호가 이어져도 남겨 두는 삼성전자 비중이다. 높이면 들고 있기에 가까워져 오르는 장에서 유리하고, 하락 신호가 맞을 때 덜 피한다.",
    "down_action": "하락 신호 때 현금으로 나갈지 인버스 ETF 를 살지다. 인버스는 하락이 맞으면 벌지만 틀리면 오르는 장에서 반대로 잃는다.",
    "feature_set": f"운영 {len(config.FEATURE_SET_BASE)}개 지표(1~3일차 지표 + RSI), 1~3일차 {len(config.FEATURE_SET_BASE_V1)}개, 사전 등록한 후보 {len(config.FEATURE_SETS['candidate'])}개 지표(A+B+D), 또는 지표 그룹을 골라 만든 묶음이다. 지표가 많으면 표현력은 늘지만 과적합 위험도 커진다. "
                   "그룹 단위 진단은 experiments_features.md 에 따로 있다.",
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
             "none": "숫자 모델만", "adjust": "뉴스 보정", "base": f"운영 지표({len(config.FEATURE_SET_BASE)}개)", "base_v1": f"1~3일차 지표({len(config.FEATURE_SET_BASE_V1)}개)", "candidate": "후보 지표"}
    if value is None:
        return "없음"
    if key == "feature_set" and str(value).startswith(config.FEATURE_GROUP_PREFIX):
        groups = value[len(config.FEATURE_GROUP_PREFIX):].split("+")
        return "그룹: " + " + ".join(config.FEATURE_GROUP_LABELS.get(g, g) for g in groups)
    if key == "position_steps":
        return "전량" if value == 1 else f"{value}단계"
    if key == "position_floor":
        return f"{value:.0%}"
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
             f"포지션 변경 {int(b['trades'])}회 · 들고 있기 {b['hold_return']:+.1%} / MDD {b['hold_mdd']:.1%}",
             f"- 3분류 정확도 {b['accuracy']:.1%} · 매일 '상승' 이라고만 했을 때 {b['always_up_accuracy']:.1%} · "
             "하락 신호 적중 " + ("없음" if pd.isna(b["down_precision"]) else f"{b['down_precision']:.1%}")
             + f"(실제 하락 비율 {b['actual_down_share']:.1%})", ""]
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


# ---- 지표 그룹 진단: 기존 그룹을 하나씩 빼고 후보 그룹을 하나씩 더해 보고, 고정 규칙으로 추천 묶음을 만든다 ----

def study_verdict(r: pd.Series, b: pd.Series) -> tuple[str, bool]:
    """기준(b) 대비 한 변경(r)의 판정과 추천 여부. 정확도·수익·MDD 중 나빠진 것 없이 하나 이상 좋아져야 추천한다."""
    d_acc = r["accuracy"] - b["accuracy"]
    d_ret = r["total_return"] - b["total_return"]
    d_mdd = abs(r["mdd"]) - abs(b["mdd"])  # + 면 더 깊게 빠졌다
    acc_tol = config.FEATURE_STUDY_ACC_TOL
    worse = [n for n, bad in [("정확도", d_acc < -acc_tol), ("수익", d_ret < -TOL), ("MDD", d_mdd > TOL)] if bad]
    better = [n for n, good in [("정확도", d_acc > acc_tol), ("수익", d_ret > TOL), ("MDD", d_mdd < -TOL)] if good]
    if worse and better:
        return f"{'·'.join(better)} 좋아짐, {'·'.join(worse)} 나빠짐", False
    if worse:
        return f"{'·'.join(worse)} 나빠짐", False
    if better:
        return f"{'·'.join(better)} 좋아짐(나빠진 것 없음) → 추천", True
    return "거의 같음", False


def feature_study(data: Data | None = None, save: bool = True) -> dict:
    """지표 그룹 진단. 기준(운영 지표 묶음)에서 기존 그룹을 하나씩 빼고, 후보 그룹을 하나씩 더해 같은 구간에서 비교한다.

    추천 규칙(study_verdict)을 통과한 변경을 모두 합친 '추천 묶음' 도 계산한다. 결과는 검증 구간을 본 뒤의 탐색이라
    운영 신호를 바꾸지 않는다. 추천 묶음은 config.LIVE_EXTRA_FEATURE_SETS 에 넣어 앞으로의 날로 다시 확인한다.
    """
    data = data or Data()
    base_cfg = dict(config.EXPERIMENT_BASE)
    base_groups = list(config.FEATURE_GROUPS_BASE)
    extras = [g for g in config.FEATURE_GROUPS if g not in base_groups]
    changes = {**{train.group_set_name([x for x in base_groups if x != g]): ("빼기", g) for g in base_groups},
               **{train.group_set_name([*base_groups, g]): ("더하기", g) for g in extras}}
    table, curves = sensitivity(base_cfg, {"feature_set": list(changes)}, data)
    ok = table[table["error"].isna()]
    if "기준" not in set(ok["name"]):
        raise ValueError("기준 조건을 계산하지 못했다: " + "; ".join(table["error"].dropna()))
    b = ok[ok["name"] == "기준"].iloc[0]
    rows, picked = [], {"빼기": [], "더하기": []}
    for _, r in ok[ok["name"] != "기준"].iterrows():
        kind, group = changes[r["value"]]
        verdict, recommend = study_verdict(r, b)
        if recommend:
            picked[kind].append(group)
        rows.append({"change": f"{kind}: {config.FEATURE_GROUP_LABELS[group]}", "feature_set": r["value"], **r[STUDY_COLS].to_dict(),
                     "verdict": verdict, "recommend": recommend})
    recommended = None
    if picked["빼기"] or picked["더하기"]:
        name = train.group_set_name([g for g in base_groups if g not in picked["빼기"]] + picked["더하기"])
        if name in changes:  # 바뀐 그룹이 하나뿐이면 위의 줄과 같은 묶음이다
            row = next(x for x in rows if x["feature_set"] == name)
        else:
            r = run_one({**base_cfg, "feature_set": name, "start": b["start"], "end": b["end"]}, data)
            curves["추천 묶음"] = r.pop("equity")
            r.pop("hold_equity")
            verdict, recommend = study_verdict(pd.Series(r), b)
            row = {"change": "추천 묶음(추천 변경을 모두 합침)", "feature_set": name, **{k: r[k] for k in STUDY_COLS},
                   "verdict": verdict, "recommend": recommend}
            rows.append(row)
        if row["recommend"]:  # 합친 묶음도 같은 규칙을 통과해야 추천한다
            recommended = {"feature_set": name, "removed": picked["빼기"], "added": picked["더하기"], **row}
    study = pd.DataFrame(rows)
    report = study_report(study, b, recommended)
    if save:
        Path(config.FEATURE_STUDY_REPORT_PATH).write_text(report, encoding="utf-8", newline="\n")
    return {"report": report, "table": study, "base": b, "curves": curves, "recommended": recommended}


STUDY_COLS = ["n_features", "first_train_rows", "accuracy", "always_up_accuracy", "up_precision", "down_precision", "total_return", "mdd", "calmar", "trades"]


def study_report(study: pd.DataFrame, b: pd.Series, recommended: dict | None) -> str:
    """지표 그룹 진단 리포트(마크다운)."""
    run_at = pd.Timestamp.now(tz=config.KST).strftime("%Y-%m-%d %H:%M")
    pct = lambda v: "" if pd.isna(v) else f"{v:.1%}"
    view = pd.concat([pd.DataFrame([{"change": f"기준(기존 {len(config.FEATURE_SET_BASE)}개)", **b[STUDY_COLS].to_dict(), "verdict": ""}]), study], ignore_index=True)
    view = pd.DataFrame({
        "변경": view["change"], "지표 수": view["n_features"].astype(int), "첫 학습 행": view["first_train_rows"].astype("Int64"), "3분류 정확도": view["accuracy"].map(pct),
        "상승 신호 적중": view["up_precision"].map(pct), "하락 신호 적중": view["down_precision"].map(pct),
        "수익": view["total_return"].map(lambda v: f"{v:+.1%}"), "MDD": view["mdd"].map(pct),
        "칼마": view["calmar"].map(lambda v: f"{v:.2f}"), "포지션 변경": view["trades"].astype(int), "판정": view["verdict"]})
    lines = ["# 지표 그룹 진단", "", f"계산 시각 {run_at} (KST) · 구간 {b['start']} ~ {b['end']} ({int(b['days'])}거래일)", "",
             "> **탐색 결과**: 검증 구간을 이미 본 뒤 지표 그룹을 바꿔 다시 학습한 숫자다. 공식 판정과 운영 신호를 바꾸지 않는다.",
             f"> **다중 비교 경고**: 묶음 {len(study)}개를 같은 구간에서 비교했다. 정답이 {config.LABEL_HORIZON_DAYS}거래일씩 겹쳐 실제로 독립인 표본은 "
             f"약 {int(b['days']) // config.LABEL_HORIZON_DAYS}개라, 정확도 몇 %p 차이는 운으로도 생긴다.", "",
             f"추천 규칙(고정): 3분류 정확도(±{config.FEATURE_STUDY_ACC_TOL:.0%}p)·수익(±{TOL * 100:.1f}%p)·MDD(±{TOL * 100:.1f}%p) 중 "
             "허용 폭 넘게 나빠진 것이 없고 하나 이상이 허용 폭 넘게 좋아진 변경만 추천한다.", "",
             "## 기준선", "",
             f"- 기준(기존 {len(config.FEATURE_SET_BASE)}개): 3분류 정확도 {b['accuracy']:.1%}, 하락 신호 적중 {pct(b['down_precision'])}, "
             f"수익 {b['total_return']:+.1%} / MDD {b['mdd']:.1%}",
             f"- 매일 '상승' 이라고만 했을 때(들고 있기): 정확도 {b['always_up_accuracy']:.1%}, 수익 {b['hold_return']:+.1%} / MDD {b['hold_mdd']:.1%}"]
    all_acc = pd.concat([study["accuracy"], pd.Series([b["accuracy"]])])
    if (all_acc < b["always_up_accuracy"]).all():
        lines.append("- **모든 묶음의 정확도가 '항상 상승' 보다 낮다.** 지표를 고르는 것만으로는 이 차이를 메우지 못한다. "
                     "보합·하락 판정이 실제보다 잦은 것이 더 큰 원인인지 함께 본다.")
    down_hits = pd.concat([study["down_precision"], pd.Series([b["down_precision"]])]).dropna()
    if len(down_hits) and (down_hits < b["actual_down_share"]).all():
        lines.append(f"- **하락 신호 적중이 모든 묶음에서 실제 하락 비율({b['actual_down_share']:.1%})보다 낮다.** 아무 날이나 하락이라고 한 것보다 못해, "
                     "지표 묶음보다 하락 신호를 내는 규칙이 문제일 가능성이 크다.")
    lines += ["", "## 결과 표", "", view.to_markdown(index=False), "", "## 추천", ""]
    if recommended is None:
        if study["recommend"].any():
            lines.append("- 개별 변경은 규칙을 통과했지만 그것들을 합친 묶음이 규칙을 통과하지 못해 추천하지 않는다.")
        lines.append(f"- 추천 규칙을 통과한 묶음이 없다. 지금 지표 묶음(기존 {len(config.FEATURE_SET_BASE)}개)을 그대로 둔다.")
    else:
        parts = [f"빼기 {', '.join(config.FEATURE_GROUP_LABELS[g] for g in recommended['removed'])}" if recommended["removed"] else "",
                 f"더하기 {', '.join(config.FEATURE_GROUP_LABELS[g] for g in recommended['added'])}" if recommended["added"] else ""]
        lines += [f"- 추천 묶음: `{recommended['feature_set']}` ({' · '.join(p for p in parts if p)}, 지표 {int(recommended['n_features'])}개)",
                  f"- 같은 구간 결과: 정확도 {recommended['accuracy']:.1%}(기준 {b['accuracy']:.1%}), 수익 {recommended['total_return']:+.1%}"
                  f"(기준 {b['total_return']:+.1%}), MDD {recommended['mdd']:.1%}(기준 {b['mdd']:.1%}) — {recommended['verdict']}",
                  "- 위 숫자는 이 묶음을 고른 구간의 결과라 선택 편향이 들어 있다. PoC 보고서·공식 판정에 옮겨 적지 않고, "
                  "추천 묶음의 성과는 추적을 시작한 뒤의 실시간 기록으로만 판단한다.",
                  "- 바로 운영 신호로 쓰지 않는다. `config.LIVE_EXTRA_FEATURE_SETS` 에 위 이름을 넣으면 매일 실시간 성적표에서 기존 모델과 나란히 채점된다. "
                  "앞으로의 날에서도 나으면 그때 운영 묶음을 바꿀지 정한다."]
    lines += ["", "## 지표 그룹", "", *[f"- `{g}` {config.FEATURE_GROUP_LABELS[g]}: {len(cols)}개" + (" (기존)" if g in config.FEATURE_GROUPS_BASE else " (후보)")
                                     for g, cols in config.FEATURE_GROUPS.items()]]
    return "\n".join(lines) + "\n"


def parse_value(key: str, text: str):
    """명령줄 값을 조건 형식으로 바꾼다."""
    text = text.strip()
    base = config.EXPERIMENT_BASE[key]
    if text.lower() in ("none", ""):
        return None if key in ("news_risk_veto", "start", "end") else base  # 빈 값은 기본 조건으로
    if key in ("horizon_days", "news_window", "news_risk_veto", "position_steps"):
        return int(text)
    if key in ("threshold", "news_min_score", "news_min_conf", "position_floor"):
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
    parser.add_argument("--features", action="store_true", help="지표 그룹 진단과 추천 리포트를 만든다")
    args = parser.parse_args(argv)
    if args.features:
        result = feature_study()
        rec = result["recommended"]
        print("추천 묶음:", rec["feature_set"] if rec else "없음(기존 묶음 유지)")
        print("저장:", config.FEATURE_STUDY_REPORT_PATH)
        return
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
