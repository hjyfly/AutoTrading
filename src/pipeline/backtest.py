from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.pipeline.labels import DOWN, FLAT, UP

CASH, LONG, INVERSE = "cash", "samsung", "inverse"
TRADING_DAYS_PER_YEAR = 252
EPS = 1e-12  # 판정 경계값 비교의 부동소수점 오차 허용


def simulate(signals: pd.Series, samsung_close: pd.Series, inverse_close: pd.Series, down_action: str | None = None) -> pd.DataFrame:
    """t일 신호대로 t일 종가에 체결하는 모의 매매. 현금으로 시작한다.

    상승 → 삼성전자 100%, 보합 → 기존 포지션 유지, 하락 → down_action(기본 config.DOWN_ACTION):
    "cash" 면 삼성전자를 팔고 현금, "inverse" 면 인버스 ETF 100%.
    포지션을 바꾸면 파는 쪽·사는 쪽마다 거래 금액의 FEE_RATE 를 뗀다. 수익은 t일 종가 → t+1일 종가로 계산한다.
    돌려주는 equity 는 각 날짜 종가 시점(그날 체결 뒤)의 자산이다.
    """
    dates = signals.index.sort_values()
    down_action = down_action or config.DOWN_ACTION
    if down_action not in config.DOWN_ACTIONS:
        raise ValueError(f"알 수 없는 하락 행동: {down_action}")
    closes = pd.DataFrame({LONG: samsung_close, INVERSE: inverse_close}).reindex(dates)
    needed = [LONG, INVERSE] if down_action == "inverse" else [LONG]  # 현금 규칙에는 인버스 가격이 필요 없다
    if closes[needed].isna().any().any():
        raise ValueError("신호 날짜에 가격이 없다")
    down_target = CASH if down_action == "cash" else INVERSE
    position, equity, rows = CASH, 1.0, []
    for i, t in enumerate(dates):
        if i > 0 and position != CASH:  # 전날 종가 → 오늘 종가 수익
            equity *= closes.loc[t, position] / closes.loc[dates[i - 1], position]
        target = {UP: LONG, DOWN: down_target}.get(signals[t], position)
        fee = 0.0
        if target != position:
            sides = (position != CASH) + (target != CASH)
            fee = 1 - (1 - config.FEE_RATE) ** sides
            equity *= 1 - fee
            position = target
        rows.append({"date": t, "signal": signals[t], "position": position, "fee": fee, "equity": equity})
    return pd.DataFrame(rows).set_index("date")


def simulate_split(signals: pd.Series, samsung_close: pd.Series, steps: int | None = None, floor: float | None = None) -> pd.DataFrame:
    """분할 매매 모의 매매. t일 신호대로 t일 종가에 체결하고 현금으로 시작한다(하락 행동은 현금만).

    목표 비중: 상승 → +1/steps(최대 1), 하락 → -1/steps(최소 floor), 보합 → 유지(첫날은 floor).
    목표 비중이 바뀐 날만 그 비중에 맞춰 사고판다. 그 사이에는 가격 변동으로 실제 비중이 움직이게 둔다.
    수수료는 사고판 금액의 FEE_RATE 다(부분 매매에서는 수수료를 떼기 전 금액 기준이라 FEE_RATE² 수준의 근사). 1단계·바닥 0 이면 simulate(down_action="cash") 와 같은 결과다.
    """
    steps = config.POSITION_STEPS if steps is None else steps
    floor = config.POSITION_FLOOR if floor is None else floor
    if steps < 1 or not 0 <= floor <= 1:
        raise ValueError(f"분할 단계는 1 이상, 바닥 비중은 0~1 이어야 한다: {steps}, {floor}")
    dates = signals.index.sort_values()
    levels, level = [], 0.0
    for t in dates:
        base = max(level, floor)  # 바닥 아래에서 시작하지 않는다(첫 상승에도 바닥 비중부터 채운다)
        level = {UP: min(1.0, base + 1 / steps), DOWN: max(floor, level - 1 / steps)}.get(signals[t], base)
        levels.append(level)
    out = simulate_target(pd.Series(levels, index=dates), samsung_close)
    out.insert(0, "signal", signals.reindex(dates))
    return out


def simulate_target(targets: pd.Series, samsung_close: pd.Series) -> pd.DataFrame:
    """목표 비중(0~1) 모의 매매. t일 목표 비중으로 t일 종가에 체결하고 현금으로 시작한다(나머지는 현금).

    목표 비중이 바뀐 날만 그 비중에 맞춰 사고판다. 그 사이에는 가격 변동으로 실제 비중이 움직이게 둔다.
    수수료는 사고판 금액의 FEE_RATE 다(부분 매매에서는 수수료를 떼기 전 금액 기준이라 FEE_RATE² 수준의 근사).
    """
    dates = targets.index.sort_values()
    closes = samsung_close.reindex(dates)
    if closes.isna().any():
        raise ValueError("신호 날짜에 가격이 없다")
    if ((targets < 0) | (targets > 1)).any():
        raise ValueError("목표 비중은 0~1 이어야 한다")
    stock, cash, level, rows = 0.0, 1.0, 0.0, []
    for i, t in enumerate(dates):
        if i > 0:
            stock *= closes[t] / closes[dates[i - 1]]
        new_level = float(targets[t])
        equity, fee = stock + cash, 0.0
        if abs(new_level - level) > EPS:
            cost = abs(new_level * equity - stock) * config.FEE_RATE  # 사고판 금액의 수수료를 자산에서 뗀 뒤 목표 비중으로 나눈다
            fee = cost / equity
            equity -= cost
            stock, cash, level = new_level * equity, (1 - new_level) * equity, new_level
        rows.append({"date": t, "position": LONG if level > EPS else CASH, "weight": level, "fee": fee, "equity": stock + cash})
    return pd.DataFrame(rows).set_index("date")


def buy_and_hold(samsung_close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """구간 첫날 종가에 삼성전자를 사서(수수료 한 번) 끝까지 들고 있을 때의 자산."""
    prices = samsung_close.reindex(dates.sort_values())
    return (1 - config.FEE_RATE) * prices / prices.iloc[0]


def max_drawdown(equity: pd.Series) -> float:
    """시작 자산 1.0 을 포함해 누적 최고점 대비 가장 크게 떨어진 비율(음수)."""
    values = np.r_[1.0, equity.to_numpy(float)]
    peak = np.maximum.accumulate(values)
    return float((values / peak - 1).min())


def total_return(equity: pd.Series) -> float:
    """시작 자산 1.0 대비 마지막 자산의 수익률."""
    return float(equity.iloc[-1] - 1)


def cagr(equity: pd.Series) -> float:
    """연 환산 수익률. 수익이 생기는 구간 수(거래일 수 - 1)로 환산한다."""
    years = max(len(equity) - 1, 1) / TRADING_DAYS_PER_YEAR
    return float(equity.iloc[-1] ** (1 / years) - 1)


def calmar(equity: pd.Series) -> float:
    """칼마 비율 = 연 환산 수익률 ÷ |MDD|. MDD 가 0 이면 NaN."""
    mdd = max_drawdown(equity)
    return float(cagr(equity) / abs(mdd)) if mdd < 0 else float("nan")


def summary(equity: pd.Series) -> dict:
    """자산 곡선의 핵심 숫자."""
    return {"total_return": total_return(equity), "mdd": max_drawdown(equity), "cagr": cagr(equity), "calmar": calmar(equity)}


def judge_a(model_equity: pd.Series, hold_equity: pd.Series) -> dict:
    """기준 A: MDD 가 들고 있기의 절반 이하이고, 누적 수익률이 들고 있기의 절반 이상(들고 있기 수익 ≤ 0 이면 들고 있기 이상)."""
    m, h = summary(model_equity), summary(hold_equity)
    mdd_ok = abs(m["mdd"]) <= 0.5 * abs(h["mdd"]) + EPS
    if h["total_return"] > EPS:
        return_rule, return_ok = "모델 수익 ≥ 들고 있기 수익의 절반", m["total_return"] >= 0.5 * h["total_return"] - EPS
    else:
        return_rule, return_ok = "모델 수익 ≥ 들고 있기 수익 (들고 있기 수익 ≤ 0)", m["total_return"] >= h["total_return"] - EPS
    return {"model": m, "hold": h, "mdd_ok": mdd_ok, "return_rule": return_rule, "return_ok": return_ok,
            "passed": mdd_ok and return_ok}


def judge_b(news_equity: pd.Series, numeric_equity: pd.Series) -> dict:
    """기준 B: 같은 구간에서 뉴스 포함 모델 MDD ≤ 숫자 모델 MDD(절댓값) 이고, 누적 수익률 ≥ 숫자 모델.

    두 곡선은 같은 날짜에 현금으로 시작한 모의 매매여야 한다(긴 곡선을 잘라 넘기면 수익 기준점이 달라진다).
    """
    if not news_equity.index.equals(numeric_equity.index):
        raise ValueError("기준 B 는 같은 날짜 구간의 두 자산 곡선으로만 판정한다")
    n, base = summary(news_equity), summary(numeric_equity)
    mdd_ok = abs(n["mdd"]) <= abs(base["mdd"]) + EPS
    return_ok = n["total_return"] >= base["total_return"] - EPS
    return {"news": n, "numeric": base, "mdd_ok": mdd_ok, "return_ok": return_ok, "passed": mdd_ok and return_ok}


def direction_hit_rate(signals: pd.Series, labels: pd.Series) -> dict:
    """'상승'·'하락' 신호를 낸 날 중 정답과 같은 비율과 치명적 오류(상승↔하락 정반대) 비율."""
    joined = pd.DataFrame({"signal": signals, "label": labels}).dropna()
    directional = joined[joined["signal"].isin([UP, DOWN])]
    if directional.empty:
        return {"n": 0, "hit_rate": float("nan"), "critical_error_rate": float("nan")}
    critical = ((directional["signal"] == UP) & (directional["label"] == DOWN)) | \
               ((directional["signal"] == DOWN) & (directional["label"] == UP))
    return {"n": len(directional), "hit_rate": float((directional["signal"] == directional["label"]).mean()),
            "critical_error_rate": float(critical.mean())}


def calibration(confidence: pd.Series, correct: pd.Series, edges=(0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> pd.DataFrame:
    """확신도 구간별 신호 수와 실제 적중률."""
    df = pd.DataFrame({"confidence": confidence, "correct": correct.astype(float)}).dropna()
    df["bin"] = pd.cut(df["confidence"], list(edges), include_lowest=True)
    return df.groupby("bin", observed=False)["correct"].agg(["count", "mean"]).rename(columns={"mean": "hit_rate"})


def constant_signals(dates: pd.DatetimeIndex, label: str) -> pd.Series:
    """항상 같은 신호를 내는 비교 기준선."""
    return pd.Series(label, index=dates)


def load_prices() -> tuple[pd.Series, pd.Series]:
    """삼성전자와 선정한 인버스 ETF 종가."""
    raw = Path(config.RAW_DIR) / "prices"
    samsung = pd.read_parquet(raw / "samsung_ohlcv.parquet")["Close"]
    inverse = pd.read_parquet(raw / f"inverse_{config.INVERSE_ETF_CODE}.parquet")["Close"]
    return samsung, inverse


def evaluate(signals: pd.DataFrame, start=None, end=None, down_action: str | None = None, labels_known_before=None,
             labels_path: str | None = None) -> dict:
    """신호 표(signal, confidence)를 구간에서 모의 매매하고 기준 A, 기준선, 보조 지표를 계산한다. down_action 은 하락 신호 때 행동.

    labels_known_before 를 주면 정답 기간이 그날 전에 끝난(그때 이미 확정된) 정답만으로 적중률을 잰다.
    주지 않으면 end 다음 날을 쓴다. 그래서 구간 끝 무렵 신호의 정답(구간 뒤 종가로 정해짐)은 적중률에 들어가지 않는다.
    """
    samsung, inverse = load_prices()
    known = pd.Timestamp(labels_known_before) if labels_known_before is not None else (
        pd.Timestamp(end) + pd.Timedelta(days=1) if end is not None else None)
    label_table = pd.read_parquet(labels_path or Path(config.PROCESSED_DIR) / "labels.parquet")  # 공식 판정은 LABELS_OFFICIAL_PATH
    if known is not None:
        label_table = label_table[label_table["future_date"] < known]
    labels = label_table["label"]
    sig = signals.loc[start:end]
    dates = sig.index
    model_eq = simulate(sig["signal"], samsung, inverse, down_action)["equity"]
    hold_eq = buy_and_hold(samsung, dates)
    baselines = {name: summary(simulate(constant_signals(dates, label), samsung, inverse, down_action)["equity"])
                 for name, label in [("항상 상승", UP), ("항상 보합(현금)", FLAT), ("항상 하락", DOWN)]}
    correct = (sig["signal"] == labels.reindex(dates))
    has_label = labels.reindex(dates).notna()
    directional = has_label & sig["signal"].isin([UP, DOWN])
    return {
        "period": (str(dates.min().date()), str(dates.max().date()), len(dates)),
        "judge_a": judge_a(model_eq, hold_eq),
        "baselines": baselines,
        "direction": direction_hit_rate(sig["signal"], labels),
        "signal_share": sig["signal"].value_counts(normalize=True).to_dict(),
        "label_share": labels.reindex(dates).dropna().value_counts(normalize=True).to_dict(),
        "accuracy": float(correct[has_label].mean()),
        "calibration": calibration(sig["confidence"][has_label], correct[has_label]),
        "calibration_directional": calibration(sig["confidence"][directional], correct[directional]),
        "hit_by_signal": correct[has_label].groupby(sig["signal"][has_label]).agg(["count", "mean"]),
        "equity": model_eq, "hold_equity": hold_eq, "down_action": down_action or config.DOWN_ACTION,
    }


def evaluate_b(numeric: pd.DataFrame, news: pd.DataFrame, down_action: str | None = None, labels_path: str | None = None) -> dict:
    """뉴스 구간에서 두 신호를 같은 날 현금으로 시작해 각각 모의 매매하고 기준 B 를 판정한다."""
    samsung, inverse = load_prices()
    dates = numeric.loc[config.CRITERION_B_START:config.CRITERION_B_END].index
    news = news.loc[dates.min():dates.max()]
    if not news.index.equals(dates):
        raise ValueError("뉴스 포함 신호와 숫자 모델 신호의 날짜가 다르다")
    if "numeric_signal" not in news or not news["numeric_signal"].equals(numeric.loc[dates, "signal"]):
        raise ValueError("뉴스 보정 신호가 비교할 숫자 모델 신호를 보정한 것이 아니다(지표 묶음이 바뀐 뒤 다시 만들지 않음)")
    news_eq = simulate(news["signal"], samsung, inverse, down_action)["equity"]
    numeric_eq = simulate(numeric.loc[dates, "signal"], samsung, inverse, down_action)["equity"]
    labels = pd.read_parquet(labels_path or Path(config.PROCESSED_DIR) / "labels.parquet")["label"]
    return {"period": (str(dates.min().date()), str(dates.max().date()), len(dates)),
            "judge_b": judge_b(news_eq, numeric_eq),
            "hold": summary(buy_and_hold(samsung, dates)),
            "direction_news": direction_hit_rate(news["signal"], labels),
            "direction_numeric": direction_hit_rate(numeric.loc[dates, "signal"], labels),
            "adjusted_days": int(news["adjusted"].sum()), "scored_days": int(news["news_scored"].sum())}


def regime(hold_return: float) -> str:
    """구간 들고 있기 수익으로 장세를 나눈다(config.REGIME_UP·REGIME_DOWN, 결과 보기 전에 고정)."""
    if hold_return >= config.REGIME_UP:
        return "상승장"
    if hold_return <= config.REGIME_DOWN:
        return "하락장"
    return "횡보장"


def period_stats(signals: pd.Series, close: pd.Series, labels: pd.DataFrame, start, end) -> dict:
    """한 구간을 현금에서 다시 시작해 모의 매매(하락=현금)하고, 들고 있기와 비교한 숫자를 돌려준다.
    적중률은 구간 끝 다음 날 전에 확정된 정답만 쓴다."""
    sig = signals.loc[start:end]
    dates = sig.index
    sim = simulate(sig, close, pd.Series(dtype=float), "cash")
    model = sim["equity"]
    hold = buy_and_hold(close, dates)
    known = labels[labels["future_date"] < pd.Timestamp(dates.max()) + pd.Timedelta(days=1)]["label"].reindex(dates)
    has = known.notna()
    m, h = summary(model), summary(hold)
    down_days = sig[has] == DOWN
    return {"start": str(dates.min().date()), "end": str(dates.max().date()), "days": len(dates),
            "model": m, "hold": h, "regime": regime(h["total_return"]), "judge_a": judge_a(model, hold)["passed"],
            "accuracy": float((sig[has] == known[has]).mean()) if has.any() else float("nan"),
            "always_up_accuracy": float((known[has] == UP).mean()) if has.any() else float("nan"),
            "always_flat_accuracy": float((known[has] == FLAT).mean()) if has.any() else float("nan"),
            "hold_share": float((sim["position"] == LONG).mean()),  # 실제로 삼성전자를 들고 있던 날 비율
            "down_precision": float((known[has][down_days] == DOWN).mean()) if down_days.any() else float("nan"),
            "actual_down_share": float((known[has] == DOWN).mean()) if has.any() else float("nan"),
            "cash_share": float((sig != UP).mean())}  # 보합(유지)·하락(현금) 신호 비율


def extended_report(save: bool = True) -> dict:
    """4일차 확장 검증 리포트: 확장 검증 구간(처음 보는 구간)·연도별·기존 검증 구간(참고)을 나눠 들고 있기와 비교한다."""
    from src.pipeline.train import ext_valid_start
    raw = Path(config.RAW_DIR) / "prices"
    close = pd.concat([pd.read_parquet(raw / f"samsung_ohlcv{config.EXT_SUFFIX}.parquet")["Close"],
                       pd.read_parquet(raw / "samsung_ohlcv.parquet")["Close"]]).sort_index().astype(float)
    signals = pd.read_parquet(config.SIGNALS_EXT_PATH)["signal"]
    labels = pd.read_parquet(config.LABELS_EXT_PATH)
    start = ext_valid_start(pd.read_parquet(config.FEATURES_EXT_PATH))
    ext_end, last = pd.Timestamp(config.EXT_VALID_END), signals.index.max()

    def years(lo, hi, tag):
        """lo~hi 를 해마다 자른 구간. 해의 일부만 들어가면 '부분' 으로 표시한다."""
        out = []
        for y in range(lo.year, hi.year + 1):
            s, e = max(lo, pd.Timestamp(f"{y}-01-01")), min(hi, pd.Timestamp(f"{y}-12-31"))
            partial = s > pd.Timestamp(f"{y}-01-10") or e < pd.Timestamp(f"{y}-12-20")
            out.append((f"{y}년{' (부분)' if partial else ''}{tag}", s, e))
        return out

    periods = [("확장 검증(처음 보는 구간)", start, ext_end), *years(start, ext_end, "")]
    periods += [("기존 검증(참고, 이미 본 구간)", pd.Timestamp(config.VALID_START), last),
                *years(pd.Timestamp(config.VALID_START), last, " · 참고"), ("전체(참고)", start, last)]
    rows = [{"name": n, "judged": i <= len(years(start, ext_end, "")), **period_stats(signals, close, labels, s, e)}
            for i, (n, s, e) in enumerate(periods)]
    report = extended_markdown(rows, start)
    if save:
        Path(config.EXT_REPORT_PATH).write_text(report, encoding="utf-8", newline="\n")
    return {"rows": rows, "report": report}


def extended_markdown(rows: list[dict], start: pd.Timestamp) -> str:
    """확장 검증 결과를 마크다운 리포트로 만든다. 성공 기준은 DAY4_PLAN.md 8절(결과 보기 전에 고정)."""
    pct = lambda v: "" if pd.isna(v) else f"{v:.1%}"
    table = pd.DataFrame([{
        "구간": r["name"], "기간": f"{r['start']} ~ {r['end']}", "일수": r["days"], "장세": r["regime"],
        "모델 수익 / MDD": f"{r['model']['total_return']:+.1%} / {r['model']['mdd']:.1%}", "모델 칼마": f"{r['model']['calmar']:.2f}",
        "들고 있기 수익 / MDD": f"{r['hold']['total_return']:+.1%} / {r['hold']['mdd']:.1%}",
        "기준 A": "통과" if r["judge_a"] else "실패", "3분류 정확도": pct(r["accuracy"]), "항상 상승 정확도": pct(r["always_up_accuracy"]),
        "항상 보합 정확도": pct(r["always_flat_accuracy"]),
        "하락 신호 적중": pct(r["down_precision"]), "실제 하락 비율": pct(r["actual_down_share"]), "상승 아닌 신호 비율": pct(r["cash_share"]), "삼성전자 보유일": pct(r["hold_share"]),
    } for r in rows])
    main = rows[0]
    years = [r for r in rows[1:] if r["judged"]]  # 판정은 확장 검증 구간 안의 해만(이미 본 구간은 참고)
    checks = []
    for r in years:
        if r["regime"] == "하락장":
            ok = abs(r["model"]["mdd"]) < abs(r["hold"]["mdd"])
            checks.append(f"- {r['name']}(하락장): 모델 MDD {r['model']['mdd']:.1%} vs 들고 있기 {r['hold']['mdd']:.1%} → {'얕다(통과)' if ok else '얕지 않다(실패)'}")
        elif r["regime"] == "상승장":
            ok = r["model"]["total_return"] >= 0.5 * r["hold"]["total_return"]
            checks.append(f"- {r['name']}(상승장): 모델 수익 {r['model']['total_return']:+.1%} vs 들고 있기의 절반 "
                          f"{0.5 * r['hold']['total_return']:+.1%} → {'이상(통과)' if ok else '미만(실패)'}")
    down_ok = pd.notna(main["down_precision"]) and main["down_precision"] > main["actual_down_share"]
    lines = ["# 4일차 확장 검증 (숫자 지표 과거 확장)", "",
             f"계산 시각 {pd.Timestamp.now(tz=config.KST):%Y-%m-%d %H:%M} (KST) · 계획 [DAY4_PLAN.md](../../DAY4_PLAN.md)", "",
             f"- 사전 등록 조건(결과를 보고 바꾸지 않는다): 지표 {len(config.FEATURE_SET_BASE)}개, 정답 {config.LABEL_HORIZON_DAYS}거래일 "
             f"±{config.LABEL_THRESHOLD:.0%}, 재학습 {config.RETRAIN_FREQS[config.RETRAIN_FREQ]}, 하락=현금, t 종가 체결, 수수료 {config.FEE_RATE:.1%}",
             f"- 확장 원본 {config.EXT_DATA_START} ~, 운영 지표가 모두 있는 첫날부터 {config.EXT_FIRST_TRAIN_MONTHS}개월은 학습만 하고 "
             f"{start.date()} 부터 검증한다. 구간마다 현금에서 다시 시작한다",
             f"- 장세: 그 구간 들고 있기 수익 ≥ {config.REGIME_UP:+.0%} 상승장, ≤ {config.REGIME_DOWN:+.0%} 하락장, 그 사이 횡보장",
             "- 공식 판정(1~3일차)과 운영 신호는 이 결과로 바뀌지 않는다",
             "- 확장 검증 구간의 데이터는 처음 보지만, 조건(지표 묶음·재학습 주기·모델 설정)은 그 뒤 구간을 보고 정했다. 완전한 사전 등록은 아니다",
             "- 기존 검증(참고) 줄은 확장 특징 표로 계산해 RSI·파라볼릭 SAR 값이 운영 표와 조금 다르다(원본 시작점 차이). 운영·공식 숫자와 직접 비교하지 않는다", "",
             "## 성공 기준 판정 (확장 검증 구간)", "",
             f"- 기준 A: 모델 {main['model']['total_return']:+.1%} / MDD {main['model']['mdd']:.1%}, 들고 있기 "
             f"{main['hold']['total_return']:+.1%} / MDD {main['hold']['mdd']:.1%} → **{'통과' if main['judge_a'] else '실패'}**",
             f"- 하락 신호 적중 {pct(main['down_precision'])} vs 실제 하락 비율 {pct(main['actual_down_share'])} → "
             f"**{'높다(통과)' if down_ok else '높지 않다(실패)'}**",
             *checks, "", "## 구간별 결과", "",
             "- 위 판정은 확장 검증 구간 안의 해만 쓴다. '· 참고' 줄은 이미 본 구간이다. '(부분)' 은 해의 일부만 들어간 구간이다",
             "- 3분류 정확도는 '항상 보합'·'항상 상승' 정확도와 함께 본다(보합이 많은 해는 보합만 내도 정확도가 높다)",
             "- '상승 아닌 신호 비율' 은 보합(포지션 유지)·하락(현금) 신호의 비율이고, 실제 보유는 '삼성전자 보유일' 이다. "
             "하락=현금 규칙이라 '항상 상승' 의 수익·MDD 는 들고 있기와 같다", "",
             table.to_markdown(index=False), ""]
    return "\n".join(lines) + "\n"


# ---- 5일차: 하락 위험 모델 판정 (DAY5_PLAN.md, 사전 등록 장부 evaluation/reports/preregistration.md) ----

def risk_weights(p: pd.Series, t_mid: float, t_high: float) -> pd.Series:
    """위험 확률을 3단계 목표 비중으로: t_high 이상 → 높음, t_mid 이상 → 중간, 그 밖 → 낮음(config.RISK_WEIGHTS)."""
    low, mid, high = config.RISK_WEIGHTS
    return pd.Series(np.where(p >= t_high, high, np.where(p >= t_mid, mid, low)), index=p.index)


def day5_thresholds(risk: pd.DataFrame) -> dict:
    """개발 구간의 위험 확률 분위수로 중간·높음 기준을 정한다(판정 구간 값은 보지 않는다)."""
    dev = risk.loc[config.DEV_START:config.DEV_END]
    q_mid, q_high = config.RISK_QUANTILES
    return {col: (float(dev[col].quantile(q_mid)), float(dev[col].quantile(q_high))) for col in ("p_risk_raw", "p_risk_cal", "p_logit")}


def day5_weights(risk: pd.DataFrame, th: dict) -> dict[str, pd.Series]:
    """사전 등록 조건(P1~P3)마다 날짜별 목표 비중."""
    p1 = risk_weights(risk["p_risk_raw"], *th["p_risk_raw"])
    p2 = risk_weights(risk["p_risk_cal"], *th["p_risk_cal"])
    votes = ((risk["p_risk_cal"] >= th["p_risk_cal"][1]).astype(int) + (risk["p_logit"] >= th["p_logit"][1]).astype(int)
             + risk["trend_vote"].astype(int))
    low, mid, high = config.RISK_WEIGHTS
    p3 = pd.Series(np.select([votes >= 3, votes == 2], [high, mid], low), index=risk.index)
    return {"P1": p1, "P2": p2, "P3": p3}


def up_month_hold(weights: pd.Series, close: pd.Series) -> tuple[float, int]:
    """그달 들고 있기 수익이 UP_MONTH_MIN_RETURN 이상인 달들에서 삼성전자를 들고 있던 날 비율과 그런 달 수."""
    px = close.reindex(weights.index)
    month = weights.index.to_period("M")
    ret = px.groupby(month).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    up = ret[ret >= config.UP_MONTH_MIN_RETURN].index
    held = (weights > EPS)[month.isin(up)]
    return (float(held.mean()) if len(held) else float("nan")), len(up)


def calibration_check(p: pd.Series, y: pd.Series) -> dict:
    """확률 구간별 실제 위험 비율과 Brier 점수. 구간은 0.1 간격, 표본 10개 미만 구간은 단조 확인에서 뺀다."""
    df = pd.DataFrame({"p": p, "y": y}).dropna()
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, config.CALIBRATION_BINS + 1), include_lowest=True)
    table = df.groupby("bin", observed=True)["y"].agg(["count", "mean"])
    used = table[table["count"] >= config.CALIBRATION_MIN_BIN]["mean"]
    return {"brier": float(((df["p"] - df["y"]) ** 2).mean()), "table": table,
            "monotonic": bool(used.is_monotonic_increasing) if len(used) >= 2 else False}


def day5_period(weights: pd.Series, close: pd.Series, start, end) -> dict:
    """한 구간을 현금에서 다시 시작해 목표 비중대로 모의 매매하고 들고 있기와 비교한다."""
    w = weights.loc[start:end]
    sim = simulate_target(w, close)
    hold = buy_and_hold(close, w.index)
    share, n_up = up_month_hold(w, close)
    return {"model": summary(sim["equity"]), "hold": summary(hold), "judge_a": judge_a(sim["equity"], hold),
            "hold_days": float((sim["weight"] > EPS).mean()), "avg_weight": float(sim["weight"].mean()),
            "up_month_hold": share, "up_months": n_up, "trades": int((sim["fee"] > 0).sum()), "equity": sim["equity"]}


def _day5_inputs():
    """판정에 쓰는 확장 구간 종가, 위험 신호, 위험 정답, 비교용 3분류 운영 모델 신호(확장 특징 표로 만든 것)."""
    from src.pipeline.labels import make_risk_labels
    raw = Path(config.RAW_DIR) / "prices"
    close = pd.concat([pd.read_parquet(raw / f"samsung_ohlcv{config.EXT_SUFFIX}.parquet")["Close"],
                       pd.read_parquet(raw / "samsung_ohlcv.parquet")["Close"]]).sort_index().astype(float)
    risk = pd.read_parquet(config.SIGNALS_RISK_PATH)
    three = pd.read_parquet(config.SIGNALS_EXT_PATH)["signal"]
    level = {UP: 1.0, DOWN: 0.0}
    w3, cur = [], 0.0
    for s in three:  # 3분류 운영 규칙(상승 100%, 보합 유지, 하락 현금)을 목표 비중으로
        cur = level.get(s, cur)
        w3.append(cur)
    return close, risk, make_risk_labels(close), pd.Series(w3, index=three.index)


def day5_evaluate(start, end) -> dict:
    """사전 등록 조건(P1~P3)과 지금 3분류 운영 모델을 한 구간에서 계산한다. 기준은 개발 구간에서만 정한다."""
    close, risk, labels, w3 = _day5_inputs()
    th = day5_thresholds(risk)
    weights = day5_weights(risk, th)
    base = day5_period(w3, close, start, end)
    y = labels["risk"].reindex(risk.loc[start:end].index)
    known = labels["future_date"].reindex(y.index) <= pd.Timestamp(end)  # 구간 끝까지 확정된 정답만
    cal = {name: calibration_check(risk.loc[start:end, col][known], y[known])
           for name, col in [("P1", "p_risk_raw"), ("P2", "p_risk_cal")]}
    results = {k: day5_period(w, close, start, end) for k, w in weights.items()}
    return {"thresholds": th, "base": base, "results": results, "calibration": cal, "start": str(start), "end": str(end),
            "risk_share": float(y[known].mean())}


def day5_markdown(ev: dict, title: str) -> str:
    """5일차 판정(또는 개발 구간) 결과 리포트."""
    pct = lambda v: "" if pd.isna(v) else f"{v:.1%}"
    b = ev["base"]
    rows = [{"조건": "지금 3분류 운영 모델(비교)", "수익 / MDD": f"{b['model']['total_return']:+.1%} / {b['model']['mdd']:.1%}",
             "칼마": f"{b['model']['calmar']:.2f}", "보유일": pct(b["hold_days"]), "평균 비중": pct(b["avg_weight"]),
             "상승장 달 보유일": pct(b["up_month_hold"]), "포지션 변경": b["trades"], "기준 A": "통과" if b["judge_a"]["passed"] else "실패"}]
    checks = []
    for k, r in ev["results"].items():
        a_ok, calmar_ok = r["judge_a"]["passed"], r["model"]["calmar"] > b["model"]["calmar"]
        up_ok = pd.notna(r["up_month_hold"]) and r["up_month_hold"] >= config.UP_MONTH_MIN_HOLD
        rows.append({"조건": f"{k}: {config.PREREGISTERED[k]}", "수익 / MDD": f"{r['model']['total_return']:+.1%} / {r['model']['mdd']:.1%}",
                     "칼마": f"{r['model']['calmar']:.2f}", "보유일": pct(r["hold_days"]), "평균 비중": pct(r["avg_weight"]),
                     "상승장 달 보유일": pct(r["up_month_hold"]), "포지션 변경": r["trades"], "기준 A": "통과" if a_ok else "실패"})
        checks.append(f"- **{k}**: 기준 A {'통과' if a_ok else '실패'} · 칼마가 운영 모델보다 {'높다' if calmar_ok else '높지 않다'}"
                      f"({r['model']['calmar']:.2f} vs {b['model']['calmar']:.2f}) · 상승장 달 보유일 {pct(r['up_month_hold'])}"
                      f"({r['up_months']}개월, 기준 {config.UP_MONTH_MIN_HOLD:.0%}) {'통과' if up_ok else '실패'}")
    cal = ev["calibration"]
    h = b["hold"]
    th = ev["thresholds"]
    lines = [f"# {title}", "", f"계산 시각 {pd.Timestamp.now(tz=config.KST):%Y-%m-%d %H:%M} (KST) · 구간 {ev['start'][:10]} ~ {ev['end'][:10]} · "
             f"계획 [DAY5_PLAN.md](../../DAY5_PLAN.md) · 장부 [preregistration.md](preregistration.md)", "",
             f"- 들고 있기: {h['total_return']:+.1%} / MDD {h['mdd']:.1%}. 구간 실제 하락 위험 비율 {ev['risk_share']:.1%}",
             f"- 기준(개발 구간 {config.DEV_START} ~ {config.DEV_END} 의 분위수 {config.RISK_QUANTILES}): "
             + ", ".join(f"{k} 중간 {v[0]:.3f} · 높음 {v[1]:.3f}" for k, v in th.items()),
             "- 하락=현금 규칙이라 '항상 상승' 의 수익·MDD 는 들고 있기와 같다",
             "- 모든 조건은 구간 첫날 현금에서 그날 목표 비중으로 들어간다. 3분류 운영 모델의 목표 비중은 이전 신호에서 이어진 값이다(보합 = 유지)", "",
             "## 판정", "", *checks]
    if cal:
        c1, c2 = cal["P1"], cal["P2"]
        lines.append(f"- **신뢰도(P2 보정)**: 구간별 실제 위험 비율이 단조 증가 {'예' if c2['monotonic'] else '아니오'}, "
                     f"Brier {c2['brier']:.4f} vs 보정 전 {c1['brier']:.4f} → {'통과' if c2['monotonic'] and c2['brier'] < c1['brier'] else '실패'}")
    lines += ["", "## 결과 표", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    if cal:
        lines += ["## 확률 구간별 실제 위험 비율", ""]
    for name, c in ([("P1 보정 전", cal["P1"]), ("P2 보정 후", cal["P2"])] if cal else []):
        t = c["table"].rename(columns={"count": "일수", "mean": "실제 위험 비율"}).reset_index().rename(columns={"bin": "확률 구간"})
        t["확률 구간"] = t["확률 구간"].astype(str)
        t["실제 위험 비율"] = t["실제 위험 비율"].map(pct)
        lines += [f"### {name} (Brier {c['brier']:.4f})", "", t.to_markdown(index=False), ""]
    return "\n".join(lines) + "\n"


def day5_dev_report() -> str:
    """개발 구간 결과(여러 번 봐도 되는 구간). 판정 구간은 쓰지 않는다."""
    return day5_markdown(day5_evaluate(config.DEV_START, config.DEV_END), "5일차 개발 구간 결과 (판정 아님)")


def day5_judge(ids: list[str] | None = None) -> str:
    """사전 등록 조건을 판정 구간에서 한 번만 판정해 리포트와 판정 기록을 남긴다. 이미 판정한 조건이 있으면 멈춘다."""
    import json
    ids = ids or list(config.PREREGISTERED)
    unknown = [i for i in ids if i not in config.PREREGISTERED]
    if unknown:
        raise ValueError(f"장부에 없는 조건은 판정하지 않는다: {unknown}")
    path = Path(config.JUDGED_PATH)
    judged = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    again = [i for i in ids if i in judged]
    if again:
        raise ValueError(f"판정 구간을 이미 쓴 조건이다(다시 판정하지 않는다): {again}")
    import hashlib
    ev = day5_evaluate(config.JUDGE_START, config.JUDGE_END)
    ev["results"] = {k: v for k, v in ev["results"].items() if k in ids}
    if "P2" not in ids:
        ev["calibration"] = None  # 보정 판정은 P2 를 판정할 때만 싣는다
    report = day5_markdown(ev, f"5일차 판정 ({', '.join(ids)}, 판정 구간, 조건마다 한 번만)")
    report_path = Path(config.DAY5_REPORT_PATH).with_name(f"day5_judgement_{'_'.join(ids)}.md")
    report_path.write_text(report, encoding="utf-8", newline="\n")
    stamp = pd.Timestamp.now(tz=config.KST).strftime("%Y-%m-%d %H:%M")
    signal_hash = hashlib.sha256(Path(config.SIGNALS_RISK_PATH).read_bytes()).hexdigest()[:16]
    base_calmar = ev["base"]["model"]["calmar"]
    for i in ids:
        r = ev["results"][i]
        judged[i] = {"judged_at": stamp, "report": report_path.as_posix(), "signals_sha256_16": signal_hash, "thresholds": ev["thresholds"],
                     "return": r["model"]["total_return"], "mdd": r["model"]["mdd"], "calmar": r["model"]["calmar"],
                     "judge_a": bool(r["judge_a"]["passed"]), "calmar_beats_operational": bool(r["model"]["calmar"] > base_calmar),
                     "up_month_hold": r["up_month_hold"],
                     "up_month_ok": bool(pd.notna(r["up_month_hold"]) and r["up_month_hold"] >= config.UP_MONTH_MIN_HOLD)}
        if i == "P2":
            c1, c2 = ev["calibration"]["P1"], ev["calibration"]["P2"]
            judged[i].update(brier_raw=c1["brier"], brier_cal=c2["brier"], monotonic=c2["monotonic"],
                             calibration_ok=bool(c2["monotonic"] and c2["brier"] < c1["brier"]))
    path.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def rule_title(rule: str) -> str:
    """하락 행동 규칙의 출력용 제목. 공식 판정 규칙이 아니면 탐색 결과라고 밝힌다."""
    label = config.DOWN_ACTION_LABELS[rule]
    if rule == config.OFFICIAL_DOWN_ACTION:
        return f"하락 = {label} (1~3일차 공식 판정 규칙)"
    return f"하락 = {label} ({config.DOWN_ACTION_CHANGED_ON} 변경. 검증 구간을 본 뒤의 재계산이라 탐색 결과)"


def print_judge_a(title: str, r: dict, official: bool = True):
    """기준 A 결과 한 묶음을 출력한다. 공식 판정 규칙이 아니거나 official=False(후보 모델 등)면 '(참고)'로 표시한다."""
    a = r["judge_a"]
    label = "기준 A" if official and r.get("down_action") == config.OFFICIAL_DOWN_ACTION else "(참고) 기준 A"
    print(f"  {title}: 수익 {a['model']['total_return']:+.1%}  MDD {a['model']['mdd']:.1%}  칼마 {a['model']['calmar']:.2f}"
          f"  → {label} {'통과' if a['passed'] else '실패'} (MDD 조건 {a['mdd_ok']}, 수익 조건 {a['return_ok']})")


def main():
    """하락 행동 규칙(인버스·현금)마다 기준 A·후보 모델·기준 B 를 계산해 출력하고 자산 곡선을 저장한다."""
    signals = pd.read_parquet(config.SIGNALS_NUMERIC_PATH)
    v1_path = Path(config.SIGNALS_BASE_V1_PATH)
    if not v1_path.exists():  # 공식 판정은 1~3일차 45개 모델로 한 것이다. 운영 신호로 대신하지 않는다
        raise FileNotFoundError(f"{v1_path} 가 없다. train.main 을 먼저 돌린다")
    official_signals = pd.read_parquet(v1_path)
    official_end = config.OFFICIAL_VALID_END  # 공식 판정은 처음 데이터 끝날까지로 고정한다. 주간 갱신으로 늘어난 날은 아래에 참고로 낸다
    base = evaluate(official_signals, end=official_end, down_action=config.OFFICIAL_DOWN_ACTION, labels_path=config.LABELS_OFFICIAL_PATH)
    a = base["judge_a"]
    print(f"[1~3일차 공식 판정 모델: 지표 {len(config.FEATURE_SET_BASE_V1)}개, 정답 {config.OFFICIAL_LABEL_HORIZON_DAYS}거래일, "
          f"재학습 {config.RETRAIN_FREQS[config.OFFICIAL_RETRAIN_FREQ]}]")
    print("구간", base["period"], f"| 들고 있기 수익 {a['hold']['total_return']:+.1%}  MDD {a['hold']['mdd']:.1%}  칼마 {a['hold']['calmar']:.2f}")
    print("방향 적중", {k: round(v, 3) for k, v in base["direction"].items()}, "| 3분류 정확도", round(base["accuracy"], 3))
    print("신호 분포", {k: round(v, 3) for k, v in base["signal_share"].items()}, "| 정답 분포", {k: round(v, 3) for k, v in base["label_share"].items()})
    print("신호별 적중\n" + base["hit_by_signal"].round(3).to_string())
    print("확신도 구간별(상승·하락 신호만)\n" + base["calibration_directional"].round(3).to_string())
    cand_path = Path(config.SIGNALS_CANDIDATE_PATH)
    news_path = Path(config.PROCESSED_DIR) / "signals_news.parquet"
    for rule in config.DOWN_ACTIONS:
        print(f"[{rule_title(rule)}]")
        if rule == config.OFFICIAL_DOWN_ACTION:
            print_judge_a(f"숫자 모델(1~3일차 {len(config.FEATURE_SET_BASE_V1)}개)", base)
            for name, s in base["baselines"].items():
                print(f"    기준선 {name}: 수익 {s['total_return']:+.1%}  MDD {s['mdd']:.1%}")
        r = evaluate(signals, end=official_end, down_action=rule)
        print_judge_a(f"운영 숫자 모델({len(config.FEATURE_SET_BASE)}개 · 정답 {config.LABEL_HORIZON_DAYS}거래일 · 재학습 "
                      f"{config.RETRAIN_FREQS[config.RETRAIN_FREQ]}, {config.BASE_CHANGED_ON} 변경 · 탐색)", r, official=False)
        print(f"    3분류 정확도 {r['accuracy']:.3f} · 방향 적중 {r['direction']['hit_rate']:.3f}")
        if signals.index.max() > pd.Timestamp(official_end):  # 늘어난 구간까지(참고, 공식 판정 아님). 화면 자산 곡선은 이 전체 곡선을 쓴다
            r = evaluate(signals, down_action=rule)
            a = r["judge_a"]
            print(f"  (참고) 갱신으로 늘어난 구간 포함 {r['period']}: 수익 {a['model']['total_return']:+.1%}  MDD {a['model']['mdd']:.1%}"
                  f" | 들고 있기 {a['hold']['total_return']:+.1%}  MDD {a['hold']['mdd']:.1%} — 공식 판정 아님")
        pd.DataFrame({"model": r["equity"], "hold": r["hold_equity"]}).to_parquet(Path(config.PROCESSED_DIR) / f"equity_numeric_{rule}.parquet")
        if cand_path.exists():  # 후보 모델(A+B+D 추가): 사전 등록 ①안, 검증 구간을 본 뒤 한 번만 실행한 탐색
            candidate = pd.read_parquet(cand_path)
            c = evaluate(candidate, end=official_end, down_action=rule)
            print_judge_a(f"후보 모델(①안 탐색, 재학습 {config.RETRAIN_FREQS[config.RETRAIN_FREQ]}, 공식 판정 아님)", c, official=False)
            if candidate.index.max() > pd.Timestamp(official_end):  # 화면 자산 곡선은 기존 모델처럼 늘어난 구간까지 저장한다
                c = evaluate(candidate, down_action=rule)
            pd.DataFrame({"model": c["equity"], "hold": c["hold_equity"]}).to_parquet(Path(config.PROCESSED_DIR) / f"equity_candidate_{rule}.parquet")
        if news_path.exists():
            news = pd.read_parquet(news_path)
            pair = next(((name, s) for name, s in [("운영", signals), ("1~3일차", official_signals)]
                         if "numeric_signal" in news and news["numeric_signal"].equals(s["signal"].reindex(news.index))), None)
            if pair is None:
                print("  기준 B: 뉴스 보정 신호가 지금 숫자 모델과 짝이 맞지 않아 계산하지 않는다(재채점 뒤 news_signal 을 다시 돌린다)")
                continue
            b = evaluate_b(pair[1], news, down_action=rule,
                           labels_path=config.LABELS_OFFICIAL_PATH if pair[0] == "1~3일차" else None)
            j = b["judge_b"]
            prefix = "" if rule == config.OFFICIAL_DOWN_ACTION and pair[0] == "1~3일차" else "(참고) "
            print(f"  {prefix}기준 B ({pair[0]} 숫자 모델 기준) {b['period']}: 뉴스 보정 {j['news']['total_return']:+.1%} MDD {j['news']['mdd']:.1%} | "
                  f"숫자 모델 {j['numeric']['total_return']:+.1%} MDD {j['numeric']['mdd']:.1%} | "
                  f"들고 있기 {b['hold']['total_return']:+.1%} MDD {b['hold']['mdd']:.1%} → {'통과' if j['passed'] else '실패'}")


if __name__ == "__main__":
    import sys
    if "--ext" in sys.argv[1:]:
        print(extended_report()["report"])
    elif "--day5-dev" in sys.argv[1:]:
        print(day5_dev_report())
    elif "--day5-judge" in sys.argv[1:]:
        print(day5_judge())
    else:
        main()
