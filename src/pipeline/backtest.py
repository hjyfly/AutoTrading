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
    closes = samsung_close.reindex(dates)
    if closes.isna().any():
        raise ValueError("신호 날짜에 가격이 없다")
    stock, cash, level, rows = 0.0, 1.0, 0.0, []
    for i, t in enumerate(dates):
        if i > 0:
            stock *= closes[t] / closes[dates[i - 1]]
        base = max(level, floor)  # 바닥 아래에서 시작하지 않는다(첫 상승에도 바닥 비중부터 채운다)
        new_level = {UP: min(1.0, base + 1 / steps), DOWN: max(floor, level - 1 / steps)}.get(signals[t], base)
        equity, fee = stock + cash, 0.0
        if abs(new_level - level) > EPS:
            cost = abs(new_level * equity - stock) * config.FEE_RATE  # 사고판 금액의 수수료를 자산에서 뗀 뒤 목표 비중으로 나눈다
            fee = cost / equity
            equity -= cost
            stock, cash, level = new_level * equity, (1 - new_level) * equity, new_level
        rows.append({"date": t, "signal": signals[t], "position": LONG if level > EPS else CASH, "weight": level,
                     "fee": fee, "equity": stock + cash})
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
    main()
