from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.pipeline.labels import DOWN, FLAT, UP

CASH, LONG, INVERSE = "cash", "samsung", "inverse"
TRADING_DAYS_PER_YEAR = 252
EPS = 1e-12  # 판정 경계값 비교의 부동소수점 오차 허용


def simulate(signals: pd.Series, samsung_close: pd.Series, inverse_close: pd.Series) -> pd.DataFrame:
    """t일 신호대로 t일 종가에 체결하는 모의 매매. 현금으로 시작한다.

    상승 → 삼성전자 100%, 하락 → 인버스 ETF 100%, 보합 → 기존 포지션 유지.
    포지션을 바꾸면 파는 쪽·사는 쪽마다 거래 금액의 FEE_RATE 를 뗀다. 수익은 t일 종가 → t+1일 종가로 계산한다.
    돌려주는 equity 는 각 날짜 종가 시점(그날 체결 뒤)의 자산이다.
    """
    dates = signals.index.sort_values()
    closes = pd.DataFrame({LONG: samsung_close, INVERSE: inverse_close}).reindex(dates)
    if closes.isna().any().any():
        raise ValueError("신호 날짜에 가격이 없다")
    position, equity, rows = CASH, 1.0, []
    for i, t in enumerate(dates):
        if i > 0 and position != CASH:  # 전날 종가 → 오늘 종가 수익
            equity *= closes.loc[t, position] / closes.loc[dates[i - 1], position]
        target = {UP: LONG, DOWN: INVERSE}.get(signals[t], position)
        fee = 0.0
        if target != position:
            sides = (position != CASH) + (target != CASH)
            fee = 1 - (1 - config.FEE_RATE) ** sides
            equity *= 1 - fee
            position = target
        rows.append({"date": t, "signal": signals[t], "position": position, "fee": fee, "equity": equity})
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


def evaluate(signals: pd.DataFrame, start=None, end=None) -> dict:
    """신호 표(signal, confidence)를 구간에서 모의 매매하고 기준 A, 기준선, 보조 지표를 계산한다."""
    samsung, inverse = load_prices()
    labels = pd.read_parquet(Path(config.PROCESSED_DIR) / "labels.parquet")["label"]
    sig = signals.loc[start:end]
    dates = sig.index
    model_eq = simulate(sig["signal"], samsung, inverse)["equity"]
    hold_eq = buy_and_hold(samsung, dates)
    baselines = {name: summary(simulate(constant_signals(dates, label), samsung, inverse)["equity"])
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
        "equity": model_eq, "hold_equity": hold_eq,
    }


def evaluate_b(numeric: pd.DataFrame, news: pd.DataFrame) -> dict:
    """뉴스 구간에서 두 신호를 같은 날 현금으로 시작해 각각 모의 매매하고 기준 B 를 판정한다."""
    samsung, inverse = load_prices()
    dates = numeric.loc[config.NEWS_START:config.NEWS_END].index
    if not news.index.equals(dates):
        raise ValueError("뉴스 포함 신호와 숫자 모델 신호의 날짜가 다르다")
    news_eq = simulate(news["signal"], samsung, inverse)["equity"]
    numeric_eq = simulate(numeric.loc[dates, "signal"], samsung, inverse)["equity"]
    labels = pd.read_parquet(Path(config.PROCESSED_DIR) / "labels.parquet")["label"]
    return {"period": (str(dates.min().date()), str(dates.max().date()), len(dates)),
            "judge_b": judge_b(news_eq, numeric_eq),
            "hold": summary(buy_and_hold(samsung, dates)),
            "direction_news": direction_hit_rate(news["signal"], labels),
            "direction_numeric": direction_hit_rate(numeric.loc[dates, "signal"], labels),
            "adjusted_days": int(news["adjusted"].sum()), "scored_days": int(news["news_scored"].sum())}


def main():
    """숫자 모델 신호로 검증 구간 전체의 기준 A 를 판정해 출력하고 자산 곡선을 저장한다."""
    signals = pd.read_parquet(config.SIGNALS_NUMERIC_PATH)
    r = evaluate(signals)
    a = r["judge_a"]
    print("구간", r["period"])
    print(f"모델  수익 {a['model']['total_return']:+.1%}  MDD {a['model']['mdd']:.1%}  칼마 {a['model']['calmar']:.2f}")
    print(f"보유  수익 {a['hold']['total_return']:+.1%}  MDD {a['hold']['mdd']:.1%}  칼마 {a['hold']['calmar']:.2f}")
    print(f"기준 A: MDD 조건 {a['mdd_ok']}, 수익 조건 {a['return_ok']} ({a['return_rule']}) → {'통과' if a['passed'] else '실패'}")
    for name, s in r["baselines"].items():
        print(f"{name}  수익 {s['total_return']:+.1%}  MDD {s['mdd']:.1%}")
    print("방향 적중", {k: round(v, 3) for k, v in r["direction"].items()}, "| 3분류 정확도", round(r["accuracy"], 3))
    print("신호 분포", {k: round(v, 3) for k, v in r["signal_share"].items()}, "| 정답 분포", {k: round(v, 3) for k, v in r["label_share"].items()})
    print("신호별 적중\n" + r["hit_by_signal"].round(3).to_string())
    print("확신도 구간별(전체)\n" + r["calibration"].round(3).to_string())
    print("확신도 구간별(상승·하락 신호만)\n" + r["calibration_directional"].round(3).to_string())
    pd.DataFrame({"model": r["equity"], "hold": r["hold_equity"]}).to_parquet(Path(config.PROCESSED_DIR) / "equity_numeric.parquet")
    news_path = Path(config.PROCESSED_DIR) / "signals_news.parquet"
    if news_path.exists():
        b = evaluate_b(signals, pd.read_parquet(news_path))
        j = b["judge_b"]
        print("기준 B 구간", b["period"], f"채점된 날 {b['scored_days']}, 보정된 날 {b['adjusted_days']}")
        print(f"뉴스 포함 수익 {j['news']['total_return']:+.1%} MDD {j['news']['mdd']:.1%} | "
              f"숫자 모델 수익 {j['numeric']['total_return']:+.1%} MDD {j['numeric']['mdd']:.1%} | "
              f"들고 있기 수익 {b['hold']['total_return']:+.1%} MDD {b['hold']['mdd']:.1%}")
        print(f"기준 B: MDD 조건 {j['mdd_ok']}, 수익 조건 {j['return_ok']} → {'통과' if j['passed'] else '실패'}")


if __name__ == "__main__":
    main()
