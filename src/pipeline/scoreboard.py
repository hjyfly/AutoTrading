"""실시간 성적표. 매일 기록한 신호(signals_live.parquet)를 정답이 확정되는 대로 채점하고 모델끼리·들고 있기와 비교한다.

- 정답: t 종가 → t+5거래일 종가 ±3%. 5거래일 뒤 종가가 확정된 신호만 채점한다.
- 모의 매매: 첫 기록일 현금 시작, 기록이 빠진 거래일은 '보합'(포지션 유지)으로 본다. 규칙은 backtest.simulate 와 같다.
- 성능 저하 경고: 최근 DRIFT_WINDOW 개 확정 신호의 정확도가 같은 기간 '항상 가장 많은 정답' 비율보다 낮으면 경고한다.
"""
import json
from pathlib import Path

import pandas as pd

from src import config, tools
from src.pipeline import backtest
from src.pipeline.labels import FLAT, make_labels

LIVE_PATH = Path(config.PROCESSED_DIR) / "signals_live.parquet"
MODELS = {"base": "운영 모델(최종 신호, 뉴스 보정 포함)", "candidate": "후보 모델(A+B+D, 비교용)",
          "base_v1": "1~3일차 45개 모델(운영 변경 전 기록, 참고)"}


def load_live() -> pd.DataFrame:
    """실시간 기록. 옛 기록에 모델 열이 없으면 기존 모델로 본다."""
    if not LIVE_PATH.exists():
        return pd.DataFrame(columns=["model", "signal", "confidence"])
    live = pd.read_parquet(LIVE_PATH)
    if "model" not in live.columns:
        live["model"] = "base"
    return live


def load_prices(until: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """until 전날까지 확정된 삼성전자·인버스 종가(저장본 + 실시간으로 이어 붙인 값)."""
    samsung = tools.series_until("prices", "samsung_ohlcv", until)[0]["Close"].astype(float)
    inverse = tools.series_until("prices", f"inverse_{config.INVERSE_ETF_CODE}", until)[0]["Close"].astype(float)
    return samsung, inverse


def realtime_mask(rows: pd.DataFrame) -> pd.Series:
    """그날 장 마감(15:30) 전에 실제로 만든 신호인가. 나중에 채워 넣은 신호는 실시간 성적에서 뺀다."""
    if "created_at" not in rows.columns:
        return pd.Series(False, index=rows.index)
    hour, minute = map(int, config.MARKET_CLOSE["prices"][1].split(":"))
    deadline = pd.Series(rows.index, index=rows.index) + pd.Timedelta(hours=hour, minutes=minute)
    return pd.to_datetime(rows["created_at"]) <= deadline


def score_model(signals: pd.Series, samsung: pd.Series, inverse: pd.Series) -> dict:
    """한 모델의 기록을 채점한다. signals 는 날짜별 신호(상승·보합·하락)."""
    signals = signals.sort_index()
    labels = make_labels(samsung)["label"]
    matured = signals[signals.index.isin(labels.index)]
    correct = matured == labels.reindex(matured.index)
    out = {"n_signals": int(len(signals)), "n_matured": int(len(matured)),
           "first_date": str(signals.index.min().date()) if len(signals) else None,
           "accuracy": float(correct.mean()) if len(matured) else None,
           "direction": backtest.direction_hit_rate(matured, labels)}
    days = samsung.index[(samsung.index >= signals.index.min())] if len(signals) else samsung.index[:0]
    if len(days) >= 2:
        daily = signals.reindex(days).fillna(FLAT)  # 기록이 빠진 날은 포지션 유지
        by_rule, curves = {}, {}
        for rule in config.DOWN_ACTIONS:  # 가격이 모자란 규칙(예: 인버스 가격 조회 실패)은 빠지고 나머지는 계산한다
            try:
                curves[rule] = backtest.simulate(daily, samsung, inverse, rule)["equity"]
                by_rule[rule] = backtest.summary(curves[rule])
            except ValueError:
                continue
        out.update(by_rule=by_rule, down_action=config.DOWN_ACTION,
                   hold=backtest.summary(backtest.buy_and_hold(samsung, days)), trading_days=int(len(days)))
        if config.DOWN_ACTION in curves:
            out.update(model=by_rule[config.DOWN_ACTION],
                       equity={str(d.date()): round(float(v), 6) for d, v in curves[config.DOWN_ACTION].items()})
        else:
            out["model_note"] = "현재 규칙에 필요한 가격이 없어 모의 매매를 계산하지 못했다"
    out["drift"] = drift_check(matured, labels)
    return out


def score_weight_model(rows: pd.DataFrame, samsung: pd.Series) -> dict:
    """목표 비중 기록(5일차 하락 위험 비교 조건)을 채점한다. 기록이 빠진 날은 앞 비중을 유지한다.
    위험 적중은 '비중 0%' 날 중 실제로 5거래일 안에 -3% 이하로 내려간 비율이다."""
    from src.pipeline.labels import make_risk_labels
    weights = rows["weight"].astype(float).sort_index()
    labels = make_risk_labels(samsung)
    matured = weights[weights.index.isin(labels.index)]
    high = matured[matured == config.RISK_WEIGHTS[2]]  # 위험 높음 비중
    out = {"n_signals": int(len(weights)), "n_matured": int(len(matured)), "accuracy": None,
           "first_date": str(weights.index.min().date()) if len(weights) else None,
           "risk_hit": float(labels.loc[high.index, "risk"].mean()) if len(high) else None,
           "drift": {"status": "해당 없음(비중 모델)"}}
    days = samsung.index[samsung.index >= weights.index.min()] if len(weights) else samsung.index[:0]
    if len(days) >= 2:
        sim = backtest.simulate_target(weights.reindex(days).ffill(), samsung)
        model = backtest.summary(sim["equity"])
        out.update(model=model, by_rule={"cash": model}, down_action="cash", trading_days=int(len(days)),
                   hold=backtest.summary(backtest.buy_and_hold(samsung, days)),
                   equity={str(d.date()): round(float(v), 6) for d, v in sim["equity"].items()})
    return out


def drift_check(matured: pd.Series, labels: pd.Series) -> dict:
    """최근 DRIFT_WINDOW 개 확정 신호의 정확도를 같은 기간 '항상 가장 많은 정답' 비율과 비교한다."""
    n = config.DRIFT_WINDOW
    if len(matured) < n:
        return {"status": "표본 부족", "needed": n, "have": int(len(matured))}
    recent = matured.iloc[-n:]
    truth = labels.reindex(recent.index)
    accuracy = float((recent == truth).mean())
    baseline = float(truth.value_counts(normalize=True).max())
    return {"status": "경고" if accuracy < baseline else "정상", "recent_accuracy": accuracy,
            "majority_baseline": baseline, "window": n,
            "message": (f"최근 {n}개 정확도 {accuracy:.0%} 가 '항상 {truth.value_counts().idxmax()}' 규칙({baseline:.0%})보다 낮다"
                        if accuracy < baseline else f"최근 {n}개 정확도 {accuracy:.0%} (기준 {baseline:.0%})")}


def build(today: pd.Timestamp | None = None) -> dict:
    """모든 모델의 성적표를 만들어 저장한다. today 보다 앞선 날의 확정 종가만 쓴다(기본: 지금까지 확정된 것)."""
    today = today or tools.resolve_as_of(None) + pd.Timedelta(days=1)  # 오늘 종가는 확정됐을 때만 들어온다(last_closed_date)
    live = load_live()
    board = {"updated_at": str(pd.Timestamp.now(tz=config.KST).floor("s")), "models": {}}
    if live.empty:
        board["note"] = "아직 실시간 기록이 없다"
    else:
        samsung, inverse = load_prices(today)
        board["price_last_date"] = str(samsung.index.max().date())
        extra = {name: f"비교용 지표 묶음 {name}" for name in config.LIVE_EXTRA_FEATURE_SETS}  # 지표 그룹 진단에서 고른 묶음
        extra[config.LIVE_RISK_MODEL] = f"하락 위험 비교 조건 {config.LIVE_RISK_CONDITION}(목표 비중, 비교용)"
        for model, title in {**MODELS, **extra}.items():
            rows = live[live["model"] == model]
            if rows.empty:
                continue
            realtime = rows[realtime_mask(rows)]
            entry = {"title": title, "n_backfilled": int(len(rows) - len(realtime))}
            if realtime.empty:
                entry.update(n_signals=0, n_matured=0, accuracy=None, drift={"status": "실시간 기록 없음"})
            elif model == config.LIVE_RISK_MODEL:
                entry.update(score_weight_model(realtime, samsung))
            else:
                entry.update(score_model(realtime["signal"], samsung, inverse))
            board["models"][model] = entry
    path = Path(config.SCOREBOARD_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return board


def summary_lines(board: dict) -> list[str]:
    """알림·화면용 한 줄 요약들."""
    lines = []
    for model, s in board.get("models", {}).items():
        part = f"{s['title']}: 실시간 기록 {s['n_signals']}일, 정답 확정 {s['n_matured']}일"
        if s.get("n_backfilled"):
            part += f" (사후 기록 {s['n_backfilled']}일 제외)"
        if s["accuracy"] is not None:
            part += f", 정확도 {s['accuracy']:.0%}"
        if "model" in s:
            other = {r: v for r, v in s.get("by_rule", {}).items() if r != s.get("down_action")}
            name = config.DOWN_ACTION_LABELS.get
            part += f", 모의 수익 {s['model']['total_return']:+.1%} (하락={name(s['down_action'])} 규칙, 들고 있기 {s['hold']['total_return']:+.1%}"
            part += "".join(f", 하락={name(r)} 규칙 {v['total_return']:+.1%}" for r, v in other.items()) + ")"
        drift = s["drift"]
        part += f" · {drift['status']}" + (f": {drift['message']}" if "message" in drift else "")
        lines.append(part)
    if lines and board.get("price_last_date"):
        lines.append(f"(가격 마지막 날 {board['price_last_date']} 기준)")
    return lines or [board.get("note", "기록 없음")]


def main():
    """성적표를 만들고 요약을 출력한다."""
    board = build()
    print(config.SCOREBOARD_PATH)
    print("\n".join(summary_lines(board)))


if __name__ == "__main__":
    main()
