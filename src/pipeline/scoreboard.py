"""실시간 성적표. 매일 기록한 신호(signals_live.parquet)를 정답이 확정되는 대로 채점하고 모델끼리·들고 있기와 비교한다.

- 정답: t 종가 → t+5거래일 종가 ±3%. 5거래일 뒤 종가가 확정된 신호만 채점한다.
  11일차 기간별 모델(h<N>)은 그 기간 정답(t+N거래일 ±config.HORIZONS[N])으로 채점한다.
- 승격 비교(5.3)는 두 모델 첫 기록일 중 늦은 날부터 둘 다 현금에서 다시 계산한다(same_period). 한쪽만 기록이 빠진 날이 많으면 보류한다.
- 모의 매매: 첫 기록일 현금 시작, 기록이 빠진 거래일은 '보합'(포지션 유지)으로 본다. 규칙은 backtest.simulate 와 같다.
- 성능 저하 경고: 최근 DRIFT_WINDOW 개 확정 신호의 정확도가 같은 기간 '항상 가장 많은 정답' 비율보다 낮으면 경고한다.
- 다른 종목(config.MULTI_TICKERS, SERVICE.md 5.6): signals_live_tickers.parquet 를 (종목, 알고리즘)마다 같은 규칙으로 채점한다.
  비교 대상은 그 종목 들고 있기이고, 정답·모의 매매 가격은 그 종목 저장 원본(전날까지 덧붙인 확정 종가)이다. 주문과는 연결하지 않는다.
"""
import json
from pathlib import Path

import pandas as pd

from src import config, tools
from src.pipeline import backtest
from src.pipeline.labels import DOWN, FLAT, make_labels

LIVE_PATH = Path(config.PROCESSED_DIR) / "signals_live.parquet"
MODELS = {"base": "운영 모델(최종 신호, 뉴스는 참고용)", "candidate": "후보 모델(A+B+D, 비교용)",
          "base_v1": "1~3일차 45개 모델(운영 변경 전 기록, 참고)"}


def load_live() -> pd.DataFrame:
    """실시간 기록. 옛 기록에 모델 열이 없으면 기존 모델로 본다."""
    if not LIVE_PATH.exists():
        return pd.DataFrame(columns=["model", "signal", "confidence"])
    live = pd.read_parquet(LIVE_PATH)
    if "model" not in live.columns:
        live["model"] = "base"
    return live


def load_ticker_live() -> pd.DataFrame:
    """다른 종목 실시간 기록. 알고리즘 열이 비어 있으면 단독 모델(S0)로 본다(SERVICE.md 11.7)."""
    from src.pipeline.multi_ticker import SINGLE_MODEL_TAG
    path = Path(config.MULTI_LIVE_PATH)
    if not path.exists():
        return pd.DataFrame(columns=["code", "model", "signal", "confidence"])
    live = pd.read_parquet(path)
    live["model"] = live["model"].fillna(SINGLE_MODEL_TAG) if "model" in live.columns else SINGLE_MODEL_TAG
    return live


def ticker_close(code: str, until: pd.Timestamp) -> pd.Series:
    """until 전날까지 확정된 그 종목 종가(data/raw/tickers 저장 원본). 거래정지일(거래량 0)은 뺀다:
    신호를 학습한 정답(5거래일)이 거래정지일을 건너뛰어 셌으므로 채점도 같은 거래일로 센다(sector_models.clean_raw 와 같은 기준)."""
    from src.pipeline.multi_ticker import raw_path
    ohlcv = pd.read_parquet(raw_path(f"{code}_ohlcv"))
    ohlcv = ohlcv[ohlcv["Volume"].fillna(0) > 0] if "Volume" in ohlcv else ohlcv
    close = ohlcv["Close"].astype(float).dropna()
    return close[close.index < until.normalize()]


def score_tickers(until: pd.Timestamp) -> dict:
    """다른 종목 성적표: (종목, 알고리즘)마다 운영 모델과 같은 성공 기준 A·B·C·성능 저하 경고. 하락 행동은 현금만 계산한다."""
    live = load_ticker_live()
    out = {}
    for (code, model), rows in live.groupby(["code", "model"], sort=True):
        rows = rows.drop(columns=["code", "model"])
        rows = rows[rows.index < until.normalize()]  # 기준일 뒤 기록은 세지 않는다(날짜 지정 재계산)
        realtime = rows[realtime_mask(rows)]
        entry = {"title": f"{config.MULTI_TICKERS.get(code, (code,))[0]}({code}) · {model}", "code": code, "algorithm": model,
                 "n_backfilled": int(len(rows) - len(realtime))}
        if realtime.empty:
            entry.update(n_signals=0, n_matured=0, accuracy=None, drift={"status": "실시간 기록 없음"})
        else:
            try:
                close = ticker_close(code, until)
                entry.update(score_model(realtime["signal"], close, pd.Series(dtype=float), down_action=config.TICKER_DOWN_ACTION),
                             price_last_date=str(close.index.max().date()))
            except Exception as e:  # 한 종목 원본 문제로 성적표 전체가 멈추지 않게 한다. 판정(criteria)은 만들지 않는다
                entry.update(n_signals=int(len(realtime)), n_matured=None, accuracy=None,
                             drift={"status": f"채점 실패({type(e).__name__}: {str(e)[:80]})"})
        out[f"{code}|{model}"] = entry
    return out


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


def score_model(signals: pd.Series, samsung: pd.Series, inverse: pd.Series, down_action: str | None = None,
                horizon: int | None = None, start: pd.Timestamp | None = None) -> dict:
    """한 모델의 기록을 채점한다. signals 는 날짜별 신호(상승·보합·하락). down_action 은 판정에 쓰는 하락 행동(기본 config.DOWN_ACTION).
    samsung 은 채점 대상 종목의 종가다(옛 이름. 다른 종목은 그 종목 종가가 들어온다). 인버스 가격이 비면 인버스 규칙은 계산에서 빠진다.
    horizon 은 정답 기간(기본 운영 기간). 기준은 config.HORIZONS[horizon].
    start 를 주면 그날부터 현금으로 모의 매매를 시작한다(승격 비교용. 그날 기록이 없으면 포지션 유지 = 현금)."""
    down_action = down_action or config.DOWN_ACTION
    n = horizon or config.OPERATIONAL_HORIZON
    signals = signals.sort_index()
    labels = make_labels(samsung, n, config.HORIZONS[n])["label"]
    matured = signals[signals.index.isin(labels.index)]
    correct = matured == labels.reindex(matured.index)
    out = {"n_signals": int(len(signals)), "n_matured": int(len(matured)),
           "first_date": str(pd.Timestamp(start).date()) if start is not None else (str(signals.index.min().date()) if len(signals) else None),
           "accuracy": float(correct.mean()) if len(matured) else None,
           "direction": backtest.direction_hit_rate(matured, labels)}
    first = pd.Timestamp(start) if start is not None else (signals.index.min() if len(signals) else None)
    days = samsung.index[(samsung.index >= first)] if first is not None and len(signals) else samsung.index[:0]
    if len(days) >= 2:
        daily = signals.reindex(days).fillna(FLAT)  # 기록이 빠진 날은 포지션 유지
        by_rule, curves, sims = {}, {}, {}
        for rule in config.DOWN_ACTIONS:  # 가격이 모자란 규칙(예: 인버스 가격 조회 실패)은 빠지고 나머지는 계산한다
            try:
                sims[rule] = backtest.simulate(daily, samsung, inverse, rule)
                curves[rule] = sims[rule]["equity"]
                by_rule[rule] = backtest.summary(curves[rule])
            except ValueError:
                continue
        out.update(by_rule=by_rule, down_action=down_action,
                   hold=backtest.summary(backtest.buy_and_hold(samsung, days)), trading_days=int(len(days)))
        if down_action in curves:
            out.update(model=by_rule[down_action],
                       equity={str(d.date()): round(float(v), 6) for d, v in curves[down_action].items()})
            known = labels.reindex(matured.index)
            down = matured == DOWN
            held = (sims[down_action]["position"] == backtest.LONG).astype(float)
            out["criteria"] = criteria(len(matured), curves[down_action], backtest.buy_and_hold(samsung, days), held, samsung,
                                       hit=float((known[down] == DOWN).mean()) if down.any() else None,
                                       base=float((known == DOWN).mean()) if len(known) else None, kind="하락 신호 적중")
        else:
            out["model_note"] = "현재 규칙에 필요한 가격이 없어 모의 매매를 계산하지 못했다"
    out["drift"] = drift_check(matured, labels)
    return out


def score_weight_model(rows: pd.DataFrame, samsung: pd.Series, start: pd.Timestamp | None = None) -> dict:
    """목표 비중 기록(5일차 하락 위험 비교 조건)을 채점한다. 기록이 빠진 날은 앞 비중을 유지한다.
    start 를 주면 그날부터 계산하고 첫 기록 전 날은 비중 0(현금)이다(승격 비교용).
    위험 적중은 '비중 0%' 날 중 실제로 5거래일 안에 -3% 이하로 내려간 비율이다."""
    from src.pipeline.labels import make_risk_labels
    weights = rows["weight"].astype(float).sort_index()
    labels = make_risk_labels(samsung)
    matured = weights[weights.index.isin(labels.index)]
    high = matured[matured == config.RISK_WEIGHTS[2]]  # 위험 높음 비중
    out = {"n_signals": int(len(weights)), "n_matured": int(len(matured)), "accuracy": None,
           "first_date": str(pd.Timestamp(start).date()) if start is not None else (str(weights.index.min().date()) if len(weights) else None),
           "risk_hit": float(labels.loc[high.index, "risk"].mean()) if len(high) else None,
           "drift": {"status": "해당 없음(비중 모델)"}}
    first = pd.Timestamp(start) if start is not None else (weights.index.min() if len(weights) else None)
    days = samsung.index[samsung.index >= first] if first is not None and len(weights) else samsung.index[:0]
    if len(days) >= 2:
        daily = weights.reindex(days).ffill().fillna(0.0)
        sim = backtest.simulate_target(daily, samsung)
        model = backtest.summary(sim["equity"])
        risk_known = labels["risk"].reindex(matured.index)
        out["criteria"] = criteria(len(matured), sim["equity"], backtest.buy_and_hold(samsung, days), daily, samsung,
                                   hit=out["risk_hit"], base=float(risk_known.mean()) if len(risk_known) else None,
                                   kind="위험 높음(비중 0%) 날의 실제 위험")
        out.update(model=model, by_rule={"cash": model}, down_action="cash", trading_days=int(len(days)),
                   hold=backtest.summary(backtest.buy_and_hold(samsung, days)),
                   equity={str(d.date()): round(float(v), 6) for d, v in sim["equity"].items()})
    return out


def criteria(n_matured: int, equity: pd.Series, hold: pd.Series, weights: pd.Series, samsung: pd.Series,
             hit: float | None, base: float | None, kind: str) -> dict:
    """SERVICE.md 5.2(다른 종목은 5.6) 성공 기준. samsung 은 채점 대상 종목 종가(옛 이름). 확정 신호가 LIVE_PROMOTION_MIN_SIGNALS 개 미만이면 '참고'로만 계산한다.
    A: MDD ≤ 들고 있기 MDD 의 절반, 수익 ≥ 들고 있기의 절반(들고 있기 ≤ 0 이면 이상)
    B: 하락(위험) 신호 적중 > 같은 기간 실제 하락(위험) 비율
    C: 그달 들고 있기 ≥ UP_MONTH_MIN_RETURN 인 달의 보유일 ≥ UP_MONTH_MIN_HOLD (그런 달이 없으면 판정 없음)"""
    need = config.LIVE_PROMOTION_MIN_SIGNALS
    a = backtest.judge_a(equity, hold)
    share, months = backtest.up_month_hold(weights, samsung)
    return {"ready": n_matured >= need, "n_matured": n_matured, "needed": need,
            "status": "판정" if n_matured >= need else f"표본 부족({n_matured}/{need}, 참고)",
            "A": {"passed": bool(a["passed"]), "mdd_ok": bool(a["mdd_ok"]), "return_ok": bool(a["return_ok"])},
            "B": {"kind": kind, "hit": hit, "base": base,
                  "passed": None if hit is None or base is None else bool(hit > base)},
            "C": {"hold_share": None if pd.isna(share) else share, "up_months": months,
                  "passed": None if months == 0 else bool(share >= config.UP_MONTH_MIN_HOLD)}}


def same_period(model: str, rows: pd.DataFrame, base_rows: pd.DataFrame, samsung: pd.Series, inverse: pd.Series) -> tuple[dict, dict, dict]:
    """승격 비교용: 두 모델 첫 기록일 중 늦은 날부터 둘 다 현금에서 다시 채점한 (비교 모델, 운영 모델) 결과와 빠진 날 정보.
    구간 안에서 기록이 빠진 날은 성적표 규칙대로 포지션 유지다. gap 은 한쪽만 기록이 빠진 거래일 수와 가장 긴 연속 일수다."""
    start = max(rows.index.min(), base_rows.index.min())
    mine_rows, base_part = rows.loc[start:], base_rows.loc[start:]
    if model == config.LIVE_RISK_MODEL:
        mine = score_weight_model(mine_rows, samsung, start=start)
    else:
        mine = score_model(mine_rows["signal"], samsung, inverse, horizon=tools.model_horizon(model), start=start)
    base = score_model(base_part["signal"], samsung, inverse, start=start)  # 둘 다 같은 날 현금에서 시작(시작일 기록이 빠진 쪽은 그날 포지션 유지)
    days = samsung.index[samsung.index >= start]
    one_side = pd.Series(days.isin(mine_rows.index) != days.isin(base_part.index), index=days)
    runs = one_side.groupby((~one_side).cumsum()).sum()
    gap = {"start": str(start.date()), "days": int(len(days)), "one_sided": int(one_side.sum()),
           "longest_run": int(runs.max()) if len(runs) else 0}
    return mine, base, gap


def promotion(entry: dict, base: dict, model: str, gap: dict | None = None) -> dict:
    """SERVICE.md 5.3 비교 모델 승격 기준: 장부 등록, 확정 신호 40개 이상, 같은 기간 운영 모델보다 칼마가 높고 MDD 가 더 깊지 않음.
    entry·base 는 같은 구간(same_period)으로 다시 계산한 결과다. gap 에서 한쪽만 기록이 빠진 날이 많으면 보류한다.
    셋을 모두 만족해도 운영 변경은 사용자가 정한다."""
    registered, reg_note = model in config.LIVE_REGISTERED_MODELS, None
    if tools.model_horizon(model) in config.DAY11_LEDGER:  # 기간별 모델은 장부 H1 ~ H4 를 과거 판정에서 통과했을 때만 승격 검토(SERVICE.md 5.7)
        registered, reg_note = horizon_registered(tools.model_horizon(model))
    ready = entry.get("n_matured", 0) >= config.LIVE_PROMOTION_MIN_SIGNALS
    if "model" not in entry or "model" not in base:
        return {"registered": registered, "registration_note": reg_note, "ready": ready, "eligible": False, "note": "모의 매매 결과가 없다"}
    if entry.get("first_date") != base.get("first_date"):  # 같은 기간이 아니면 비교하지 않는다
        return {"registered": registered, "registration_note": reg_note, "ready": ready, "eligible": False,
                "note": f"시작일이 운영 모델과 달라({entry.get('first_date')} vs {base.get('first_date')}) 판정 보류"}
    if gap and (gap["one_sided"] > config.PROMOTION_MAX_GAP_SHARE * gap["days"] or gap["longest_run"] >= config.PROMOTION_MAX_GAP_RUN):
        return {"registered": registered, "registration_note": reg_note, "ready": ready, "eligible": False, "gap": gap,
                "note": f"{gap['start']} 부터 한쪽만 기록이 빠진 날 {gap['one_sided']}일(가장 긴 연속 {gap['longest_run']}일)이라 판정 보류"}
    if pd.isna(entry["model"]["calmar"]) or pd.isna(base["model"]["calmar"]):
        return {"registered": registered, "registration_note": reg_note, "ready": ready, "eligible": False, "note": "MDD 가 0 이라 칼마를 비교할 수 없다(판정 불가)"}
    better = (entry["model"]["calmar"] > base["model"]["calmar"] + backtest.EPS
              and abs(entry["model"]["mdd"]) <= abs(base["model"]["mdd"]) + backtest.EPS)  # judge_day11 과 같은 오차 허용
    return {"registered": registered, "registration_note": reg_note, "ready": ready, "better_than_operational": bool(better),
            "eligible": bool(registered and ready and better),
            "calmar": entry["model"]["calmar"], "operational_calmar": base["model"]["calmar"], "gap": gap,
            "compare_from": entry.get("first_date"), "n_matured_same_period": entry.get("n_matured")}


def horizon_registered(n: int) -> tuple[bool, str | None]:
    """기간 N 이 과거 판정(장부 H1 ~ H4, config.DAY11_JUDGED_PATH)을 통과했는지와, 아니면 그 이유."""
    hid = config.DAY11_LEDGER[n]
    path = Path(config.DAY11_JUDGED_PATH)
    if not path.exists():
        return False, f"장부 {hid} 판정 전"
    try:  # 기록이 깨져도 성적표·메일은 만든다
        verdict = json.loads(path.read_text(encoding="utf-8")).get("verdict", {}).get(hid)
    except (OSError, ValueError, AttributeError):
        return False, f"장부 {hid} 판정 기록 읽기 실패"
    if not verdict or not verdict.get("passed"):
        return False, f"장부 {hid} 과거 판정 불통과"
    return True, None


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
        extra.update({tools.horizon_model(n): f"{config.HORIZON_LABELS[n]}({n}거래일 ±{config.HORIZONS[n]:.1%}) 신호(기간별 참고, 장부 {config.DAY11_LEDGER.get(n, '-')})"
                      for n in config.HORIZONS if n != config.OPERATIONAL_HORIZON})
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
                entry.update(score_model(realtime["signal"], samsung, inverse, horizon=tools.model_horizon(model)))
            board["models"][model] = entry
        base_rows = live[live["model"] == "base"]
        base_rows = base_rows[realtime_mask(base_rows)]
        for model, entry in board["models"].items():
            if model in ("base", "base_v1"):
                continue
            rows = live[live["model"] == model]
            rows = rows[realtime_mask(rows)]
            if rows.empty or base_rows.empty:
                entry["promotion"] = promotion(entry, {}, model)
                continue
            mine, same, gap = same_period(model, rows, base_rows, samsung, inverse)  # 두 모델 모두 기록이 있는 첫날부터 다시 계산
            entry["promotion"] = promotion(mine, same, model, gap)
    try:  # 다른 종목 기록 문제로 삼성전자 성적표·알림이 멈추지 않게 한다
        board["tickers"] = score_tickers(today)
    except Exception as e:
        board["tickers"] = {}
        board["tickers_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    path = Path(config.SCOREBOARD_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return board


def summary_lines(board: dict) -> list[str]:
    """알림·화면용 한 줄 요약들(삼성전자 운영·비교 모델)."""
    lines = [entry_line(s) for s in board.get("models", {}).values()]
    if lines and board.get("price_last_date"):
        lines.append(f"(가격 마지막 날 {board['price_last_date']} 기준)")
    return lines or [board.get("note", "기록 없음")]


def ticker_lines(board: dict) -> list[str]:
    """다른 종목 성적표 요약(종목·알고리즘마다 한 줄)."""
    if board.get("tickers_error"):
        return [f"다른 종목 성적표 계산 실패: {board['tickers_error']}"]
    return [entry_line(s) for s in board.get("tickers", {}).values()] or ["아직 실시간 기록이 없다"]


def entry_line(s: dict) -> str:
    """성적표 항목 하나의 한 줄 요약."""
    part = f"{s['title']}: 실시간 기록 {s['n_signals']}일, 정답 확정 {'-' if s['n_matured'] is None else s['n_matured']}일"
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
    if "criteria" in s:
        c = s["criteria"]
        mark = lambda v: "-" if v is None else ("통과" if v else "실패")
        part += f" · 성공 기준[{c['status']}] A {mark(c['A']['passed'])}, B {mark(c['B']['passed'])}, C {mark(c['C']['passed'])}"
    if "promotion" in s:
        p = s["promotion"]
        part += (" · 승격 검토 대상" if p["eligible"] else
                 " · 승격 조건 " + ((p.get("registration_note") or "장부 미등록") if not p["registered"] else "미충족(표본 부족)" if not p["ready"]
                                 else f"보류({p['note']})" if "note" in p else "미충족(운영보다 낫지 않음)"))
    return part


def main():
    """성적표를 만들고 요약을 출력한다."""
    board = build()
    print(config.SCOREBOARD_PATH)
    print("\n".join(summary_lines(board)))
    print("\n".join(ticker_lines(board)))


if __name__ == "__main__":
    main()
