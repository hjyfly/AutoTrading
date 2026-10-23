import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, llm
from src.pipeline.labels import DOWN, FLAT, UP

LADDER = [DOWN, FLAT, UP]


def adjust(signal: str, consensus: float, confidence: float, min_score: float | None = None, min_conf: float | None = None) -> str:
    """뉴스 합의 점수가 강하면 숫자 모델 신호를 한 단계 올리거나 내린다(DAY3_PLAN.md 3절 방법 A).

    합의 ≥ +NEWS_ADJUST_SCORE 이고 확신도 ≥ NEWS_ADJUST_MIN_CONF 면 한 단계 위(하락→보합→상승),
    합의 ≤ -NEWS_ADJUST_SCORE 이고 확신도가 같으면 한 단계 아래. 그 밖에는 그대로 둔다.
    min_score·min_conf 는 조건 비교 실험에서만 바꾼다.
    """
    min_score = config.NEWS_ADJUST_SCORE if min_score is None else min_score
    min_conf = config.NEWS_ADJUST_MIN_CONF if min_conf is None else min_conf
    if pd.isna(consensus) or pd.isna(confidence) or confidence < min_conf:
        return signal
    i = LADDER.index(signal)
    if consensus >= min_score:
        i = min(i + 1, len(LADDER) - 1)
    elif consensus <= -min_score:
        i = max(i - 1, 0)
    return LADDER[i]


def news_signals(numeric: pd.DataFrame, scores: pd.DataFrame, field: str | None = None, window: int | None = None,
                 risk_veto: int | None = None, min_score: float | None = None, min_conf: float | None = None) -> pd.DataFrame:
    """숫자 모델 신호에 같은 날짜 t 의 뉴스 점수를 붙여 보정한다.

    뉴스 점수는 t일 15:15 까지의 기사로 만든 값이라 하루 늦추지 않고 t 행에 그대로 붙인다.
    채점하지 않은 날은 보정하지 않는다(앞뒤 값으로 채우지 않는다).
    아래 인자는 조건 비교 실험용이고, 주지 않으면 config 기본값(합의 점수, 그날 하루, 위험 거부 없음)을 쓴다.
    - field: 보정에 쓸 점수 열(config.NEWS_SCORE_FIELDS). 분류별 점수는 채점 버전 2 에만 있다
    - window: 최근 k개 채점일(t 포함, t 뒤는 없음)의 점수·확신도 평균. 채점이 빠진 날이 있으면 그만큼 더 오래된 날까지 간다
    - risk_veto: 하락 위험(news_downside_risk)이 이 값 이상이면 '상승' 을 '보합' 으로 낮춘다
    """
    field = field or config.NEWS_SCORE_FIELD
    window = config.NEWS_WINDOW_DAYS if window is None else window
    risk_veto = config.NEWS_RISK_VETO if risk_veto is None else risk_veto
    if field not in scores.columns or scores[field].isna().all():
        raise ValueError(f"점수 표에 '{field}' 값이 없다(분류별 점수는 채점 버전 2 에만 있다)")
    if risk_veto is not None and ("news_downside_risk" not in scores.columns or scores["news_downside_risk"].isna().all()):
        raise ValueError("점수 표에 하락 위험 값이 없다(채점 버전 2 에만 있다)")
    s = scores.set_index("signal_date").sort_index()
    s = pd.DataFrame({
        "news_consensus": s["news_consensus"], "news_consensus_conf": s["news_consensus_conf"], "no_news": s["no_news"],
        "news_used": s[field].rolling(window, min_periods=1).mean(),  # 과거 쪽으로만 굴리는 평균
        "news_used_conf": s["news_consensus_conf"].rolling(window, min_periods=1).mean(),
        **({"news_downside_risk": s["news_downside_risk"]} if risk_veto is not None else {}),
    })
    out = numeric[["signal", "confidence"]].join(s, how="left")
    out["numeric_signal"] = out["signal"]
    out["signal"] = [adjust(sig, c, conf, min_score, min_conf) for sig, c, conf in
                     zip(out["numeric_signal"], out["news_used"], out["news_used_conf"])]
    if risk_veto is not None:
        out.loc[(out["signal"] == UP) & (out["news_downside_risk"] >= risk_veto), "signal"] = FLAT
    out["news_scored"] = out["news_consensus"].notna()
    out["adjusted"] = out["signal"] != out["numeric_signal"]
    return out


def load_matching(numeric: pd.DataFrame | None = None) -> pd.DataFrame:
    """저장된 뉴스 보정 신호 중 보정 전 숫자 신호가 지금 운영 숫자 신호와 같은 날만 돌려준다.

    운영 지표 묶음을 바꾼 뒤 아직 다시 만들지 않은 보정 신호(옛 숫자 모델 기준)가 화면·도구에 섞이지 않게 한다.
    """
    path = Path(config.PROCESSED_DIR) / "signals_news.parquet"
    if not path.exists():
        return pd.DataFrame()
    news = pd.read_parquet(path)
    numeric = pd.read_parquet(config.SIGNALS_NUMERIC_PATH) if numeric is None else numeric
    same = numeric["signal"].reindex(news.index) == news["numeric_signal"]
    return news[same]


# ---- 9일차: 기사 피처 → 일별 피처 → V1·V2 로지스틱 워크포워드, A1 하락 위험 경보 (DAY9_PLAN.md 3.2, 장부 V1·V2·A1) ----

FEATURE_CATS = ["samsung_direct", "semi_ai", "macro"]
NEWS_FEATURES = ([f"f_{c}" for c in FEATURE_CATS] + [f"f_{c}_new" for c in FEATURE_CATS] +
                 ["f_total", "f_risk_max", "f_risk_n2", "f_risk_sum", "f_mag3", "f_relevant", "f_volume_z"])
CLASSES = [UP, FLAT, DOWN]


def load_article_features(dates=None) -> pd.DataFrame:
    """저장된 기사 피처(data/scores/features)를 기사 한 줄씩 모은다. 지금 프롬프트 버전·모델 것만."""
    rows = []
    for path in sorted(Path(config.FEATURES_DIR).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("prompt_version") != config.FEATURE_PROMPT_VERSION or rec.get("model") != llm.scoring_model_id():
            continue
        if dates is not None and rec["signal_date"] not in dates:
            continue
        day = {"signal_date": pd.Timestamp(rec["signal_date"]), "n_collected": rec.get("n_collected")}
        rows += [{**day, **a} for a in rec["articles"]]
        if not rec["articles"]:  # 기사가 없던 날도 일별 표에 남긴다
            rows.append(day)
    return pd.DataFrame(rows)


def news_volume(days: pd.DatetimeIndex) -> pd.Series:
    """신호일마다 모은 전체 기사 수(그날 15:15 까지 모인 것, 중복 포함 = 보도량)."""
    path = Path(config.PROCESSED_DIR) / "news.parquet"
    if not path.exists():
        return pd.Series(np.nan, index=days)
    counts = pd.read_parquet(path, columns=["signal_date"])["signal_date"].value_counts()
    return counts.reindex(days).fillna(0).astype(float)


def daily_features(articles: pd.DataFrame, volume: pd.Series | None = None) -> pd.DataFrame:
    """기사 피처를 신호일 단위로 모은다. 그날 기사만 쓰고, 보도량 z 는 그날까지의 지난 NEWS_VOLUME_WINDOW 신호일로 계산한다."""
    days = pd.DatetimeIndex(sorted(articles["signal_date"].unique()))
    out = pd.DataFrame(index=days, columns=NEWS_FEATURES, dtype=float).fillna(0.0)
    a = articles.dropna(subset=["direction"]) if "direction" in articles else articles.iloc[0:0]
    if len(a):
        a = a.assign(w=a["direction"] * a["magnitude"] * a["relevance"], rel=a["relevance"] >= config.ALERT_EVIDENCE_MIN_RELEVANCE)
        g = a.groupby("signal_date")
        for c in FEATURE_CATS:
            out[f"f_{c}"] = a[a["category"] == c].groupby("signal_date")["w"].sum().reindex(days).fillna(0)
            out[f"f_{c}_new"] = a[(a["category"] == c) & (a["novelty"] == "new")].groupby("signal_date")["w"].sum().reindex(days).fillna(0)
        out["f_total"] = out[[f"f_{c}" for c in FEATURE_CATS]].sum(axis=1)
        out["f_risk_max"] = g["downside_risk"].max().reindex(days).fillna(0)
        out["f_risk_n2"] = a[(a["downside_risk"] >= 2) & a["rel"]].groupby("signal_date").size().reindex(days).fillna(0)
        out["f_risk_sum"] = (a["downside_risk"] * a["relevance"]).groupby(a["signal_date"]).sum().reindex(days).fillna(0)
        out["f_mag3"] = a[(a["magnitude"] == 3) & a["rel"] & (a["category"] != "irrelevant")].groupby("signal_date").size().reindex(days).fillna(0)
        out["f_relevant"] = a[a["rel"] & (a["category"] != "irrelevant")].groupby("signal_date").size().reindex(days).fillna(0)
    vol = news_volume(days) if volume is None else volume.reindex(days)
    if volume is None and "n_collected" in articles:  # 추출 때 남긴 보도량(실시간 채점일은 news.parquet 에 없다)
        recorded = articles.groupby("signal_date")["n_collected"].max().reindex(days)
        vol = recorded.where(recorded.notna(), vol).astype(float)
    roll = vol.rolling(config.NEWS_VOLUME_WINDOW, min_periods=10)
    out["f_volume_z"] = ((vol - roll.mean()) / roll.std()).replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def walk_forward_logit(X: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """월 첫 신호일마다 다시 학습하는 로지스틱 회귀 3분류(확장 창). 학습 행은 정답 기간(future_date)이 재학습일 전에 끝난 것만 쓰고,
    표준화도 학습 행으로만 맞춘다. 학습 행이 NEWS_MODEL_MIN_ROWS 보다 적은 달은 신호를 내지 않는다."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = X.dropna()
    data = X.join(labels[["label", "future_date"]], how="left")
    rows = []
    for _, block in X.groupby(X.index.to_period("M")):
        start = block.index.min()
        train = data[(data.index < start) & data["future_date"].notna() & (data["future_date"] < start)]
        if len(train) < config.NEWS_MODEL_MIN_ROWS or train["label"].nunique() < 2:
            continue
        scaler = StandardScaler().fit(train[X.columns])
        model = LogisticRegression(C=config.NEWS_MODEL_C, max_iter=1000).fit(scaler.transform(train[X.columns]), train["label"])
        proba = pd.DataFrame(model.predict_proba(scaler.transform(block)), index=block.index, columns=model.classes_)
        proba = proba.reindex(columns=CLASSES, fill_value=0.0)
        for t, p in proba.iterrows():
            rows.append({"date": t, "signal": p.idxmax(), "confidence": float(p.max()), **{f"p_{c}": float(p[c]) for c in CLASSES},
                         "retrained_on": start, "train_rows": len(train), "train_last_future_date": train["future_date"].max()})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["signal", "confidence"])


def news_models(daily: pd.DataFrame, numeric: pd.DataFrame, labels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """V1(뉴스 피처만)과 V2(운영 숫자 모델 확률 + 뉴스 피처) 워크포워드 신호."""
    v1 = walk_forward_logit(daily[NEWS_FEATURES], labels)
    both = daily[NEWS_FEATURES].join(numeric[["p_상승", "p_하락"]].rename(columns=lambda c: f"num_{c}"), how="inner")
    v2 = walk_forward_logit(both, labels)
    return {"V1": v1, "V2": v2}


def v1_signal(t: pd.Timestamp, close: pd.Series, articles: pd.DataFrame | None = None) -> dict | None:
    """t 일 V1(뉴스만 로지스틱) 참고 지표. t 까지의 기사 피처와 t 전 종가만으로 과거 검증과 같은 월 재학습 규칙을 그대로 돌린다.
    확인 구간 판정 통과(장부 V1, 2026-10-23)로 참고 표시만 한다. 운영 신호는 바꾸지 않는다."""
    from src.pipeline.labels import make_labels
    t = pd.Timestamp(t)
    articles = load_article_features() if articles is None else articles
    if articles.empty:
        return None
    articles = articles[articles["signal_date"] <= t]
    if t not in set(articles["signal_date"]):
        return None
    out = walk_forward_logit(daily_features(articles)[NEWS_FEATURES], make_labels(close[close.index < t]))
    if t not in out.index:
        return None
    r = out.loc[t]
    return {"signal": r["signal"], "confidence": float(r["confidence"]), "score": float(r[f"p_{UP}"] - r[f"p_{DOWN}"]),
            **{f"p_{c}": float(r[f"p_{c}"]) for c in CLASSES}, "retrained_on": str(pd.Timestamp(r["retrained_on"]).date()),
            "train_rows": int(r["train_rows"])}


def _event_key(article: dict) -> str:
    """경보 쿨다운용 사건 키: 사건 종류 + 공백·기호를 지운 제목 앞부분."""
    return article.get("event_type", "") + ":" + re.sub(r"[\W_]+", "", str(article.get("title", "")).lower())[: config.SELECT_DEDUP_PREFIX]


def alert_evidence(day_articles: pd.DataFrame) -> list[dict]:
    """근거 기사: 하락 위험·관련도 기준을 넘는 기사를 위험 × 관련도 × 새로움 가중 × log(1 + 보도량) 순으로, 같은 사건은 하나만."""
    if day_articles.empty or "downside_risk" not in day_articles:
        return []
    c = day_articles[(day_articles["downside_risk"] >= config.ALERT_EVIDENCE_MIN_RISK)
                     & (day_articles["relevance"] >= config.ALERT_EVIDENCE_MIN_RELEVANCE)].copy()
    if c.empty:
        return []
    c["rank"] = (c["downside_risk"] * c["relevance"] * c["novelty"].map(config.NOVELTY_WEIGHTS).fillna(0)
                 * np.log1p(c.get("cluster_size", pd.Series(1, index=c.index)).fillna(1)))
    c["key"] = [_event_key(r) for r in c.to_dict("records")]
    c = c.sort_values("rank", ascending=False).drop_duplicates("key").head(config.ALERT_TOP_K)
    keep = ["title", "url", "published_kst", "source", "event_type", "downside_risk", "relevance", "novelty", "one_line_summary", "reasoning", "key"]
    return [{k: r.get(k) for k in keep} for r in c.to_dict("records")]


def risk_alerts(articles: pd.DataFrame, daily: pd.DataFrame, consensus_risk: pd.Series | None = None) -> pd.DataFrame:
    """A1 경보를 신호일 순서대로 판단한다(그날 기사와 그날까지의 일별 피처만 쓴다). 쿨다운 안의 같은 사건은 등급이 오를 때만 다시 알린다."""
    risk_sum = daily["f_risk_sum"]
    roll = risk_sum.rolling(config.NEWS_VOLUME_WINDOW, min_periods=10)
    z = ((risk_sum - roll.mean()) / roll.std()).replace([np.inf, -np.inf], np.nan)
    days = list(daily.index)
    rows, recent = [], []  # recent: (신호일 순번, 사건 키, 등급)
    rank = {"MEDIUM": 1, "HIGH": 2}
    for i, t in enumerate(days):
        day = articles[articles["signal_date"] == t] if "downside_risk" in articles else articles.iloc[0:0]
        high = day[(day["relevance"] >= config.ALERT_HIGH_RELEVANCE) & (day["downside_risk"] == 3) & (day["novelty"] == "new")] if len(day) else day
        reasons = []
        if len(high):
            reasons.append(f"새 고위험 기사 {len(high)}건(관련도 ≥ {config.ALERT_HIGH_RELEVANCE}, 하락 위험 3)")
        if pd.notna(z.get(t)) and z[t] >= config.ALERT_RISK_Z:
            reasons.append(f"하락 위험 합 60일 z {z[t]:.1f}")
        cr = None if consensus_risk is None else consensus_risk.get(t)
        if cr is not None and pd.notna(cr) and cr >= config.ALERT_CONSENSUS_RISK:
            reasons.append(f"채점자 종합 하락 위험 {int(cr)}/{config.RISK_MAX}")
        if not reasons:
            continue
        level = "HIGH" if len(high) else "MEDIUM"
        evidence = alert_evidence(day)
        keys = {e["key"] for e in evidence} or {f"day:{t.date()}"}
        blocked = [r for r in recent if i - r[0] <= config.ALERT_COOLDOWN_DAYS and r[1] in keys and rank[r[2]] >= rank[level]]
        if blocked and {r[1] for r in blocked} >= keys:  # 근거 사건이 모두 최근에 같은 등급 이상으로 알린 것이면 다시 알리지 않는다
            continue
        recent += [(i, k, level) for k in keys]
        rows.append({"date": t, "level": level, "reasons": reasons, "risk_z": None if pd.isna(z.get(t)) else round(float(z[t]), 2),
                     "evidence": evidence})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["level", "reasons", "risk_z", "evidence"])


def render_alert(t: pd.Timestamp, alert: dict, status: str = "") -> str:
    """경보 메일 문구. 사실은 추출 결과(제목·한 줄 요약·근거) 안에서만 쓴다."""
    lines = [f"[하락 위험 경보 · {alert['level']}] {pd.Timestamp(t).date()} {config.SIGNAL_CUTOFF_KST} 기준 삼성전자" + (f" ({status})" if status else ""),
             "■ 이유: " + " / ".join(alert["reasons"])]
    if alert["evidence"]:
        lines.append("■ 근거 기사")
        for k, e in enumerate(alert["evidence"], start=1):
            lines.append(f"  {k}) {e['title']} · {e['published_kst']} · {e['source']}")
            summary = str(e["one_line_summary"]).replace(config.ANONYMIZED_NAME, "삼성전자")  # 추출은 익명으로 했다
            lines.append(f"     → {summary} (하락 위험 {e['downside_risk']}/3, {e['novelty']})")
            if e.get("url"):
                lines.append(f"     {e['url']}")
    lines.append("※ 뉴스 기반 자동 분석이며 투자 권유가 아닙니다.")
    return "\n".join(lines)


def record_alert(t: pd.Timestamp, alert: dict):
    """경보 이력을 ALERTS_PATH(jsonl)에 덧붙인다. 같은 날 기록이 있으면 덧붙이지 않는다."""
    path = Path(config.ALERTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    day = str(pd.Timestamp(t).date())
    if path.exists() and any(json.loads(line).get("date") == day for line in path.read_text(encoding="utf-8").splitlines() if line.strip()):
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": day, **alert}, ensure_ascii=False, default=str) + "\n")


def main():
    """뉴스 구간의 숫자 모델 신호를 보정해 data/processed/signals_news.parquet 에 저장한다."""
    numeric = pd.read_parquet(config.SIGNALS_NUMERIC_PATH).loc[config.NEWS_START:config.NEWS_END]
    scores = pd.read_parquet(config.SCORES_PATH)
    models = set(scores["model"])
    if models != {llm.scoring_model_id()}:  # 유출 테스트를 통과한 지금 모델의 점수만 쓴다
        raise ValueError(f"점수 표의 모델 {sorted(models)} 이 현재 채점 모델 {llm.scoring_model_id()} 과 다르다")
    out = news_signals(numeric, scores)
    path = Path(config.PROCESSED_DIR) / "signals_news.parquet"
    out.to_parquet(path)
    print(path, len(out), f"뉴스 채점된 날 {int(out['news_scored'].sum())}, 보정된 날 {int(out['adjusted'].sum())}")


if __name__ == "__main__":
    main()
