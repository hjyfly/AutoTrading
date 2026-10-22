from pathlib import Path

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
