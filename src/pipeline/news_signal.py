from pathlib import Path

import pandas as pd

from src import config, llm
from src.pipeline.labels import DOWN, FLAT, UP

LADDER = [DOWN, FLAT, UP]


def adjust(signal: str, consensus: float, confidence: float) -> str:
    """뉴스 합의 점수가 강하면 숫자 모델 신호를 한 단계 올리거나 내린다(DAY3_PLAN.md 3절 방법 A).

    합의 ≥ +NEWS_ADJUST_SCORE 이고 확신도 ≥ NEWS_ADJUST_MIN_CONF 면 한 단계 위(하락→보합→상승),
    합의 ≤ -NEWS_ADJUST_SCORE 이고 확신도가 같으면 한 단계 아래. 그 밖에는 그대로 둔다.
    """
    if pd.isna(consensus) or pd.isna(confidence) or confidence < config.NEWS_ADJUST_MIN_CONF:
        return signal
    i = LADDER.index(signal)
    if consensus >= config.NEWS_ADJUST_SCORE:
        i = min(i + 1, len(LADDER) - 1)
    elif consensus <= -config.NEWS_ADJUST_SCORE:
        i = max(i - 1, 0)
    return LADDER[i]


def news_signals(numeric: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """숫자 모델 신호에 같은 날짜 t 의 뉴스 점수를 붙여 보정한다.

    뉴스 점수는 t일 15:15 까지의 기사로 만든 값이라 하루 늦추지 않고 t 행에 그대로 붙인다.
    채점하지 않은 날은 보정하지 않는다(앞뒤 값으로 채우지 않는다).
    """
    s = scores.set_index("signal_date")[["news_consensus", "news_consensus_conf", "no_news"]]
    out = numeric[["signal", "confidence"]].join(s, how="left")
    out["numeric_signal"] = out["signal"]
    out["signal"] = [adjust(sig, c, conf) for sig, c, conf in
                     zip(out["numeric_signal"], out["news_consensus"], out["news_consensus_conf"])]
    out["news_scored"] = out["news_consensus"].notna()
    out["adjusted"] = out["signal"] != out["numeric_signal"]
    return out


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
