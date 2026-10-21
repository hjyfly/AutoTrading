import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import BaseCallbackHandler

from src import config

load_dotenv()

COST_LOG_FIELDS = ["timestamp", "role", "model", "input_tokens", "output_tokens", "cost_usd"]


class CostLimitExceeded(RuntimeError):
    """누적 LLM 비용이 한도에 도달했을 때 던지는 예외."""


def get_price(model: str) -> tuple[float, float]:
    """모델 ID 에 맞는 100만 토큰당 (입력, 출력) 가격을 돌려준다. 가격표에 없으면 예외를 던진다."""
    for key, price in config.LLM_PRICES_PER_MTOK.items():
        if key in model:
            return price
    raise ValueError(f"가격표에 없는 모델: {model}. config.LLM_PRICES_PER_MTOK 에 추가한다")


def total_cost() -> float:
    """cost_log.csv 에 기록된 누적 비용(달러)을 돌려준다."""
    path = Path(config.COST_LOG_PATH)
    if not path.exists():
        return 0.0
    with path.open(encoding="utf-8", newline="") as f:
        return sum(float(row["cost_usd"]) for row in csv.DictReader(f))


class CostTracker(BaseCallbackHandler):
    """LLM 호출마다 토큰과 비용을 cost_log.csv 에 기록하고, 누적 비용이 한도에 도달하면 다음 호출을 막는다."""

    raise_error = True

    def __init__(self, role: str, model: str):
        self.role = role
        self.model = model
        self.price = get_price(model)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """호출 전에 누적 비용을 확인하고, 한도에 도달했으면 예외를 던진다."""
        spent = total_cost()
        if spent >= config.COST_LIMIT_POC_USD:
            raise CostLimitExceeded(f"누적 비용 ${spent:.2f} 가 한도 ${config.COST_LIMIT_POC_USD:.2f} 에 도달했다")

    def on_llm_end(self, response, **kwargs):
        """응답의 토큰 사용량으로 비용을 계산해 cost_log.csv 에 한 줄씩 남긴다."""
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cost = (input_tokens * self.price[0] + output_tokens * self.price[1]) / 1_000_000
                self._write(input_tokens, output_tokens, cost)

    def _write(self, input_tokens: int, output_tokens: int, cost: float):
        """cost_log.csv 에 한 줄을 추가한다. 파일이 없으면 머리글부터 쓴다."""
        path = Path(config.COST_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COST_LOG_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "role": self.role,
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": f"{cost:.6f}",
            })


def get_judge_llm() -> ChatBedrockConverse:
    """평가용 judge LLM 을 만든다. 모델·리전·temperature 는 config 상수로 고정한다."""
    return ChatBedrockConverse(
        model=config.JUDGE_MODEL,
        region_name=config.JUDGE_REGION,
        temperature=config.JUDGE_TEMPERATURE,
        callbacks=[CostTracker("judge", config.JUDGE_MODEL)],
    )


def get_scoring_llm() -> ChatBedrockConverse:
    """뉴스 채점용 LLM 을 만든다. 공급자와 모델은 .env 로 고른다."""
    provider = os.environ.get("LLM_PROVIDER", "bedrock")
    if provider == "bedrock":
        # Sonnet 5.5 로 옮기면 temperature 를 기본값이 아닌 값으로 보낼 때 400 오류가 나므로 설정하지 않는다
        model = os.environ.get("BEDROCK_MODEL_ID", config.SCORING_MODEL_DEFAULT)
        return ChatBedrockConverse(
            model=model,
            region_name=os.environ["AWS_DEFAULT_REGION"],
            callbacks=[CostTracker("scoring", model)],
        )
    raise ValueError(f"지원하지 않는 LLM_PROVIDER: {provider}")
