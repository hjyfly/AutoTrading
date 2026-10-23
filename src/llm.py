import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings

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


def check_limit():
    """누적 비용이 한도에 도달했으면 예외를 던진다."""
    spent = total_cost()
    if spent >= config.COST_LIMIT_POC_USD:
        raise CostLimitExceeded(f"누적 비용 ${spent:.2f} 가 한도 ${config.COST_LIMIT_POC_USD:.2f} 에 도달했다")


def record_cost(role: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """호출 한 번의 토큰과 비용을 cost_log.csv 에 한 줄 남기고 비용을 돌려준다. 파일이 없으면 머리글부터 쓴다."""
    price = get_price(model)
    cost = (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
    path = Path(config.COST_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COST_LOG_FIELDS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": f"{cost:.6f}",
        })
    return cost


class CostTrackedEmbeddings(Embeddings):
    """Bedrock 임베딩을 감싸 호출마다 비용을 기록하고 한도를 확인한다. 응답에 토큰 수가 없어 글자 수로 넉넉히 추정한다."""

    def __init__(self, model: str, region: str):
        self.model = model
        self.inner = BedrockEmbeddings(model_id=model, region_name=region)
        get_price(model)  # 가격표에 없으면 여기서 막는다

    def _track(self, texts: list[str]):
        check_limit()
        record_cost("embedding", self.model, sum(len(t) for t in texts), 0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서 여러 개를 임베딩한다."""
        self._track(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """검색어 하나를 임베딩한다."""
        self._track([text])
        return self.inner.embed_query(text)


def aws_region() -> str:
    """채점·에이전트·임베딩 호출 리전. .env 의 AWS_DEFAULT_REGION, 없으면 config 기본값."""
    return os.environ.get("AWS_DEFAULT_REGION") or config.AWS_REGION_DEFAULT


def get_embeddings() -> CostTrackedEmbeddings:
    """RAG 색인·검색용 임베딩. 모델은 .env 의 BEDROCK_EMBEDDING_MODEL_ID, 없으면 config 기본값."""
    return CostTrackedEmbeddings(os.environ.get("BEDROCK_EMBEDDING_MODEL_ID") or config.EMBEDDING_MODEL_DEFAULT, aws_region())


class CostTracker(BaseCallbackHandler):
    """LLM 호출마다 토큰과 비용을 cost_log.csv 에 기록하고, 누적 비용이 한도에 도달하면 다음 호출을 막는다."""

    raise_error = True

    def __init__(self, role: str, model: str):
        self.role = role
        self.model = model
        self.price = get_price(model)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """호출 전에 누적 비용을 확인하고, 한도에 도달했으면 예외를 던진다."""
        check_limit()

    def on_llm_end(self, response, **kwargs):
        """응답의 토큰 사용량으로 비용을 계산해 cost_log.csv 에 한 줄씩 남긴다."""
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                record_cost(self.role, self.model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


def get_judge_llm() -> ChatBedrockConverse:
    """평가용 judge LLM 을 만든다. 모델·리전·temperature 는 config 상수로 고정한다."""
    return ChatBedrockConverse(
        model=config.JUDGE_MODEL,
        region_name=config.JUDGE_REGION,
        temperature=config.JUDGE_TEMPERATURE,
        callbacks=[CostTracker("judge", config.JUDGE_MODEL)],
    )


def scoring_model_id() -> str:
    """뉴스 채점에 쓰는 모델 ID. .env 의 BEDROCK_MODEL_ID 가 없으면 config 기본값."""
    return os.environ.get("BEDROCK_MODEL_ID", config.SCORING_MODEL_DEFAULT)


def get_agent_llm() -> ChatBedrockConverse:
    """/query 질의응답 에이전트용 LLM. 채점과 같은 모델이며 비용은 'agent' 로 기록한다."""
    model = scoring_model_id()
    return ChatBedrockConverse(model=model, region_name=aws_region(), callbacks=[CostTracker("agent", model)])


def get_scoring_llm() -> ChatBedrockConverse:
    """뉴스 채점용 LLM 을 만든다. 공급자와 모델은 .env 로 고르고, 없으면 config 기본값(Bedrock, Sonnet 4.6)."""
    provider = os.environ.get("LLM_PROVIDER") or config.LLM_PROVIDER_DEFAULT
    if provider == "bedrock":
        # Sonnet 5.5 로 옮기면 temperature 를 기본값이 아닌 값으로 보낼 때 400 오류가 나므로 설정하지 않는다
        model = scoring_model_id()
        return ChatBedrockConverse(
            model=model,
            region_name=aws_region(),
            callbacks=[CostTracker("scoring", model)],
        )
    raise ValueError(f"지원하지 않는 LLM_PROVIDER: {provider}")
