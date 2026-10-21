import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src import config, llm


@pytest.fixture(autouse=True)
def temp_cost_log(tmp_path, monkeypatch):
    """비용 기록을 임시 파일로 돌려 실제 cost_log.csv 를 건드리지 않는다."""
    monkeypatch.setattr(config, "COST_LOG_PATH", str(tmp_path / "cost_log.csv"))


def make_result(input_tokens: int, output_tokens: int) -> LLMResult:
    """토큰 사용량이 담긴 가짜 LLM 응답을 만든다."""
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_비용을_가격표대로_기록한다():
    tracker = llm.CostTracker("scoring", "us.anthropic.claude-sonnet-5-5")
    tracker.on_llm_end(make_result(1_000_000, 100_000))
    assert llm.total_cost() == pytest.approx(2.0 + 1.0)


def test_judge_모델_가격을_찾는다():
    assert llm.get_price(config.JUDGE_MODEL) == (3.0, 15.0)


def test_가격표에_없는_모델은_거부한다():
    with pytest.raises(ValueError):
        llm.CostTracker("scoring", "unknown-model")


def test_한도에_도달하면_호출을_막는다(monkeypatch):
    monkeypatch.setattr(config, "COST_LIMIT_POC_USD", 1.0)
    tracker = llm.CostTracker("judge", config.JUDGE_MODEL)
    tracker.on_llm_end(make_result(400_000, 0))  # $1.20
    with pytest.raises(llm.CostLimitExceeded):
        tracker.on_chat_model_start({}, [])


def test_한도_초과_예외가_실제_호출까지_전달된다(monkeypatch):
    monkeypatch.setattr(config, "COST_LIMIT_POC_USD", 0.0)
    model = GenericFakeChatModel(
        messages=iter([AIMessage("hi")]),
        callbacks=[llm.CostTracker("judge", config.JUDGE_MODEL)],
    )
    with pytest.raises(llm.CostLimitExceeded):
        model.invoke("질문")


def test_judge_LLM_에_비용_기록이_붙는다():
    judge = llm.get_judge_llm()
    assert judge.temperature == 0
    assert any(isinstance(cb, llm.CostTracker) and cb.role == "judge" for cb in judge.callbacks)
