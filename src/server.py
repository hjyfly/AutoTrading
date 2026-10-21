"""FastAPI 서버. POST /query 로 question 을 받아 answer, contexts, trace 를 돌려준다(제출 규약)."""
from fastapi import FastAPI
from pydantic import BaseModel

from src import agent

app = FastAPI(title="AutoTrading 질의응답")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    contexts: list
    trace: list


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """질문 하나에 답한다."""
    return QueryResponse(**agent.answer(req.question))
