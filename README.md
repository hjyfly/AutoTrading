# AutoTrading — 종목별 5거래일 방향 신호 에이전트

매 거래일 15:15(한국장 마감 전)에 종목마다 **5거래일 뒤 주가 방향(상승·보합·하락)** 신호를 내고, 그 근거를 질의응답으로 설명하는 개인용 에이전트입니다. 차트·수급·거시·해외 지표로 학습한 숫자 모델(LightGBM)이 신호를 내고, 뉴스는 LLM 이 채점해 참고 근거로 보여 줍니다.

> **투자 권유가 아닙니다.** 신호와 근거만 제공하고 매매는 사람이 직접 합니다. 신호로 자동 주문하는 코드는 없습니다.

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 종목 | 종목 레지스트리(`config.MULTI_TICKERS`): 삼성전자(운영·매매), SK하이닉스, 한화에어로스페이스, 레인보우로보틱스. 설정 한 줄 + 명령 하나로 더할 수 있다 |
| 신호 정의 | t 일 15:15 신호, t 종가 → t+5 거래일 종가가 +3% 이상 상승 / ±3% 보합 / -3% 이하 하락 |
| 모델 | LightGBM 3분류, 지표 46개, 매주 재학습(워크포워드). 로봇 분야는 3종목 공동 학습(S4) |
| 뉴스 | 삼성전자만. 채점자 3명 토론(LangGraph) + 기사별 구조화 추출. **참고용**이며 신호를 바꾸지 않는다 |
| 판정 | 앞으로의 날을 기록하는 **실시간 성적표**로만 판정한다. 과거 구간 숫자는 탐색이다 |
| 운영 | 평일 15:20 자동 실행 → 전 종목 신호 → 성적표 → Gmail 알림. 금요일 데이터 갱신·조건 모니터링 |
| 질의응답 | `POST /query`(FastAPI) 와 Streamlit 화면. 답은 `answer`·`contexts`·`trace` |
| LLM | Amazon Bedrock Sonnet 4.6(채점·에이전트), judge Sonnet 4.5. 비용 기록과 한도(`src/llm.py`) |
| 데이터 | 무료 출처만: FinanceDataReader, 네이버 증권, yfinance, GDELT, Google 뉴스 RSS, DART |

지금까지의 결론(자세히는 [SERVICE.md](SERVICE.md) 11절): 과거 구간에서 들고 있기보다 나았던 것은 삼성전자(검증 구간 +352.7% vs +300.9%)뿐이라 매매는 삼성전자 위주입니다. 다른 종목과 뉴스의 쓸모는 2026-10-23 부터 쌓이는 실시간 성적표로 판단합니다.

## 구조

```
[수집]  일봉·수급·거시·해외 (무료 API, 새 확정 날짜만 덧붙임)      뉴스 GDELT + RSS · DART 공시 (운영 종목)
           │                                                        │
[가공]  특징 표(시점 규칙: 국내 t-1 종가, 해외 현지 날짜 < t) · 정답(5거래일 ±3%) · 기사 선별 30건
           │                                                        │
[판단]  숫자 모델 워크포워드 (종목별 알고리즘 S0 / 분야 S4) ─→ 신호        뉴스 채점(3명 토론) · 기사별 추출 ─→ 참고 근거·경보
           │
[운영]  평일 15:20 daily_run → 실시간 기록 → 성적표(scoreboard) → Gmail     /query 에이전트(도구 9개 + RAG) · Streamlit 화면
[검증]  실시간 성적표(판정) · 사전 등록 장부 · 미래 정보 유출 테스트 · 검증 서브에이전트(leak-guard 등)
```

## 빠른 시작

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

저장소 루트에 `.env` 파일을 만들고(이미 있으면 그대로 쓴다) 아래 키를 채웁니다. 값은 코드·문서에 적지 않습니다.

`.env` 키: `AWS_ACCESS_KEY_ID`·`AWS_SECRET_ACCESS_KEY`·`AWS_DEFAULT_REGION`(Bedrock), `DART_API_KEY`, `BEDROCK_EMBEDDING_MODEL_ID`(RAG). 선택: `GMAIL_SENDER`·`GMAIL_APP_PASSWORD`·`NOTIFY_TO`(알림), `KIS_*`(한국투자증권 조회·사람 실행 주문, 기본 모의·주문 꺼짐).

```powershell
.venv\Scripts\streamlit run src/ui.py                          # 화면 (종목 선택)
.venv\Scripts\python -m uvicorn src.server:app --port 8000     # API: POST /query {"question": "2026-10-15 SK하이닉스 신호는?"}
.venv\Scripts\python -m src.daily_run --no-email               # 하루치 자동 실행을 손으로
.venv\Scripts\python -m src.pipeline.scoreboard                # 실시간 성적표
.venv\Scripts\python -m src.pipeline.multi_ticker analysis     # 종목별 분석·비교
.venv\Scripts\python -m src.pipeline.experiment --monitor --code 000660   # 종목별 조건 모니터링 (--all 전 종목)
.venv\Scripts\python -m evaluation.run_eval --judge            # /query 평가
.venv\Scripts\python -m pytest -q                              # 테스트
```

## 종목 추가

1. `src/config.py` 의 `MULTI_TICKERS` 에 한 줄: `"005380": ("현대차", "자동차", None)`
2. `python -m src.pipeline.multi_ticker add 005380` → 원본 수집 → 과거 신호 → 분석
3. 다음 평일 15:20 부터 실시간 신호·성적표·메일에 들어가고, 화면·`/query` 에서는 바로 조회됩니다.

매매 대상(`TRADE_TICKERS`)과 뉴스 종목(`NEWS_TICKERS`)은 역할 설정으로 따로 정합니다. 매매 대상은 사용자 결정으로만 바꿉니다.

## 폴더

| 위치 | 내용 |
|---|---|
| `src/agent.py` · `src/tools.py` · `src/retriever.py` | `/query` 에이전트(create_agent), 도메인 도구, RAG(Chroma) |
| `src/server.py` · `src/ui.py` · `src/daily_run.py` · `src/notify.py` | API, 화면, 평일 자동 실행, Gmail |
| `src/config.py` · `src/llm.py` | 모든 설정값·종목 레지스트리, LLM·임베딩 생성과 비용 한도 |
| `src/pipeline/` | 수집·특징·정답·학습·뉴스 채점·모의 매매·성적표·실험·다종목 |
| `src/broker.py` | 한국투자증권 주문(사람이 직접 실행할 때만, 기본 모의·꺼짐) |
| `data/` | 원본(`raw`), 가공 표(`processed`), 채점·비용 기록(`scores`), 색인(`index`). 기사 원문·제목은 git 제외 |
| `evaluation/` | 평가 질문(`test_queries.csv`), 평가 실행, 리포트(`reports/`) |
| `tests/` | pytest (시점 규칙, 손계산, 실시간 = 과거 검증) |
| `.claude/agents/` | 개발용 검증 서브에이전트(leak-guard, backtest-auditor, spec-checker, data-probe) |

## 문서

| 문서 | 내용 |
|---|---|
| [SERVICE.md](SERVICE.md) | 기획·성공 기준·검증 결과·운영 기능 전체 (기준 문서) |
| [CLAUDE.md](CLAUDE.md) | 개발 규칙, 파일 역할, 하지 말 것 |
| [evaluation/reports/README.md](evaluation/reports/README.md) | 리포트 목록과 산출물 네이밍 규칙 |
| `DAY1_PLAN.md` ~ `DAY10_PLAN.md` | 일차별 계획과 결과 |
| [evaluation/reports/preregistration.md](evaluation/reports/preregistration.md) | 사전 등록 장부(모든 판정의 기준과 결과) |

## 원칙

- **시점 규칙:** 모든 데이터 읽기는 `as_of` 를 받고 그 시점에 알 수 없던 값은 쓰지 않습니다(테스트와 leak-guard 로 확인).
- **사전 등록:** 새 조건은 결과를 보기 전에 장부에 적고, 판정 구간은 한 번만 씁니다. 이미 본 구간의 숫자는 탐색으로 표시합니다.
- **무료 데이터·비용 한도:** 유료 출처를 쓰지 않고, LLM 호출은 모두 비용을 기록하며 한도에서 멈춥니다.
- **사람이 매매:** 신호로 주문하지 않습니다. 운영 경로(daily_run·live_signal·agent·ui·tools·server)는 주문 모듈을 부르지 않습니다.
