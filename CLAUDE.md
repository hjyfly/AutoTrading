# 프로젝트 규칙

## 대화 언어
- 사용자에게 하는 모든 답변(진행 상황 보고·요약·질문·오류 설명)은 **한국어로만** 쓴다. 영어로 답하지 않는다
- 코드 식별자·명령어·파일 경로는 그대로 두고, 설명 문장은 한국어로 쓴다

## 기술 스택
- Python, LangChain, LangGraph
- 모델은 Amazon Bedrock (ChatBedrockConverse)
  - 개발 단계는 Bedrock 을 쓰고, 이후 Claude API(ChatAnthropic)로 옮길 수 있다
  - LLM 객체는 src/llm.py 한 곳에서만 만든다. 공급자는 .env 의 LLM_PROVIDER(bedrock | anthropic)로 고른다
  - 다른 파일에서 ChatBedrockConverse·ChatAnthropic 을 직접 만들지 않는다
  - Sonnet 5.5 는 temperature·top_p 를 기본값이 아닌 값으로 보내거나 budget_tokens 를 보내면 400 오류가 난다. 이 값들을 설정하지 않는다
  - judge(평가용 LLM)는 src/config.py 의 상수(JUDGE_MODEL·JUDGE_REGION·JUDGE_TEMPERATURE)로 고정하고 바꾸지 않는다. 버전 비교는 항상 같은 judge 로 한다
  - judge 는 .env 의 리전과 별개로 JUDGE_REGION 에서 호출한다
  - 개발 테스트: judge 와 채점 모델이 같은 계열이어도 된다. 같은 모델이면 평가 리포트에 "같은 모델 평가(편향 가능)"라고 적는다
  - 채점·에이전트는 global. 프로필의 **Sonnet 4.6**(`global.anthropic.claude-sonnet-4-6`, 2026-10-22 사용자 결정), judge 는 us. 프로필의 **Sonnet 4.5** 고정이다
  - 2026-10-21 요청 제한으로 채점 경로를 us. → global. 으로 옮겼고(Sonnet 4.5), 2026-10-22 채점 모델을 Sonnet 4.6 으로 바꿨다. Sonnet 4.6 ID 에는 날짜가 붙지 않는다
  - 채점 모델 ID 를 바꾸면 유출 테스트를 다시 돌리고, 뉴스 구간 전체를 다시 채점한다(점수 표에 모델이 섞이면 멈춘다)
  - 최종 검증: judge 를 채점 모델과 다른 모델로 바꾼다. 바꾼 뒤에는 개발 중 평가 점수와 직접 비교하지 않는다
  - judge 호출 비용도 PoC 한도에 포함한다. cost_log.csv 에 역할(scoring | judge | agent | embedding)을 함께 기록한다
- Agent 생성은 langchain.agents 의 create_agent 를 쓴다
  - create_agent: /query 질의응답 에이전트(src/agent.py)
  - LangGraph StateGraph 직접 구성: 뉴스 채점 흐름(채점자 3명 → 토론 1회 → 최종 정리). 순서가 고정된 흐름이라 create_agent 로 만들지 않는다
- 숫자 모델은 LightGBM, /query 서버는 FastAPI
- 벡터 저장소는 로컬 Chroma (무료). 임베딩도 src/llm.py 에서 공급자별로 만든다
- 기획과 성공 기준은 SERVICE.md 를 따른다. 이 파일과 충돌하면 사용자에게 먼저 묻는다

## 폴더 구조 (제출 규약. 바꾸지 않는다)
src/agent.py       메인 에이전트 그래프
src/tools.py       도메인 도구
src/retriever.py   RAG 파이프라인
data/              사용한 문서와 데이터
evaluation/        test_queries.csv 와 평가 리포트

### 제출 규약 파일의 역할
- src/agent.py      /query 에이전트. 판단 기록을 근거로 "왜 그 신호였는가"에 답한다
- src/tools.py      에이전트 도구 9개: 일봉·수급·거시 지표·차트 지표·공시·뉴스 조회, 신호·판단 기록·모의 매매 결과 조회
                    수집·계산 함수(src/pipeline)를 감싸기만 하고 새로 계산하지 않는다. 모든 도구는 as_of 시점에 알 수 있던 정보만 돌려준다
                    저장된 데이터를 먼저 쓰고, 저장 범위를 넘으면 실시간으로 받되 data/raw 에 쓰지 않는다
- src/retriever.py  판단 기록·기사·DART 공시 검색(RAG). 모든 검색은 as_of 날짜 이후 문서를 제외한다

### 추가로 허용하는 파일 (이 목록 밖의 파일은 만들기 전에 묻는다)
src/config.py               기간, ±3% 임계값, 수수료 0.1%, 랜덤 시드, 비용 한도 등 모든 설정값
src/llm.py                  LLM·임베딩 생성, 토큰·비용 기록, 비용 한도 도달 시 중단
src/server.py               FastAPI, POST /query
src/pipeline/collect_prices.py   삼성전자·인버스 ETF 일봉 (FinanceDataReader, 예비 pykrx), 순매수 (네이버 증권 모바일 API), 확장 지표용 국내 종가(코스피·코스피200·SK하이닉스)
src/pipeline/collect_macro.py    yfinance 거시·해외 지표, 확장 지표용 해외 시계열(마이크론·TSMC·달러 인덱스) (FRED 는 공개 지연으로 미사용)
src/pipeline/collect_news.py     GDELT(한국어 매체 포함) + Google 뉴스 RSS(한국어 보조)
src/pipeline/collect_dart.py     시점별 DART 공시
src/pipeline/refresh_data.py     저장 가격·지표 주간 갱신(새 확정 날짜 행만 덧붙임, 하나라도 실패하면 쓰지 않음, 쓰기 전 data/backup 에 복사) → 신호·모의 매매 재계산
src/pipeline/features.py         차트 지표 계산, 시점 규칙에 맞춘 학습용 표
src/pipeline/labels.py           5거래일 ±3% 정답 데이터 (기간·기준은 인자로 바꿀 수 있고 기본은 config 값)
src/pipeline/select_news.py      신호일마다 채점할 기사 30건 선별
src/pipeline/scoring.py          뉴스 채점 LangGraph (버전 2: 종합 + 분류별 점수 + 하락 위험. 버전이 섞이면 점수 표를 만들지 않는다)
src/pipeline/leak_test.py        미래 정보 유출 테스트
src/pipeline/train.py            LightGBM 워크포워드 학습
src/pipeline/news_signal.py      숫자 모델 신호를 뉴스 합의 점수로 보정 (기준 B 용, 고정 규칙)
src/pipeline/live_signal.py      오늘(지정일) 신호를 실시간 데이터로 생성. 과거 검증과 같은 규칙(재학습 주기 config.RETRAIN_FREQ 안의 첫 거래일 재학습 — 지금 매주, 정답 기간 간격)
                                 뉴스 채점은 15:15 이후 요청할 때만(LLM 7회). 2026-10-22 부터 뉴스는 참고용(config.NEWS_ADJUST_IN_SIGNAL). 결과는 data/processed/signals_live.parquet
src/ui.py                        Streamlit 화면 (채팅 + 신호·자산 곡선·지표·판단 기록·실시간 성적)
src/pipeline/scoreboard.py       실시간 성적표: 매일 기록한 신호를 정답 확정 뒤 채점, 모델·들고 있기 비교, 성능 저하 경고
src/daily_run.py                 평일 15:20 자동 실행: 오늘 신호(운영·비교 모델, 뉴스 채점은 참고용) → 성적표 → Gmail 알림
src/notify.py                    Gmail 알림 (.env 의 GMAIL_SENDER·GMAIL_APP_PASSWORD·NOTIFY_TO)
src/pipeline/backtest.py         모의 매매와 성공 기준 A·B 판정
src/pipeline/experiment.py       조건 비교 실험(정답 기간·기준, 지표 묶음, 재학습 주기, 체결 시점, 하락 행동, 뉴스 보정 방법). 결과는 모두 탐색이고 공식 판정을 바꾸지 않는다
                                 조건 모니터링(하나씩 바꾼 민감도 분석 + 해석 리포트 experiments_monitor.md, 금요일 자동 실행)
tests/                           위 모듈의 테스트
.claude/agents/                  개발용 검증 서브에이전트 (아래 "검증 에이전트" 참고)
.gitignore, .gitattributes, requirements.txt   git 제외 목록, 줄바꿈 규칙(LF 고정), 의존성
evaluation/run_eval.py, evaluation/__init__.py /query 평가 실행
*.md                             기획·계획 문서 (SERVICE.md, DAY1_PLAN.md 등)

data/raw/        수집 원본. 이미 저장한 행은 수정하지 않는다(주간 갱신은 새 날짜 행만 덧붙인다)
data/backup/     주간 갱신 전 원본 복사본(git 제외)
data/processed/  지표·학습용 표·정답 데이터
data/scores/     뉴스 채점 결과(news_scores.parquet)와 판단 기록(traces/<날짜>.json, 기사 제목이 있어 git 제외)
data/index/      Chroma 색인
evaluation/reports/  PoC 보고서, 조건 비교 실험 보고서(experiments_*.md)
data/experiments/    조건 비교 실험 결과 표(csv, 재생성 가능해 git 제외)

## 주고받는 형식 (제출 규약)
- POST /query 로 받고 question 필드를 읽는다
- 답은 answer, contexts, trace 세 키로 돌려준다
  - answer: 방향(상승/보합/하락) + 확신도 + 근거 요약
  - contexts: 답에 쓴 기사·DART 공시·지표 값 (출처와 날짜 포함)
  - trace: 에이전트가 부른 도구 순서와 인자, 질문 시점(as_of), 비용. 채점자 3명 점수·토론은 get_news_judgement 결과로 contexts 에 들어간다
- 질문 예: "2026-03-05 삼성전자 신호는?", "3월 둘째 주에 하락 신호가 나온 이유는?"
- 질문에 날짜가 있으면 그 날짜를 as_of 로 쓴다. 날짜가 없으면 오늘(KST)을 쓴다
- 요청마다 tools.request_as_of(as_of) 안에서 에이전트를 돌린다. 도구는 그 날짜보다 뒤를 조회하지 않는다(늦은 날짜는 낮추거나 거부)
- 날짜를 콕 집은 질문은 get_signal·get_news_judgement 로, 이유·맥락 질문은 retriever.search(as_of 필터) 로 찾는다
- evaluation/test_queries.csv 열: question, as_of, expected_direction, expected_sources(;로 구분), note(judge 가 보는 기대 사항)
- 평가 실행: evaluation/run_eval.py. 자동 검사(시점·도구·방향·시점 이후 자료)와 judge 채점 → evaluation/reports/query_eval.md
  (질문별 상세 결과는 기사 제목이 들어 있어 data/scores/eval/ 에 두고 git 에 올리지 않는다)

## 코드 규칙
- 파일 하나에 한 가지 역할만 둔다
- 함수와 도구에는 한국어 docstring 을 쓴다
- 비밀 값은 .env 에서 읽고 코드에 적지 않는다
  - 있음: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  - 있음: DART_API_KEY, FRED_API_KEY(지금은 미사용. 옛 FRED 원본을 다시 만들 때만)
  - 선택: LLM_PROVIDER(기본 bedrock), BEDROCK_MODEL_ID(기본 config.SCORING_MODEL_DEFAULT = Sonnet 4.6)
  - 추가 필요: BEDROCK_EMBEDDING_MODEL_ID (RAG 색인 단계에서)
  - 선택: GMAIL_SENDER, GMAIL_APP_PASSWORD, NOTIFY_TO (Gmail 알림. 값은 코드·로그·문서에 남기지 않는다)
  - 나중에 추가: ANTHROPIC_API_KEY
- 시점 규칙(SERVICE.md 섹션 6): 데이터를 읽는 함수는 모두 as_of 날짜를 받고, 그 시점에 알 수 없던 데이터는 돌려주지 않는다
  - 숫자 데이터는 t-1일 종가까지, 뉴스는 t-1일 15:15 ~ t일 15:15 (KST)
  - 시점 규칙을 지키는지 확인하는 테스트를 tests/ 에 둔다
- 비용 한도: LLM 호출은 모두 src/llm.py 를 거치고, 호출마다 토큰과 비용을 data/scores/cost_log.csv 에 남긴다
  - PoC(개발 단계) 누적 $150(2026-10-22 $80 에서 상향, 사용자 결정), 운영 월 $100 에 도달하면 예외를 던지고 멈춘다
- 더미 데이터를 쓰면 데이터와 보고서에 is_dummy 표시를 붙인다
- 설정값은 src/config.py 에만 두고 코드 곳곳에 숫자를 적지 않는다
- 지표 묶음은 config.FEATURE_SETS 로 고른다. base_v1(1~3일차 45개)은 이름으로 고정했고, candidate(A+B+D)는 사전 등록 목록이라 결과를 본 뒤 바꾸지 않는다
  - 운영 base 는 base_v1 + RSI(rsi14) 46개다(2026-10-22 사용자 결정). 운영 base 를 바꾸는 것은 사용자 결정으로만 한다
  - 탐색·실시간 비교용으로 config.FEATURE_GROUPS 의 그룹을 이은 'groups:' 묶음도 쓸 수 있다
  - 새 지표를 더할 때는 base 계산 결과가 바뀌지 않았는지(features·signals_numeric 이 이전과 같은지) 확인한다
- 환경: Python 3.11 이상, 의존성은 requirements.txt, 테스트는 pytest

## 검증 에이전트 (.claude/agents/)
- 개발 중 Claude Code 에서 돌리는 서브에이전트다. Bedrock PoC 예산을 쓰지 않는다. 모두 코드를 고치지 않고 보고만 한다
- leak-guard: src/pipeline·src/tools.py·src/retriever.py 를 바꾼 뒤, 학습·모의 매매를 돌리기 전에 반드시 돌린다
- backtest-auditor: src/pipeline/backtest.py·train.py 를 바꾼 뒤, PoC 보고서 숫자를 확정하기 전에 반드시 돌린다
- spec-checker: 커밋하기 전에 돌린다
- data-probe: 1일차 수집 전, 2~4월 전체 뉴스 수집·본 채점 직전에 돌린다
- 확정 유출·판정 오류·위반이 보고되면 고친 뒤 같은 에이전트를 다시 돌려 통과를 확인한다

## 하지 말 것
- 요청하지 않은 파일을 새로 만들지 않는다
- 기존 파일을 통째로 다시 쓰지 않는다. 바뀐 부분만 고친다 (코드와 문서 모두)
- 증권사 주문 API나 자동 주문 코드를 만들지 않는다
- 유료 데이터 출처를 추가하지 않는다
- 학습·검증·채점에서 as_of 이후 데이터를 쓰지 않는다
- .env 와 data/raw 의 기사 원문, 기사 제목·URL 을 모은 data/processed/news.parquet·selected_news.parquet 는 git 에 올리지 않는다