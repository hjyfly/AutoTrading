# AutoTrading — 종목별 5거래일 방향 신호 에이전트

매 거래일 15:15(한국장 마감 전)에 종목마다 **5거래일 뒤 주가 방향(상승·보합·하락)** 신호를 내고, 그 근거를 질의응답으로 설명하는 개인용 에이전트입니다. 차트·수급·거시·해외 지표로 학습한 숫자 모델(LightGBM)이 신호를 내고, 뉴스는 LLM 이 채점해 참고 근거로 보여 줍니다.

> **투자 권유가 아닙니다.** 신호와 근거만 제공하고 매매는 사람이 직접 합니다. 신호로 자동 주문하는 코드는 없습니다.

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 종목 | 종목 레지스트리(`config.MULTI_TICKERS`): 삼성전자(운영·매매), SK하이닉스, 한화에어로스페이스, 레인보우로보틱스. 설정 한 줄 + 명령 하나로 더할 수 있다 |
| 신호 정의 | t 일 15:15 신호, t 종가 → t+5 거래일 종가가 +3% 이상 상승 / ±3% 보합 / -3% 이하 하락 |
| 기간별 신호 | 삼성전자는 1·3·10·20거래일(±1.3%·±2.3%·±4.2%·±6.0%) 참고 신호도 낸다(11일차). 매매 기준은 5거래일뿐 |
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

처음 설치하거나 다른 PC 로 옮길 때는 아래 [수행 가이드](#수행-가이드-다른-pc-에서-이어-개발실행하기)를 따릅니다. 설치가 끝난 뒤 자주 쓰는 명령:

```powershell
.venv\Scripts\streamlit run src/ui.py                          # 화면 (종목 선택)
.venv\Scripts\python -m uvicorn src.server:app --port 8000     # API: POST /query {"question": "2026-10-15 SK하이닉스 신호는?"}
.venv\Scripts\python -m src.daily_run --no-email               # 하루치 자동 실행을 손으로(오늘 신호를 기록한다. 자동 실행과 겹치지 않게)
.venv\Scripts\python -m src.pipeline.scoreboard                # 실시간 성적표
.venv\Scripts\python -m src.pipeline.multi_ticker analysis     # 종목별 분석·비교
.venv\Scripts\python -m src.pipeline.experiment --monitor --code 000660   # 종목별 조건 모니터링 (--all 전 종목)
.venv\Scripts\python -m evaluation.run_eval --judge            # /query 평가
.venv\Scripts\python -m pytest -q                              # 테스트
```

## 수행 가이드: 다른 PC 에서 이어 개발·실행하기

### 1. 코드와 파이썬

```powershell
git clone https://github.com/hjyfly/AutoTrading.git
cd AutoTrading
python -m venv .venv                      # Python 3.11 이상 (개발 PC 는 3.14.8)
.venv\Scripts\pip install -r requirements.txt   # 개발 PC 에서 확인한 버전으로 고정돼 있다
```

macOS·Linux 는 `.venv/bin/python`, `.venv/bin/pip` 로 바꿔 쓴다.

### 2. `.env` 설정 (git 에 올리지 않는다)

저장소 루트에 `.env` 를 만든다. **비밀 값과 주문 스위치만** 여기에 두고, 나머지 실행 설정은 git 에 올라가는 `src/config.py` 에 기본값이 있다(`.env` 에 같은 키를 쓰면 `.env` 가 이긴다).

| 키 | 필수 | 용도 | 얻는 곳 / 기본값 |
|---|---|---|---|
| `AWS_ACCESS_KEY_ID` · `AWS_SECRET_ACCESS_KEY` | **필수** | Bedrock 호출(뉴스 채점·기사 추출·`/query` 에이전트·judge·RAG 임베딩) | AWS IAM 사용자 키. Bedrock 모델 접근 권한: Sonnet 4.6(`global.` 프로필), Sonnet 4.5(`us.` 프로필, judge), Titan Embeddings v2 |
| `DART_API_KEY` | **필수** | DART 공시 수집, `/query` 공시 조회 | opendart.fss.or.kr 인증키 |
| `AWS_DEFAULT_REGION` | 선택 | 채점·에이전트·임베딩 리전 | 기본 `config.AWS_REGION_DEFAULT` = `us-east-1` (judge 리전은 `config.JUDGE_REGION`) |
| `BEDROCK_MODEL_ID` | 선택 | 채점·에이전트 모델 | 기본 `config.SCORING_MODEL_DEFAULT` = Sonnet 4.6. **바꾸면 유출 테스트와 뉴스 전체 재채점이 필요하다**(CLAUDE.md) |
| `BEDROCK_EMBEDDING_MODEL_ID` | 선택 | RAG 임베딩 | 기본 `config.EMBEDDING_MODEL_DEFAULT` = `amazon.titan-embed-text-v2:0`. 바꾸면 색인을 다시 만든다 |
| `LLM_PROVIDER` | 선택 | LLM 공급자 | 기본 `config.LLM_PROVIDER_DEFAULT` = `bedrock` (`anthropic` 은 이전 대비용, 지금 미구현) |
| `GMAIL_SENDER` · `GMAIL_APP_PASSWORD` · `NOTIFY_TO` | 선택 | 평일 15:20 알림 메일. 없으면 메일만 건너뛴다 | Google 계정 2단계 인증 뒤 "앱 비밀번호" |
| `KIS_APP_KEY` · `KIS_APP_SECRET` · `KIS_ACCOUNT_NO` | 선택 | 한국투자증권 조회(계좌 잔고)·사람 실행 주문. 없으면 계좌 줄만 건너뛴다 | KIS Developers 앱 키. 계좌는 `8자리-2자리` |
| `KIS_ENV` | 선택 | `mock`(모의투자) / `real` | 기본 `config.KIS_DEFAULT_ENV` = `mock` |
| `KIS_TRADING_ENABLED` | 선택 | **주문 스위치.** `true` 일 때만 `src/broker.py` 를 사람이 직접 실행해 주문할 수 있다(신호로 자동 주문하지 않는다) | 기본 꺼짐. 안전을 위해 config 에 두지 않는다 |
| `FRED_API_KEY` | 안 씀 | 옛 FRED 원본을 다시 만들 때만 | — |

빈 템플릿(값을 채워 `.env` 로 저장):

```dotenv
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
DART_API_KEY=
# 선택: 비우면 config 기본값
# AWS_DEFAULT_REGION=us-east-1
# BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-6
# BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
# LLM_PROVIDER=bedrock
# 선택: 알림
GMAIL_SENDER=
GMAIL_APP_PASSWORD=
NOTIFY_TO=
# 선택: 한국투자증권
KIS_APP_KEY=
KIS_APP_SECRET=
KIS_ACCOUNT_NO=
KIS_ENV=mock
KIS_TRADING_ENABLED=false
```

### 3. 데이터 옮기기

학습 표·신호·뉴스 점수·비용 기록 등 일부는 git 에 있지만, 아래는 git 에 없다(기사 원문·제목이 있거나 크다). **기존 PC 의 `data/` 폴더를 통째로 복사하는 것이 가장 안전하다**(USB·사내 저장소 등, 공개 저장소에 올리지 않는다).

| 위치 | 다시 만들 수 있나 | 없으면 |
|---|---|---|
| `data/processed/signals_live.parquet`, `data/processed/tickers/signals_live_tickers.parquet` | **아니요** — 장 마감 전에 기록한 실시간 신호(판정 근거) | 실시간 성적표가 처음부터 다시 쌓인다. 반드시 복사 |
| `data/raw/` (일봉·수급·거시 원본, 뉴스 원문, 다른 종목 원본) | 일부만 — 다시 받으면 이미 저장한 날짜의 값이 달라질 수 있고(저장 행 불변 원칙) 뉴스는 수집에 오래 걸린다 | 반드시 복사 |
| `data/scores/traces/`, `data/scores/features/`, `data/scores/alerts/` | 가능하지만 LLM 비용(약 $70)이 든다 | 복사 권장 |
| `data/processed/news.parquet`, `selected_news.parquet` | `data/raw` 가 있으면 다시 만든다 | 복사 권장 |
| `data/index/` (RAG 색인) | `python -m src.retriever` (판단 기록이 있어야 한다, 비용 약 $0.01) | 다시 만들어도 된다 |
| `data/processed/tickers/`(신호 기록 외), `data/processed/sectors/` | `python -m src.pipeline.sector_models run` → `multi_ticker analysis` | 다시 만들어도 된다 |
| `data/processed/horizons/` (기간별 과거 신호) | `python -m src.pipeline.experiment --horizons`(약 15분, LLM 없음. H1 ~ H4 판정은 `day11_judged.json` 기록을 그대로 쓰고 다시 하지 않는다) | 없으면 과거 날짜의 기간별 신호 조회·평가 문항 P08 이 '신호 없음' 이 된다 |
| `data/scores/traces_archive/` | 아니요 — 옛 모델(Sonnet 4.5)·옛 채점 버전 판단 기록 보관본 | 지금 운영에는 쓰지 않는다. 기록을 남기려면 복사 |
| `data/processed/scoreboard.json`, `data/experiments/`, `data/logs/`, `data/backup/`, `data/kis/`(토큰), `data/scores/eval/`(평가 상세) | 실행하면 다시 생긴다 | 복사하지 않아도 된다 |

### 4. 확인

```powershell
.venv\Scripts\python -m pytest -q                         # 모두 통과해야 한다
.venv\Scripts\python -m src.pipeline.scoreboard           # 실시간 기록이 옮겨졌는지 (실시간 기록 N일)
.venv\Scripts\python -c "from src import tools; print(tools.signal_for('2026-10-15', 'SK하이닉스'))"   # 읽기만 하는 조회
.venv\Scripts\streamlit run src/ui.py
```

### 5. 평일 자동 실행 옮기기

**두 PC 가 동시에 자동 실행하지 않게 한다.** 같은 날 두 번 기록되면 실시간 기록이 엇갈린다. 새 PC 에 등록하기 전에 기존 PC 의 작업을 끈다: `schtasks /Change /TN AutoTrading_Daily /DISABLE`.

Windows (작업 스케줄러, 평일 15:20, PC 가 켜져 있어야 한다):

```powershell
schtasks /Create /TN AutoTrading_Daily /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:20 /TR "cmd /c cd /d C:\경로\AutoTrading && .venv\Scripts\python.exe -m src.daily_run >> data\logs\scheduler.log 2>&1"
```

macOS·Linux (cron, 시스템 시간대가 KST 여야 한다):

```
20 15 * * 1-5 cd /경로/AutoTrading && .venv/bin/python -m src.daily_run >> data/logs/scheduler.log 2>&1
```

### 6. 개발을 이어 갈 때

- Claude Code 로 열면 [CLAUDE.md](CLAUDE.md) 규칙과 `.claude/agents/` 검증 에이전트(leak-guard·backtest-auditor·spec-checker·data-probe)가 그대로 적용된다.
- 기준 문서는 [SERVICE.md](SERVICE.md), 산출물 목록·네이밍 규칙은 [evaluation/reports/README.md](evaluation/reports/README.md), 지난 작업은 `DAY<N>_PLAN.md` 의 결과 절에 있다.
- 비용은 `data/scores/cost_log.csv`(git 에 있음)로 이어서 센다. 개발 한도는 `config.COST_LIMIT_POC_USD`.

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
| `DAY1_PLAN.md` ~ `DAY11_PLAN.md` | 일차별 계획과 결과 |
| [evaluation/reports/preregistration.md](evaluation/reports/preregistration.md) | 사전 등록 장부(모든 판정의 기준과 결과) |

## 원칙

- **시점 규칙:** 모든 데이터 읽기는 `as_of` 를 받고 그 시점에 알 수 없던 값은 쓰지 않습니다(테스트와 leak-guard 로 확인).
- **사전 등록:** 새 조건은 결과를 보기 전에 장부에 적고, 판정 구간은 한 번만 씁니다. 이미 본 구간의 숫자는 탐색으로 표시합니다.
- **무료 데이터·비용 한도:** 유료 출처를 쓰지 않고, LLM 호출은 모두 비용을 기록하며 한도에서 멈춥니다.
- **사람이 매매:** 신호로 주문하지 않습니다. 운영 경로(daily_run·live_signal·agent·ui·tools·server)는 주문 모듈을 부르지 않습니다.
