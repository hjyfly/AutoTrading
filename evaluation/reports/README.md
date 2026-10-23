# 산출물 목록과 네이밍 규칙

> 작성 2026-10-23(10일차) · 기준 문서 [SERVICE.md](../../SERVICE.md) · 개발 규칙 [CLAUDE.md](../../CLAUDE.md)
> 이 폴더(`evaluation/reports/`)의 리포트와 저장소 루트의 기획·계획 문서를 종류별로 정리한다. 남기는 기준: 판정 기록, 지금 상태를 보여 주는 상시 리포트, 결정의 근거가 된 점검 기록. 결론을 옮긴 일회성 스냅숏은 지운다(git 에 남는다). 새 산출물을 만들면 아래 규칙대로 이름을 짓고 이 목록에 한 줄 더한다.
> **판정** = 사전 등록한 기준으로 한 번만 계산한 결과(바꾸지 않는다). **탐색** = 이미 본 구간을 다시 계산한 숫자(판정에 쓰지 않는다). **재생성** = 데이터가 늘면 다시 만드는 상시 리포트.

## 1. 네이밍 규칙

| 종류 | 위치·이름 | 예 |
|---|---|---|
| 기획·규칙 | 루트 `SERVICE.md`(기획·성공 기준, 하나만), `CLAUDE.md`(개발 규칙) | — |
| 일차 계획 | 루트 `DAY<N>_PLAN.md` (N 은 일차 숫자, 자리 맞춤 없음) | `DAY10_PLAN.md` |
| 외부 지시서 | 루트 `<주제>_PLAN.md` 중 사용자가 가져온 문서(판단은 해당 일차 계획에 적는다) | `IMPROVEMENT_PLAN.md` |
| 일차 결과 리포트 | `day<N>_<주제>.md` (주제는 영어 소문자 snake_case 한두 단어) | `day9_news.md` |
| 상시 리포트 | `<종류>.md` — 종목마다 따로면 `<종류>_<종목코드>.md`(운영 종목은 코드 없이) | `experiments_monitor_000660.md` |
| 구간 변형 | 종류 뒤 `_<구간>`(예: `_ext` 확장 구간), 종목 코드는 맨 끝 | `experiments_features_ext.md` |
| 일회성 조건 비교 | `experiments_<YYYYMMDD>_<HHMM>_<이름>[_<종목코드>].md` (명령줄 `--name`, 화면 저장은 이름 `ui`) | `experiments_20261022_1516_news_rules.md`(이름 예시) |
| 데이터 출처 점검 | `data_probe_<YYYYMMDD>[_<범위>].md` | `data_probe_20261022_news.md` |
| 장부 | `preregistration.md` 하나에 사전 등록과 판정을 이어 적는다(ID: 일차마다 글자 하나 + 번호, 예: P1·Q1·R1·V1·A1·M1) | — |
| 데이터 원본 | `data/raw/<종류>/<이름>.parquet`, 확장 원본은 `<이름>_ext`, 다른 종목은 `data/raw/tickers/<종목코드>_<ohlcv\|flows>.parquet`·`sector_<etf>`·`kr_<지수>` | `data/raw/tickers/000660_ohlcv.parquet` |
| 신호 표 | `data/processed/signals_<모델>.parquet`, 종목·알고리즘별은 `data/processed/sectors/signals_<알고리즘>_<종목코드>.parquet`, 실시간 기록은 `signals_live*.parquet` | `signals_S4_277810.parquet` |
| 날짜별 기록 | `data/scores/<종류>/<YYYY-MM-DD>.json`(판단 기록 `traces`, 기사 피처 `features`) | `data/scores/features/2026-10-15.json` |
| 화면용 표 | `data/processed/tickers/analysis/<표 이름>.parquet` (재생성 가능, git 제외) | `periods.parquet` |

공통: 날짜는 파일 이름에서 `YYYYMMDD`(리포트) 또는 `YYYY-MM-DD`(날짜별 기록), 종목은 6자리 코드, 영어 소문자와 `_` 만 쓴다. 기사 제목·URL 이 들어가는 파일은 `data/` 아래에 두고 git 에 올리지 않는다(리포트에는 싣지 않는다).

## 2. 기획·계획 문서 (루트)

| 문서 | 내용 |
|---|---|
| [README.md](../../README.md) | 시스템 소개·빠른 시작·종목 추가·폴더·문서 지도 |
| [SERVICE.md](../../SERVICE.md) | 시스템 기획·성공 기준·검증 결과·운영 기능 전체(10일차에 SERVICE2.md 통합) |
| [CLAUDE.md](../../CLAUDE.md) | 개발 규칙, 파일 역할, 하지 말 것 |
| [DAY1](../../DAY1_PLAN.md) · [DAY2](../../DAY2_PLAN.md) · [DAY3](../../DAY3_PLAN.md) | 데이터 수집 / 뉴스 채점 / 모델 학습·과거 검증 |
| [DAY4](../../DAY4_PLAN.md) · [DAY5](../../DAY5_PLAN.md) · [DAY6](../../DAY6_PLAN.md) | 과거 확장 검증 / 하락 위험 모델 / 한국투자증권 API |
| [DAY7](../../DAY7_PLAN.md) · [DAY8](../../DAY8_PLAN.md) | 다종목 분석 / 분야별 알고리즘 |
| [DAY9](../../DAY9_PLAN.md) · [DAY10](../../DAY10_PLAN.md) | 뉴스 신호 고도화 / 종목별 플랫폼·모듈화 |
| [IMPROVEMENT_PLAN.md](../../IMPROVEMENT_PLAN.md) | 외부 뉴스 고도화 지시서(판단은 DAY9_PLAN.md 2절) |

## 3. 일차 결과 리포트

| 리포트 | 성격 | 핵심 |
|---|---|---|
| [day1_data.md](day1_data.md) | 기록 | 1일차 수집 데이터 현황 |
| [day2_scoring.md](day2_scoring.md) | 기록 | 뉴스 채점 그래프와 첫 채점 |
| [day3_backtest.md](day3_backtest.md) | **판정** | PoC 공식 판정 기준 A·B (실패) |
| [day4_extended.md](day4_extended.md) | 탐색 | 2019 ~ 확장 원본으로 장세별 검증 |
| [day5_judgement_P1_P2_P3.md](day5_judgement_P1_P2_P3.md) | **판정** | 하락 위험 P1 ~ P3 (모두 불합격). 이름은 판정 기록이라 그대로 둔다(규칙 예외) |
| [day7_multi_ticker.md](day7_multi_ticker.md) | **판정** | 다종목 Q1 ~ Q3 (Q1·Q3 실패, Q2 채택 안 함) |
| [day8_sector_models.md](day8_sector_models.md) | **판정** | 분야별 R1 ~ R3 (로봇 S4 만 채택) |
| [day9_news.md](day9_news.md) | **판정** + 탐색 | 뉴스 V1 통과(약함)·V2·A1 불통과, 점수별 IC |

## 4. 상시 리포트 (재생성)

| 리포트 | 대상 | 다시 만드는 명령 |
|---|---|---|
| [preregistration.md](preregistration.md) | 사전 등록 장부(모든 판정의 기준과 결과) | 손으로 적는다(결과를 보기 전에 등록) |
| [query_eval.md](query_eval.md) | `/query` 평가 16문항(규칙 + judge) | `python -m evaluation.run_eval --judge` |
| [leak_test.md](leak_test.md) | LLM 미래 정보 유출 테스트 | `python -m src.pipeline.leak_test` (채점 모델을 바꾸면 반드시) |
| [tickers_analysis.md](tickers_analysis.md) | 전 종목 분석·비교(뉴스 제외, 탐색) | `python -m src.pipeline.multi_ticker analysis` |
| [experiments_monitor.md](experiments_monitor.md) | 조건 모니터링 — 삼성전자(운영) | `python -m src.pipeline.experiment --monitor` (금요일 자동) |
| [experiments_monitor_000660.md](experiments_monitor_000660.md) · [_012450](experiments_monitor_012450.md) · [_277810](experiments_monitor_277810.md) | 조건 모니터링 — SK하이닉스·한화에어로스페이스·레인보우로보틱스 | `... --monitor --code <코드>` 또는 `--all` (금요일 자동, 메일 뒤) |
| [experiments_features.md](experiments_features.md) | 지표 그룹 진단 — 삼성전자 | `python -m src.pipeline.experiment --features` |
| [experiments_features_ext.md](experiments_features_ext.md) | 지표 그룹 진단 — 삼성전자, 확장 구간 | `... --features --ext` |
| [experiments_features_000660.md](experiments_features_000660.md) · [_012450](experiments_features_012450.md) · [_277810](experiments_features_277810.md) | 지표 그룹 진단 — 다른 종목 | `... --features --code <코드>` |

모두 탐색이다(장부는 예외). 화면 "리포트·조건 비교"에서 사이드바 종목으로 같은 리포트를 보고 다시 계산할 수 있다.

## 5. 일회성 조건 비교 (탐색)

`python -m src.pipeline.experiment --vary ... --name <이름>` 이나 화면 "보고서로 저장"으로 만들고, 이 표에 한 줄 더한다. 결론을 SERVICE.md 로 옮긴 뒤에는 지워도 된다(같은 명령으로 다시 만든다).

| 리포트 | 비교한 것 |
|---|---|
| (지금 없음) | 2026-10-22 의 6개(분할 매매·학습 창·뉴스 보정 규칙·점수 종류·하락 위험 거부·뉴스 끄기)는 결론을 SERVICE.md 11.3·11.5·12절로 옮기고 2026-10-23 정리했다. 원본은 git 커밋 `1008b79` 에서 볼 수 있다 |

## 6. 데이터 출처 점검

| 리포트 | 시점 |
|---|---|
| [data_probe_20261021.md](data_probe_20261021.md) | 1일차·2 ~ 4월 뉴스 수집 전 |
| [data_probe_20261022_ext.md](data_probe_20261022_ext.md) | 4일차 숫자 지표 과거 확장 |
| [data_probe_20261022_news.md](data_probe_20261022_news.md) | 뉴스 확장 수집 결과 |
| [data_probe_20261022_news_aug_oct.md](data_probe_20261022_news_aug_oct.md) | 뉴스 8 ~ 10월 추가분 |
