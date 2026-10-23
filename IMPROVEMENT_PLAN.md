# 뉴스 기반 삼성전자 스코어링 에이전트 — 고도화 작업 지시서

> 이 문서는 Claude Code가 읽고 코드베이스를 변경하기 위한 작업 지시서다.
> 작업 전에 반드시 기존 코드 구조(`config`, `ScoringState`, `Consensus`, `SCORING_PERSONAS`, `TASK`,
> `_persona_messages`, `moderator`, `time_header` 등)를 먼저 파악하고, 기존 스타일·네이밍·주석 밀도에 맞춰 수정할 것.
> 각 Phase는 독립적으로 머지 가능하도록 작게 나눠 진행하고, Phase마다 완료 기준(Acceptance)을 확인한다.

---

## 0. 현재 구조 요약 (변경 전)

- 데이터: GDELT GKG(뉴스 제목), FRED, yfinance — 무료 데이터만 사용 (빅카인즈 제외)
- 스코어링: LangGraph 스타일 상태(`ScoringState`)
  1. 세 페르소나(`short_trader`, `value_investor`, `risk_manager`)가 1차 의견 (`_persona_messages`)
  2. 토론(round2)
  3. `moderator`가 `Consensus` 구조화 출력으로 최종 점수 산출
- 출력: 종합 점수(`SCORE_MIN~SCORE_MAX`), 분류별 점수(삼성전자 직접 / 반도체·AI 업황 / 거시), 하방 위험(`RISK_MIN~RISK_MAX`)
- 예측 대상: `LABEL_HORIZON_DAYS` 거래일 수익률
- 문제: 정확도가 기대보다 낮음

## 목표

1. **평가 체계 정비** — 정확도 수치가 실제로 무엇을 의미하는지 확정
2. **구조 개선** — LLM은 기사별 구조화 피처 추출, 집계·예측은 통계 모델
3. **하방 위험 경보 + 근거 기사 요약 메시지** 기능 추가 (신규)
4. 데이터 품질 개선

---

## Phase 1. 평가 체계 정비 (최우선)

새 모듈 예: `evaluation.py` (기존 평가 코드가 있으면 그 파일을 확장)

### 1-1. 라벨을 초과수익으로
- yfinance로 `005930.KS`, `^KS11`(KOSPI), `^SOX` 일별 데이터 수집
- 롤링 베타(예: 120거래일)로 초과수익 계산
  ```python
  excess_ret = ret_005930 - beta_kospi * ret_kospi   # 옵션: SOX 베타도 차감
  ```
- 라벨 종류를 config로 선택 가능하게: `LABEL_TYPE = "raw" | "excess_kospi" | "excess_kospi_sox"`

### 1-2. 다중 호라이즌
- `LABEL_HORIZONS = [1, 2, 5]` 거래일 + 익일 시가→종가 수익률(`open_to_close_t1`)을 모두 계산
- 기존 `LABEL_HORIZON_DAYS`는 기본 호라이즌으로 유지

### 1-3. 지표
- **IC**: Spearman(점수, 미래수익률) — 전체 + 롤링 60일
- **적중률**: 방향 일치율 (점수 0은 제외)
- **분위 수익률**: 점수 상/하위 20% 그룹의 평균 수익률 차이
- **겹침 보정**: 호라이즌 > 1일 때 Newey-West 표준오차(lag = horizon-1) 또는 non-overlapping 샘플링으로 t-stat 계산
- **분류별 점수**(직접/업황/거시)와 **위험 점수**도 각각 따로 평가
  - 위험 점수는 "향후 N일 최저 수익률" 또는 "초과수익 < -2σ 발생 여부"에 대한 AUC/정밀도로 평가

### 1-4. 베이스라인
동일 기간·동일 지표로 비교 리포트 출력:
- 항상 0 / 항상 상승
- 최근 5일 모멘텀
- 전일 미국장 `^SOX` 수익률
- 현재 LLM 점수

### 1-5. 룩어헤드 방지
- 사용 LLM의 학습 컷오프 날짜를 `config.LLM_KNOWLEDGE_CUTOFF`로 두고, 리포트에서 **컷오프 이전/이후 구간을 분리**해서 표시
- 옵션 `ANONYMIZE_COMPANY=True`: 프롬프트에서 "삼성전자/Samsung" → "A사/Company A" 치환(과거 구간 백테스트용)
- GDELT 타임스탬프(UTC) → KST 변환 확인, `SIGNAL_CUTOFF_KST` 이후 기사가 t일 신호에 섞이지 않는지 단위 테스트 추가

**Acceptance**
- `python -m evaluation --start YYYY-MM-DD --end YYYY-MM-DD` 같은 진입점으로 위 지표 표가 출력됨
- 베이스라인 대비 LLM 점수 성능이 한눈에 비교됨
- 컷오프 전/후 구간이 분리되어 표시됨

---

## Phase 2. 데이터 품질

### 2-1. 기사 본문 확보
- GKG의 `DocumentIdentifier`(URL)로 본문 앞부분(최대 ~800자) 추출: `trafilatura`
- 실패 시 제목만 사용 (fallback), 캐시 저장(재수집 방지)

### 2-2. 중복 제거
- 제목(+본문 앞부분) 임베딩 → 코사인 유사도 클러스터링(임계값 config화)
- 클러스터당 대표 기사 1건 + `cluster_size`(보도량 = 관심도 피처)

### 2-3. 관련도 사전 필터
- 저렴한 모델(또는 키워드 + 임베딩 유사도)로 관련도 1차 필터 → 상위 K개만 본 스코어링에 투입

### 2-4. 무료 추가 소스 (각각 on/off 가능하게 config화)
| 소스 | 용도 | 라이브러리/API |
|---|---|---|
| OpenDART | 공시(실적, 자사주, 지분 변동) — 강한 신호 | OpenDART API (무료 키) |
| 네이버 뉴스 검색 API | 국문 뉴스 | 네이버 개발자센터 (무료 키) |
| KRX 수급 | 외국인/기관 순매수 | `pykrx` |
| 해외 업황 | `^SOX`, `NVDA`, `MU`, `TSM` 전일 수익률 | `yfinance` |
| 환율 | 원/달러 | `yfinance`(`KRW=X`) 또는 FRED |

API 키는 `.env`로 관리하고 코드/리포지토리에 하드코딩 금지.

**Acceptance**
- 하루치 입력 기사 수가 중복 제거 후 감소하고, 각 기사에 `cluster_size`, `body_snippet`(선택) 포함
- 소스별 on/off가 config로 동작

---

## Phase 3. 구조 개선 — LLM = 피처 추출기

### 3-1. 기사 단위 구조화 추출
기존 "하루치 뉴스 → 점수" 대신 **기사별** 구조화 출력. (`reasoning`을 첫 필드로 두어 근거 먼저 생성)

```python
class ArticleFeature(BaseModel):
    article_id: str
    reasoning: str
    relevance: float                      # 0~1
    category: Literal["samsung_direct", "semi_ai", "macro", "irrelevant"]
    event_type: Literal["earnings", "guidance", "hbm_qualification", "customer_win_loss",
                        "capex", "export_control", "tariff_trade", "rates", "fx",
                        "litigation_regulation", "labor", "shareholder_return",
                        "supply_chain", "geopolitics", "other"]
    direction: Literal[-1, 0, 1]
    magnitude: Literal[1, 2, 3]           # 아래 기준표 사용
    novelty: Literal["new", "follow_up", "rehash"]
    downside_risk: Literal[0, 1, 2, 3]    # 방향과 별개의 급락 위험 기여도
    one_line_summary: str                 # 경보 메시지용 한 줄 요약(한국어)
```

### 3-2. 점수 기준표(앵커)를 프롬프트에 명시
```
magnitude 3: 실적/가이던스 컨센서스 대비 큰 서프라이즈, 핵심 고객 공급 확정·이탈, 신규 수출규제
magnitude 2: 의미 있는 업황 변화, 경쟁사 주요 이벤트, 정책 방향 변화
magnitude 1: 우호/비우호적이나 예상된 내용, 업황 일반론, 반복 보도
downside_risk 3: 수출규제·제재, 실적 쇼크, 대형 소송/리콜, 지정학 급변
```

### 3-3. 이전 맥락 주입
- 프롬프트에 "최근 5거래일 주요 뉴스 요약" + "최근 5일 수익률/상대수익률"을 포함 → `novelty` 판단 근거 제공
- 미래 정보가 들어가지 않도록 `signal_date` 기준 과거 데이터만 사용 (테스트로 검증)

### 3-4. 일별 집계 피처
기사 피처 → 일 단위:
- 카테고리별 `sum(direction * magnitude * relevance)`, `novelty == "new"`만 별도 집계
- `max(downside_risk)`, `count(downside_risk >= 2)`, 이벤트 타입별 카운트
- 총 보도량, 보도량 z-score

### 3-5. 통계 모델 결합
- 입력: 일별 뉴스 피처 + 가격 피처(전일 SOX/NVDA/MU/TSM 수익률, 원/달러 변화, 외국인 순매수, 5/20일 모멘텀, 20일 변동성)
- 모델: 로지스틱 회귀(방향) / 릿지(수익률) 먼저, 이후 LightGBM 비교
- **워크포워드 학습**(확장 윈도우, 월 단위 재학습), 미래 데이터 누설 금지
- 출력 점수 보정: 롤링 60일 z-score + isotonic 보정으로 "상승확률" 형태 제공

### 3-6. 기존 페르소나/토론 구조 처리
- 삭제하지 말고 **ablation 비교** 가능하게 모드화: `SCORING_MODE = "single" | "self_consistency" | "debate" | "feature_model"`
  - `single`: 단일 호출
  - `self_consistency`: 단일 호출 N회(temperature 0.7) 평균, N은 config
  - `debate`: 현재 페르소나 + moderator
  - `feature_model`: 3-1 ~ 3-5
- Phase 1 평가 도구로 네 모드를 같은 기간에 비교한 리포트 생성
- 점수 분포 히스토그램 출력(토론 모드의 0 근처 수렴 여부 확인)

**Acceptance**
- 모드별 IC/적중률/분위수익률 비교표가 생성됨
- `feature_model` 모드가 워크포워드로만 학습·예측됨(누설 테스트 통과)

---

## Phase 4. 하방 위험 경보 + 근거 기사 요약 메시지 (신규 기능)

### 4-1. 목적
방향 예측보다 실용적 가치가 큰 **"오늘 하방 위험이 높다 + 그 근거 기사"** 를 메시지로 띄운다.
투자 권유가 아닌 정보 제공 알림이다.

### 4-2. 트리거 조건 (모두 config화)
아래 중 하나라도 충족하면 경보:
1. 종합 위험 점수 ≥ `ALERT_RISK_THRESHOLD` (예: `RISK_MAX`의 70%)
2. 위험 점수의 롤링 60일 z-score ≥ `ALERT_RISK_Z` (예: 2.0)
3. `relevance ≥ 0.6` 이고 `downside_risk == 3` 이고 `novelty == "new"` 인 기사가 1건 이상
4. (옵션) 가격 조건: 전일 `^SOX` 수익률 ≤ `ALERT_SOX_DROP` (예: -3%)

경보 등급:
- `HIGH`: 조건 3 충족, 또는 조건 1과 2 동시 충족
- `MEDIUM`: 그 외 조건 하나 충족

### 4-3. 근거 기사 선택
- 후보: `downside_risk >= 2` 이고 `relevance >= 0.5`인 기사
- 정렬: `downside_risk * relevance * (novelty 가중치: new=1.0, follow_up=0.6, rehash=0.2) * log1p(cluster_size)`
- 상위 `ALERT_TOP_K`(기본 3)건, 중복 클러스터는 대표 1건만

### 4-4. 메시지 생성
- 근거 기사 요약은 LLM 호출 1회로 생성(구조화 출력), 사실은 기사 내용 안에서만, 추측 금지
```python
class RiskAlert(BaseModel):
    level: Literal["HIGH", "MEDIUM"]
    headline: str                 # 한 줄 핵심 (예: "美 대중 반도체 장비 규제 확대 보도")
    risk_drivers: list[str]       # 위험 요인 2~3개, 각 1문장
    evidence: list[EvidenceItem]  # 근거 기사
    watch_points: list[str]       # 확인해야 할 후속 이벤트(공시, 발표 일정 등)

class EvidenceItem(BaseModel):
    article_id: str
    title: str
    url: str
    published_kst: str
    why_it_matters: str           # 1~2문장
```

메시지 템플릿(텍스트 렌더링):
```
[하방 위험 경보 · HIGH] 2026-10-23 (KST 기준 신호)
삼성전자 | 위험점수 8/10 (60일 z=2.4)

■ 핵심: 美 대중 반도체 장비 규제 확대 보도
■ 위험 요인
  - ...
  - ...
■ 근거 기사
  1) [제목](URL) · 10/22 21:10
     → 왜 중요한가: ...
  2) ...
■ 확인할 것: ...

※ 뉴스 기반 자동 분석이며 투자 권유가 아닙니다.
```

### 4-5. 알림 채널 (플러그형)
```python
class Notifier(Protocol):
    def send(self, alert: RiskAlert, rendered: str) -> None: ...
```
구현체:
- `ConsoleNotifier` (기본, 항상 활성)
- `FileNotifier`: `alerts/YYYY-MM-DD.md` 및 `alerts/alerts.jsonl`에 기록 (평가용 이력)
- `TelegramNotifier`, `SlackWebhookNotifier`: 토큰/웹훅 URL은 `.env`에서만 읽고, 미설정 시 자동 비활성
- `config.ALERT_CHANNELS = ["console", "file"]`

### 4-6. 스팸 방지
- 같은 클러스터(사건)에 대해 `ALERT_COOLDOWN_DAYS`(기본 2) 내 재경보 금지, 단 등급이 MEDIUM→HIGH로 오르면 허용
- 경보 이력은 `alerts.jsonl`로 관리

### 4-7. 그래프 통합
- 기존 스코어링 그래프 마지막에 `risk_alert` 노드 추가: `final`(또는 feature_model 결과) + 기사 피처 → 트리거 판단 → 메시지 생성 → notifier 전송
- 트리거 미충족 시 LLM 호출 없이 종료(비용 절감)
- `ScoringState`에 `alert: dict | None` 필드 추가

### 4-8. 경보 성능 평가 (Phase 1 도구 확장)
- 정답 이벤트: 향후 `ALERT_EVAL_DAYS`(예: 3) 거래일 내 초과수익 누적 ≤ -2σ 또는 최저 수익률 ≤ `ALERT_EVAL_DROP`
- 지표: 정밀도(경보 중 실제 하락 비율), 재현율, 경보 빈도(월 평균), 무작위 경보 대비 lift
- 등급별(HIGH/MEDIUM)로 분리 리포트

**Acceptance**
- 과거 날짜를 지정해 실행하면 트리거 충족일에 콘솔 + `alerts/` 파일로 메시지가 생성됨
- 근거 기사마다 URL·발행시각(KST)·`why_it_matters` 포함
- 쿨다운이 동작함(같은 사건 연속 경보 없음)
- 경보 정밀도/재현율/lift 리포트가 출력됨

---

## Phase 5. 운영·기타

- LLM 호출 결과(기사 피처) 캐시: `(article_id, prompt_version, model)` 키로 저장 → 재실험 비용 절감
- `PROMPT_VERSION`을 config에 두고 결과에 기록
- 기본 temperature 0(추출), self-consistency 모드에서만 상향
- 테스트:
  - 시점 누설 테스트(신호일 이후 데이터 미포함)
  - KST 컷오프 경계 테스트
  - 트리거/쿨다운 단위 테스트
  - 구조화 출력 파싱 실패 시 재시도/폴백

---

## 작업 순서 요약

1. Phase 1 — 평가 체계 (라벨, 다중 호라이즌, IC, 베이스라인, 컷오프 분리)
2. Phase 4 — 하방 위험 경보 (기존 `final` 위험 점수로 먼저 동작시키고, Phase 3 이후 기사 피처 기반으로 강화)
3. Phase 2 — 데이터 품질 (본문, 중복 제거, DART/네이버/수급)
4. Phase 3 — 기사별 피처 추출 + 통계 모델 + 모드별 ablation
5. Phase 5 — 캐시, 버전 관리, 테스트

## 주의사항

- 대형주 방향 예측은 본질적으로 어렵다. 성능 개선 여부는 반드시 **컷오프 이후 구간 + 겹침 보정 통계**로 판단한다.
- 모든 출력 메시지에 "투자 권유 아님" 문구를 포함한다.
- API 키·토큰은 `.env`로만 관리하고 커밋하지 않는다(`.gitignore` 확인).
