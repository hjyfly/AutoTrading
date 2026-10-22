# judge 는 버전 비교의 기준이므로 상수로 고정한다. 바꾸면 이전 평가 결과와 비교할 수 없다
# judge 는 Sonnet 4.5 고정(채점·에이전트는 2026-10-22 부터 Sonnet 4.6). 최종 검증 때는 다른 모델로 바꾼다 (모델 미정)
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"  # 채점 모델을 4.6 으로 옮겨도 judge 는 고정(2026-10-22 사용자 확인)
JUDGE_REGION = "us-east-1"  # "us." 교차 리전 프로필은 미국 리전에서 호출한다
JUDGE_TEMPERATURE = 0

# 뉴스 채점 모델. .env 의 BEDROCK_MODEL_ID 가 없으면 이 값을 쓴다
# 2026-10-21 마이그레이션: us. 프로필이 요청 제한으로 호출당 2~5분 걸려 같은 Sonnet 4.5 의 global. 교차 리전 프로필로 옮겼다.
# us. 경로로 채점한 판단 기록은 data/scores/traces_archive/us_sonnet45/ 에 보관했다. judge 는 기존 그대로 둔다(사용자 결정)
# 2026-10-22: 채점·에이전트 모델을 Sonnet 4.6(global. 프로필, ID 에 날짜 없음)으로 바꾼다(사용자 결정). 유출 테스트를 다시 돌리고 뉴스 구간 전체를 다시 채점한다
SCORING_MODEL_DEFAULT = "global.anthropic.claude-sonnet-4-6"

# 데이터 기간 (SERVICE.md 섹션 5)
DATA_START = "2023-10-23"
DATA_END_INITIAL = "2026-10-20"  # 처음 수집한 끝날
# 주간 갱신(src/pipeline/refresh_data.py) 결과. 있으면 저장 데이터 끝날(DATA_END)은 이 파일의 data_end 를 따른다
DATA_STATUS_PATH = "data/processed/data_status.json"


def data_status_file():
    """갱신 결과 파일의 경로(실행 위치와 상관없이 프로젝트 폴더 기준)."""
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / DATA_STATUS_PATH


def _refreshed_data_end() -> str | None:
    """주간 갱신이 남긴 저장 데이터 끝날. 없거나 읽지 못하면 None."""
    import json
    try:
        return json.loads(data_status_file().read_text(encoding="utf-8"))["data_end"]
    except (OSError, ValueError, KeyError):
        return None


DATA_END = _refreshed_data_end() or DATA_END_INITIAL
# 1~3일차 공식 기준 A 판정의 끝날. 데이터가 늘어도 공식 판정은 이 날까지로 계산하고, 늘어난 구간은 참고로 따로 낸다
OFFICIAL_VALID_END = DATA_END_INITIAL
# 뉴스 수집·채점 구간 (2026-10-22 사용자 결정으로 2~4월에서 앞뒤 6개월 확장)
NEWS_START = "2025-11-01"
NEWS_END = "2026-10-20"  # 2026-10-22 사용자 결정: 8~10월 추가(10-21 은 실시간 기록이라 제외)
# 공식 기준 B 판정 구간과 유출 테스트 구간 (1~3일차 그대로 고정. 뉴스 구간을 넓혀도 바꾸지 않는다)
CRITERION_B_START = "2026-02-01"
CRITERION_B_END = "2026-04-30"

# 데이터 저장 위치. data/raw 는 이미 저장한 행을 수정하지 않는다(주간 갱신은 새 날짜 행만 덧붙이고, 덮어쓰기 전 원본을 data/backup 에 복사한다)
RAW_DIR = "data/raw"
RAW_BACKUP_DIR = "data/backup"
REFRESH_OVERLAP_DAYS = 10   # 갱신 때 저장 끝날 앞쪽으로 겹쳐 받아 제공처 값 변화를 점검하는 달력 일수
REFRESH_WEEKDAY = 4         # 평일 자동 실행 중 이 요일(4=금)에 저장 데이터를 갱신한다(모니터링 리포트 직전)
PROCESSED_DIR = "data/processed"

# 비공식 출처 호출 사이 대기 시간(초)
REQUEST_PAUSE_SEC = 0.5

# 정답 기준 (SERVICE.md 섹션 4)
# 2026-10-22 3거래일을 시험했으나 매주 재학습과 함께 쓰면 나빠져 5거래일로 되돌림(사용자 결정)
LABEL_HORIZON_DAYS = 5
LABEL_THRESHOLD = 0.03
OFFICIAL_LABEL_HORIZON_DAYS = 5  # 1~3일차 공식 판정의 정답 기간(공식 숫자 재현용, 바꾸지 않는다)
LABELS_OFFICIAL_PATH = "data/processed/labels_official.parquet"

# 워크포워드 구간 (SERVICE.md 섹션 8, 기본값). 처음 18개월은 학습에만 쓴다
TRAIN_END = "2025-04-22"
VALID_START = "2025-04-23"
# 재학습 주기. 2026-10-22 사용자 결정: 매월 → 매주. 1~3일차 공식 판정은 매월(OFFICIAL_RETRAIN_FREQ)
RETRAIN_FREQ = "W"
OFFICIAL_RETRAIN_FREQ = "M"
RETRAIN_FREQS = {"W": "매주", "M": "매월", "Q": "분기"}
# 학습 창(년). None = 처음부터 누적(지금 운영). DAY4 D4-10: 조건 비교에서 최근 2·3년만 쓰는 경우와 비교한다
TRAIN_WINDOW_YEARS = None
# 체결 시점 (기본 t 종가 = 공식 규칙). next_open 은 신호를 낸 다음 거래일 시가에 체결하는 비교용
EXECUTION = "close"
EXECUTIONS = {"close": "t 종가", "next_open": "t+1 시가"}

# 신호 시각. 뉴스는 t-1일 이 시각부터 t일 이 시각 전까지를 t일 뉴스로 본다 (SERVICE.md 섹션 6)
SIGNAL_CUTOFF_KST = "15:15"
KST = "Asia/Seoul"

# 시장별 (시간대, 종가 확정 시각). 실시간 조회는 종가가 확정된 날까지만 받는다
MARKET_CLOSE = {"prices": ("Asia/Seoul", "15:30"), "hstech": ("Asia/Hong_Kong", "16:10"),
                "dxy": ("America/New_York", "17:15"), "wti_fut": ("America/New_York", "17:15"),  # ICE·NYMEX 선물은 17:00 ET 마감
                "usdkrw_yf": ("UTC", "23:59")}  # yfinance 환율 일봉은 UTC 하루 단위로 끝난다
US_MARKET_CLOSE = ("America/New_York", "16:30")
KRX_LATE_CLOSE = {"2026-11-19": "16:30"}  # 수능일 등 늦게 마감하는 날(KRX 공지 기준으로 추가한다)

# Google 뉴스 RSS 검색을 신호 구간보다 며칠 앞에서 시작할지. 발행일을 하루 넘게 늦춰 배정하므로 넓게 잡는다
RSS_LOOKBACK_DAYS = 4

# 종목
SAMSUNG_CODE = "005930"
INVERSE_ETF_CANDIDATES = {"114800": "KODEX 인버스", "123310": "TIGER 인버스"}
INVERSE_ETF_CODE = "114800"  # D1-03: 삼성전자 수익률과 상관 -0.908 (TIGER 인버스 -0.901)
# 하락 신호 때 행동 (사용자 결정 2026-10-22): "cash" = 삼성전자 매도 후 현금, "inverse" = 인버스 ETF 로 전환(1~3일차 규칙)
# 이미 검증 구간을 본 뒤의 변경이라 과거 재계산은 탐색 결과로만 보고, 확정은 실시간 성적표(앞으로의 성과)로 한다
DOWN_ACTION = "cash"
DOWN_ACTIONS = ["cash", "inverse"]  # 성적표·보고서에서 비교로 함께 계산하는 규칙
DOWN_ACTION_LABELS = {"cash": "현금", "inverse": "인버스"}
OFFICIAL_DOWN_ACTION = "inverse"  # 1~3일차 공식 판정에 쓴 규칙
DOWN_ACTION_CHANGED_ON = "2026-10-22"
# 분할 매매 (2026-10-22 사용자 요청, 조건 비교 실험용. 운영 규칙은 전량 매매 그대로)
# 상승 신호마다 삼성전자 비중을 1/POSITION_STEPS 씩 늘리고 하락 신호마다 같은 만큼 줄인다. 보합은 유지한다
# POSITION_FLOOR 는 하락 신호가 이어져도 남겨 두는 기본 보유 비중이다. 1단계·바닥 0 이면 지금 규칙(전량)과 같다
POSITION_STEPS = 1
POSITION_FLOOR = 0.0
DART_CORP_CODE = "00126380"  # 삼성전자

# 거시·해외 지표 출처 ID
# FRED 는 공개가 늦어(WTI 4~5영업일, 환율 주 1회, 금리·VIX 다음 영업일 오후) 15:15 에 볼 수 있는 yfinance 값을 쓴다
# (WTI·환율은 사용자 결정, 금리·VIX 는 같은 이유로 leak-guard 2차 점검 뒤 교체. 2026-10-21)
FRED_SERIES = {}
YF_TICKERS = {
    "us10y_yf": "^TNX",
    "vix_yf": "^VIX",
    "wti_fut": "CL=F",
    "usdkrw_yf": "KRW=X",
    "sox": "^SOX",
    "nvda": "NVDA",
    "hstech": "3033.HK",  # 항셍테크 지수는 yfinance 에 없어 추종 ETF(CSOP 항셍테크)로 대신한다
}
# 확장 지표용 해외 시계열 (2026-10-21 사전 등록, 후보 모델 전용)
YF_TICKERS_EXT = {
    "mu": "MU",          # 마이크론 (메모리 동종)
    "tsm": "TSM",        # TSMC (파운드리 경쟁)
    "dxy": "DX-Y.NYB",   # 달러 인덱스
}
# 확장 지표용 국내 시계열. 국내 값이라 t-1 종가까지만 쓴다
KR_EXTRA = {"kospi": "KS11", "kospi200": "KS200", "hynix": "000660"}

# 수익률 대신 수준과 차이로 특징을 만드는 시계열 (금리·변동성)
LEVEL_SERIES = {"us10y_yf", "vix_yf"}

# 개발 중 샘플 채점 요일 (SERVICE.md 섹션 7). 0=월 … 2=수. 휴장이면 다음 거래일
SAMPLE_WEEKDAY = 2

# 뉴스 수집 동시 다운로드 수 (GDELT GKG 원본 파일)
NEWS_DOWNLOAD_WORKERS = 4

# 뉴스 키워드
NEWS_KEYWORDS_KO = ["삼성전자", "반도체", "HBM", "파운드리", "유가", "금리", "연준", "관세", "중국 반도체", "엔비디아"]
NEWS_KEYWORDS_EN = ["Samsung Electronics", "semiconductor", "HBM", "foundry", "oil price", "Federal Reserve", "tariff", "China chip", "Nvidia"]

# 기사 선별 (DAY2_PLAN.md 3절, 기본값). 제목에 들어 있는 단어로 분류하고 분류별로 할당한다
SELECT_TOTAL = 30
SELECT_QUOTAS = {"samsung": 12, "semis": 10, "macro": 8}  # 앞에 있는 분류부터 부족분을 채운다
SELECT_PATTERNS = {
    "samsung": r"삼성|samsung",
    # 'AI' 단독은 일반 AI 기사가 너무 많이 걸려 넣지 않는다 (2일차 선별 결과 확인 뒤 제외)
    "semis": r"반도체|semiconductor|hbm|파운드리|foundry|엔비디아|nvidia|\bchips?\b|메모리|memory|하이닉스|hynix|tsmc|마이크론|micron|d램|dram|낸드|nand",
    "macro": r"유가|\boil\b|금리|\bfed\b|연준|관세|tariff|중국|china|환율|달러|dollar|inflation|인플레",
}
SELECT_MAX_PER_SOURCE = 3
SELECT_DEDUP_PREFIX = 30  # 제목 앞 N 글자가 같으면 같은 기사로 본다
TITLE_MAX_CHARS = 200

# 뉴스 채점 (SERVICE.md 섹션 7)
SCORE_MIN, SCORE_MAX = -5, 5
SCORING_PERSONAS = {
    "short_trader": "단기 트레이더. 앞으로 1주일 안의 가격 반응, 모멘텀, 수급과 투자심리에 영향을 주는 뉴스를 중시한다.",
    "value_investor": "장기 가치투자자. 실적, 산업 구조, 경쟁력 같은 펀더멘털 변화를 중시하고 단기 소음은 할인한다.",
    "risk_manager": "리스크 관리자. 하방 위험, 거시 충격, 정책·지정학 불확실성을 중시하고 낙관을 경계한다.",
}
SCORES_PATH = "data/scores/news_scores.parquet"
# 채점 형식 버전. 2 = 분류별 점수(삼성 직접·반도체·거시)와 하락 위험 점수 추가 (2026-10-22 사용자 결정). 버전이 섞이면 점수 표를 만들지 않는다
SCORING_VERSION = 2
NEWS_CATEGORIES = {"samsung": "삼성전자 직접 뉴스", "semis": "반도체·AI 업황 뉴스", "macro": "거시(유가·금리·환율·관세·중국) 뉴스"}
RISK_MIN, RISK_MAX = 0, 5
THROTTLE_RETRIES = 10  # Bedrock 요청 제한(Throttling) 때 다시 시도하는 횟수
THROTTLE_WAIT_SEC = 60
TRACES_DIR = "data/scores/traces"

# 미래 정보 유출 테스트 (DAY2_PLAN.md 3절, 기본값)
# 채점 모델을 바꾸면 반드시 다시 돌린다. 지식 기준 시점이 2026년 2~4월보다 늦은 모델은 기억으로 결과를 알 수 있다
LEAK_TEST_DAYS = 20
LEAK_TEST_ALPHA = 0.05

# 차트 지표 설정값 (사용자 확인, 2026-10-21)
MA_WINDOWS = [5, 20, 60, 120]
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
ICHIMOKU = {"tenkan": 9, "kijun": 26, "senkou_b": 52}
PSAR_STEP = 0.02
PSAR_MAX = 0.2
VOLUME_MA_WINDOW = 20

# ---- 지표 묶음 (2026-10-21 사전 등록. 후보 모델 결과를 보기 전에 고정한다. 바꾸지 않는다) ----
# A. 가격 기반 기술 지표
RSI_WINDOW = 14
MACD = {"fast": 12, "slow": 26, "signal": 9}
STOCH = {"k": 14, "d": 3}
ATR_WINDOW = 14
VOLATILITY_WINDOW = 20
OBV_WINDOW = 20
HIGH_LOW_WINDOW = 120  # 52주 대신 120일 (52주는 첫 학습 데이터가 너무 줄어든다)
# B. 시장·동종 종목: KR_EXTRA, YF_TICKERS_EXT 의 1·5일 수익률, 코스피 대비 삼성 상대 강도 5·20일
RELATIVE_STRENGTH_WINDOWS = [5, 20]
RELATIVE_STRENGTH_BENCHMARK = "kospi"  # KR_EXTRA 의 이름
# D. 일정: 옵션 만기일(매월 둘째 목요일), 만기까지 남은 거래일 수, 월말, 요일, 잠정실적 발표 시기(분기 첫 달 1~10일)
EARNINGS_MONTHS = [1, 4, 7, 10]
OPTION_EXPIRY_WEEKDAY = 3  # 목요일 (월=0)
OPTION_EXPIRY_NTH = 2  # 둘째 주
EARNINGS_DAY_MAX = 10

# 1~3일차 모델 특징 45개 (1~3일차 결과와 같은 모델을 다시 만들 수 있게 이름으로 고정한다)
FEATURE_SET_BASE_V1 = [
    "ret_1d", "ret_5d", "ret_20d", "volume_ratio", "psar_gap", "psar_trend",
    "ma5_gap", "ma20_gap", "ma60_gap", "ma120_gap", "bb_pctb", "bb_width",
    "ichi_tenkan_gap", "ichi_kijun_gap", "ichi_tenkan_kijun", "ichi_cloud_top_gap", "ichi_cloud_bottom_gap", "ichi_cloud_thickness",
    "foreign_net_ratio", "institution_net_ratio", "foreign_net_ratio_5d", "foreign_hold_change_5d",
    "us10y_yf_level", "us10y_yf_chg_1d", "us10y_yf_chg_5d", "us10y_yf_age_days",
    "vix_yf_level", "vix_yf_chg_1d", "vix_yf_chg_5d", "vix_yf_age_days",
    "wti_fut_ret_1d", "wti_fut_ret_5d", "wti_fut_age_days", "usdkrw_yf_ret_1d", "usdkrw_yf_ret_5d", "usdkrw_yf_age_days",
    "sox_ret_1d", "sox_ret_5d", "sox_age_days", "nvda_ret_1d", "nvda_ret_5d", "nvda_age_days",
    "hstech_ret_1d", "hstech_ret_5d", "hstech_age_days",
]
FEATURE_SET_A = ["rsi14", "macd_line", "macd_hist", "stoch_k", "stoch_d", "atr_ratio", "volatility_20d", "obv_change_20d",
                 "hl120_position", "gap_open"]
FEATURE_SET_B = ["kospi_ret_1d", "kospi_ret_5d", "kospi200_ret_1d", "kospi200_ret_5d", "hynix_ret_1d", "hynix_ret_5d",
                 "rel_strength_5d", "rel_strength_20d",
                 "mu_ret_1d", "mu_ret_5d", "mu_age_days", "tsm_ret_1d", "tsm_ret_5d", "tsm_age_days",
                 "dxy_ret_1d", "dxy_ret_5d", "dxy_age_days"]
FEATURE_SET_D = ["is_option_expiry", "days_to_option_expiry", "is_month_end", "weekday", "is_earnings_season"]
# 운영 모델 특징 46개 = 1~3일차 45개 + RSI(14일) (2026-10-22 사용자 결정. 검증 구간을 본 뒤의 변경이라 그 구간 숫자는 탐색으로 본다)
FEATURE_SET_BASE = FEATURE_SET_BASE_V1 + ["rsi14"]
BASE_CHANGED_ON = "2026-10-22"
FEATURE_SETS = {
    "base": FEATURE_SET_BASE,
    "base_v1": FEATURE_SET_BASE_V1,
    "candidate": FEATURE_SET_BASE_V1 + FEATURE_SET_A + FEATURE_SET_B + FEATURE_SET_D,  # 사전 등록 목록 그대로(77개)
}
SIGNALS_CANDIDATE_PATH = "data/processed/signals_candidate.parquet"
SIGNALS_BASE_V1_PATH = "data/processed/signals_base_v1.parquet"  # 1~3일차 45개 모델 신호(공식 판정 재현용)

# 지표 그룹 (2026-10-22 사용자 요청: 그룹 단위로 빼고 더해 보는 진단). 운영 46개와 후보 A·B·D 를 겹치지 않게 나눈다
# 그룹 묶음 이름은 "groups:price+chart+flow" 처럼 FEATURE_GROUP_PREFIX 뒤에 그룹을 + 로 잇는다
FEATURE_GROUPS = {
    "price": ["ret_1d", "ret_5d", "ret_20d", "volume_ratio"],
    "chart": ["psar_gap", "psar_trend", "ma5_gap", "ma20_gap", "ma60_gap", "ma120_gap", "bb_pctb", "bb_width",
              "ichi_tenkan_gap", "ichi_kijun_gap", "ichi_tenkan_kijun", "ichi_cloud_top_gap", "ichi_cloud_bottom_gap", "ichi_cloud_thickness"],
    "flow": ["foreign_net_ratio", "institution_net_ratio", "foreign_net_ratio_5d", "foreign_hold_change_5d"],
    "macro": ["us10y_yf_level", "us10y_yf_chg_1d", "us10y_yf_chg_5d", "vix_yf_level", "vix_yf_chg_1d", "vix_yf_chg_5d",
              "wti_fut_ret_1d", "wti_fut_ret_5d", "usdkrw_yf_ret_1d", "usdkrw_yf_ret_5d"],
    "peer": ["sox_ret_1d", "sox_ret_5d", "nvda_ret_1d", "nvda_ret_5d", "hstech_ret_1d", "hstech_ret_5d"],
    "age": ["us10y_yf_age_days", "vix_yf_age_days", "wti_fut_age_days", "usdkrw_yf_age_days", "sox_age_days", "nvda_age_days",
            "hstech_age_days"],
    "rsi": ["rsi14"],
    "tech_extra": [c for c in FEATURE_SET_A if c != "rsi14"],
    "market_extra": FEATURE_SET_B,
    "calendar": FEATURE_SET_D,
}
FEATURE_GROUP_LABELS = {
    "price": "가격·거래량", "chart": "차트(이동평균·볼린저·일목·PSAR)", "flow": "수급(외국인·기관)",
    "macro": "거시(금리·VIX·유가·환율)", "peer": "해외 동종(SOX·엔비디아·항셍테크)", "age": "데이터 경과일", "rsi": "RSI(14일)",
    "tech_extra": "추가 기술 지표(A, RSI 제외)", "market_extra": "시장·동종 추가(B)", "calendar": "일정(D)",
}
FEATURE_GROUPS_BASE = ["price", "chart", "flow", "macro", "peer", "age", "rsi"]  # 합치면 FEATURE_SET_BASE
FEATURE_GROUP_PREFIX = "groups:"
# 지표 그룹 진단의 추천 규칙(결과를 보기 전에 정한 고정 규칙): 정확도·수익·MDD 중 어느 것도 허용 폭 넘게 나빠지지 않고
# 하나 이상이 허용 폭 넘게 좋아진 변경만 추천한다. 수익·MDD 허용 폭은 MONITOR_TOL, 정확도 허용 폭은 아래 값
# (2026-10-22 실제 데이터로 진단을 처음 돌리기 전에 정함. 이후 바꾸면 사후 조정이라고 리포트에 적는다)
FEATURE_STUDY_ACC_TOL = 0.01
FEATURE_STUDY_REPORT_PATH = "evaluation/reports/experiments_features.md"
# 기존·후보 모델 말고 실시간 성적표에서 함께 추적할 지표 묶음 (사용자가 진단 결과를 보고 고른 뒤 추가한다. 최종 신호는 바꾸지 않는다)
# 2026-10-22 사용자 결정: 진단 추천 묶음(데이터 경과일 7개 제외)을 추적한다
LIVE_EXTRA_FEATURE_SETS: list[str] = ["groups:price+chart+flow+macro+peer"]

# 특징 계산 기간(거래일)
RETURN_WINDOWS = [1, 5, 20]  # 삼성전자 수익률
FLOW_WINDOW = 5  # 순매수 합계·보유비율 변화
FOREIGN_CHANGE_WINDOWS = [1, 5]  # 해외·거시 지표 변화

# 모의 매매 (SERVICE.md 섹션 9)
FEE_RATE = 0.001  # 매수·매도 각각 거래 금액의 0.1%
SIGNALS_NUMERIC_PATH = "data/processed/signals_numeric.parquet"

# 숫자 모델 (DAY3_PLAN.md 3절, 기본값). 검증 구간에 맞춰 조정하지 않는다
RANDOM_SEED = 42
LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "random_state": RANDOM_SEED,
    "deterministic": True,  # 같은 데이터면 같은 결과
    "force_row_wise": True,
    "verbose": -1,
}

MIN_TRAIN_ROWS = 120  # 재학습에 필요한 최소 학습 행

# 매일 자동 실행과 알림 (사용자 결정: Gmail)
DAILY_RUN_TIME_KST = "15:20"  # 작업 스케줄러 등록 시각. 뉴스 구간(~15:15)이 닫힌 뒤
LOG_DIR = "data/logs"
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
NAVER_BASIC_URL = "https://m.stock.naver.com/api/stock/{code}/basic"  # 오늘 거래 여부 확인용(마지막 체결 시각)

# 실시간 성적표 (②안 앞으로의 성과 비교, 기본값)
SCOREBOARD_PATH = "data/processed/scoreboard.json"
DRIFT_WINDOW = 20  # 최근 정답 확정 신호 N 개로 성능 저하를 본다
# 경고: 최근 N 개 3분류 정확도가 같은 기간 '항상 가장 많은 정답' 비율보다 낮으면 (모델이 단순 규칙보다 못함)

# 뉴스 보정 규칙 (DAY3_PLAN.md 3절 방법 A, 사용자 결정). 결과를 보기 전에 고정한다
NEWS_ADJUST_SCORE = 3
NEWS_ADJUST_MIN_CONF = 0.6
# 뉴스 보정에 쓰는 점수 열·평균 기간·하락 위험 거부 기준 (기본값 = 1~3일차 규칙 그대로. 조건 비교 실험에서 바꿔 본다)
NEWS_SCORE_FIELD = "news_consensus"
NEWS_SCORE_FIELDS = {"news_consensus": "종합 합의", "news_samsung_score": "삼성 직접", "news_semis_score": "반도체", "news_macro_score": "거시"}
NEWS_WINDOW_DAYS = 1      # 최근 k개 채점일 평균 (1 = 그날 점수만)
NEWS_RISK_VETO = None     # 하락 위험이 이 값 이상이면 상승 신호를 보합으로. None = 쓰지 않음

# 조건 비교 실험(src/pipeline/experiment.py)의 기준 조건 = 지금 운영 설정. 결과는 모두 탐색이며 공식 판정을 바꾸지 않는다
EXPERIMENT_BASE = {
    "horizon_days": LABEL_HORIZON_DAYS, "threshold": LABEL_THRESHOLD, "feature_set": "base", "retrain_freq": RETRAIN_FREQ,
    "execution": EXECUTION, "down_action": DOWN_ACTION, "position_steps": POSITION_STEPS, "position_floor": POSITION_FLOOR,
    "train_window": TRAIN_WINDOW_YEARS, "news_mode": "none", "news_field": NEWS_SCORE_FIELD,
    "news_window": NEWS_WINDOW_DAYS, "news_risk_veto": NEWS_RISK_VETO, "news_min_score": NEWS_ADJUST_SCORE,
    "news_min_conf": NEWS_ADJUST_MIN_CONF, "start": None, "end": None,
}
EXPERIMENT_DIR = "data/experiments"
EXPERIMENT_REPORT_DIR = "evaluation/reports"   # 조건 비교·모니터링 보고서 폴더
EXPERIMENT_MAX_COMBOS = 48                     # 화면에서 한 번에 돌릴 수 있는 조합 수 상한
# 화면 "조건 비교" 탭의 숫자 조건 선택지 (고를 수 있는 값이 정해진 조건은 위 사전에서 가져온다)
EXPERIMENT_UI_OPTIONS = {
    "horizon_days": [3, 5, 10, 20], "threshold": [0.02, 0.03, 0.05], "position_steps": [1, 2, 3, 4],
    "position_floor": [0.0, 0.25, 0.5], "train_window": [None, 2, 3], "news_window": [1, 3, 5],
    "news_risk_veto": [None, 3, 4], "news_min_score": [2, 3, 4], "news_min_conf": [0.5, 0.6, 0.7],
}
# 조건 모니터링: 기준 조건에서 하나씩만 바꿔 볼 값(사전 등록. 결과를 보고 목록을 바꾸지 않는다). 2026-10-22 사용자 요청
MONITOR_AXES = {
    "threshold": [0.02, 0.05], "horizon_days": [3, 10], "retrain_freq": ["M", "Q"],  # 2026-10-22 기준 재학습이 매주로 바뀌어 비교 값을 매월·분기로 바꿈
    "execution": ["next_open"], "down_action": ["inverse"], "feature_set": ["candidate"],
}
MONITOR_NEWS_AXES = {  # 뉴스 묶음의 기준은 news_mode="adjust"(기준 B 규칙). 분류별·하락 위험은 채점 버전 2 에서만 계산된다
    "news_mode": ["none"], "news_min_score": [2, 4], "news_min_conf": [0.7], "news_window": [3, 5],
    "news_field": ["news_samsung_score", "news_semis_score", "news_macro_score"], "news_risk_veto": [3, 4],
}
MONITOR_RECENT_DAYS = [20, 60, 120]     # 리포트에 따로 보여 줄 최근 구간(거래일)
MONITOR_TOL = 0.005                     # 기준 대비 수익·MDD 차이가 이 안이면 '거의 같다'(0.5%p)
MONITOR_MONTH_TOL = 1e-4                # 달별 수익 차이가 이 안이면 같은 달로 본다
MONITOR_STEADY_RATIO, MONITOR_STEADY_MIN = 2, 3   # 나은(못한) 달이 반대보다 이 배 이상이고 이 개월 이상이면 '꾸준히'
MONITOR_MDD_MARGIN = 0.05               # 들고 있기보다 MDD 가 이만큼 넘게 얕으면 따로 알려 준다(5%p)
MONITOR_WEEKDAY = 4                     # 평일 자동 실행 중 이 요일(0=월 … 4=금)에 리포트를 다시 만든다
MONITOR_REPORT_PATH = "evaluation/reports/experiments_monitor.md"
MONITOR_HISTORY_PATH = "data/experiments/monitor_history.csv"

# ---- 4일차: 숫자 지표 과거 확장 검증 (DAY4_PLAN.md, 2026-10-22 사용자 결정: 기본값) ----
# 확장 원본은 기존 원본과 겹치지 않게 DATA_START 전날까지만 받아 data/raw 안 '<이름>_ext' 파일에 따로 둔다(기존 원본 불변)
EXT_DATA_START = "2019-01-02"
EXT_DATA_END = "2023-10-20"
EXT_SUFFIX = "_ext"
EXT_FIRST_TRAIN_MONTHS = 18          # 확장 특징 표에서 운영 지표가 모두 있는 첫날부터 이 기간은 학습에만 쓴다
EXT_VALID_END = "2025-04-22"         # 확장 검증 끝(기존 검증 시작 전날). 그 뒤는 이미 여러 번 본 구간이라 참고로만 싣는다
FEATURES_EXT_PATH = "data/processed/features_ext.parquet"
LABELS_EXT_PATH = "data/processed/labels_ext.parquet"
SIGNALS_EXT_PATH = "data/processed/signals_ext.parquet"
EXT_REPORT_PATH = "evaluation/reports/day4_extended.md"
FEATURE_STUDY_EXT_REPORT_PATH = "evaluation/reports/experiments_features_ext.md"  # 확장 검증 구간의 지표 그룹 진단
# 장세 구분(결과 보기 전에 고정): 그 구간 들고 있기 수익이 이 값 이상이면 상승장, 이하이면 하락장, 그 사이는 횡보장
REGIME_UP, REGIME_DOWN = 0.15, -0.15

# ---- 5일차: 하락 위험 모델·신뢰도 측정 (DAY5_PLAN.md, 2026-10-22 사용자 결정: 기본값) ----
# 확장 검증 구간을 개발·판정으로 나눈다. 판정 구간은 사전 등록 조건마다 한 번만 쓴다
DEV_START, DEV_END = "2022-03-04", "2023-12-31"
JUDGE_START, JUDGE_END = "2024-01-01", "2025-04-22"
RISK_HORIZON_DAYS = 5          # 하락 위험: t+1 ~ t+5 거래일 중 최저 종가가 t 종가 대비
RISK_THRESHOLD = 0.03          # -3% 이하로 한 번이라도 내려가면 1
RISK_WEIGHTS = (1.0, 0.5, 0.0) # 위험 낮음 / 중간 / 높음일 때 삼성전자 비중(나머지는 현금)
RISK_QUANTILES = (0.6, 0.85)   # 중간·높음 기준 = 개발 구간 위험 확률의 이 분위수(개발 구간에서만 정한다)
CALIBRATION_SHARE = 0.2        # 확률 보정: 재학습 때 학습 구간 마지막 이 비율로 보정기를 맞춘다
CALIBRATION_MIN_ROWS = 60      # 보정용 행이 이보다 적으면 보정하지 않는다
TREND_MA = "ma120_gap"         # 합의 모델의 추세 규칙: 이 이동평균 괴리가 0 미만이면 위험 표
LOGIT_C = 1.0                  # 로지스틱 회귀 규제 강도(기본값, 조정하지 않는다)
LOGIT_MAX_ITER = 2000
CALIBRATION_BINS = 10          # 신뢰도 단조 확인: 확률 0~1 을 이 개수로 나눈 구간(판정 때 값 그대로)
CALIBRATION_MIN_BIN = 10       # 표본이 이보다 적은 구간은 단조 확인에서 뺀다
UP_MONTH_MIN_RETURN = 0.05     # 반등장 확인: 그달 들고 있기 수익이 이 값 이상인 달
UP_MONTH_MIN_HOLD = 0.70       # 그런 달에 삼성전자 보유일 비율이 이 값 이상이어야 한다
LIVE_PROMOTION_MIN_SIGNALS = 40  # 운영 변경 검토에 필요한 실시간 확정 신호 수
PREREG_PATH = "evaluation/reports/preregistration.md"
JUDGED_PATH = "data/processed/day5_judged.json"   # 판정 구간을 이미 쓴 조건 기록(다시 판정하지 않는다)
DAY5_REPORT_PATH = "evaluation/reports/day5_judgement.md"
SIGNALS_RISK_PATH = "data/processed/signals_risk_ext.parquet"
# 실시간 성적표에 비교용으로 기록할 하락 위험 조건(2026-10-22 사용자: 판정 기준을 일부 통과한 P2). 운영 신호는 바꾸지 않는다
LIVE_RISK_CONDITION = "P2"
LIVE_RISK_MODEL = "risk_p2"
# 사전 등록 조건(2026-10-22 등록. 판정 결과를 보기 전에 고정. 바꾸려면 새 ID 로 등록한다)
PREREGISTERED = {
    "P1": "하락 위험 LightGBM(운영 46개·매주 재학습), 보정 없음, 3단계 비중",
    "P2": "P1 과 같고 확률 보정(isotonic) 적용",
    "P3": "합의: P2 위험 높음 표 + 로지스틱 회귀 위험 표 + 120일선 아래 표. 0~1표 100%, 2표 50%, 3표 0%",
}

# LLM 비용 한도 (SERVICE.md 섹션 10). 채점과 judge 호출을 합산한다
COST_LIMIT_POC_USD = 150.0  # 개발 단계 한도 (2026-10-22 $80 → $150, 사용자 결정)
COST_LIMIT_MONTHLY_USD = 100.0  # 운영 단계 월 한도 (운영 코드에서 쓴다)
COST_LOG_PATH = "data/scores/cost_log.csv"

# 100만 토큰당 달러 (입력, 출력). 모델 ID 에 키 문자열이 들어 있으면 그 가격을 쓴다
LLM_PRICES_PER_MTOK = {
    "claude-sonnet-4-5": (3.0, 15.0),  # Bedrock 표준가 (리전에 따라 더 비쌀 수 있다)
    "claude-sonnet-4-6": (3.0, 15.0),  # Anthropic 공식가 기준. Bedrock 가격을 확인하면 고친다
    "claude-sonnet-5-5": (2.0, 10.0),  # Anthropic 공식가. Bedrock 가격을 확인하면 고친다
    "titan-embed-text-v2": (0.02, 0.0),  # 임베딩. 기억에 의존한 값이라 AWS 가격표 확인 필요. 토큰 대신 글자 수로 넉넉히 기록한다
}

# RAG 색인 (src/retriever.py)
INDEX_DIR = "data/index"
INDEX_COLLECTION = "autotrading"
RAG_CHUNK_CHARS = 1500
RAG_TOP_K = 5
RAG_REPORTS = ["evaluation/reports/day1_data.md", "evaluation/reports/day3_backtest.md",
               "evaluation/reports/leak_test.md", "evaluation/reports/data_probe_20261021.md"]
