# judge 는 버전 비교의 기준이므로 상수로 고정한다. 바꾸면 이전 평가 결과와 비교할 수 없다
# 개발 중에는 채점 모델과 같은 Sonnet 4.5 를 쓰고, 최종 검증 때는 다른 모델로 바꾼다 (모델 미정)
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
JUDGE_REGION = "us-east-1"  # "us." 교차 리전 프로필은 미국 리전에서 호출한다
JUDGE_TEMPERATURE = 0

# 뉴스 채점 모델. .env 의 BEDROCK_MODEL_ID 가 없으면 이 값을 쓴다
# 2026-10-21 마이그레이션: us. 프로필이 요청 제한으로 호출당 2~5분 걸려 같은 Sonnet 4.5 의 global. 교차 리전 프로필로 옮겼다.
# us. 경로로 채점한 판단 기록은 data/scores/traces_archive/us_sonnet45/ 에 보관했다. judge 는 기존 그대로 둔다(사용자 결정)
SCORING_MODEL_DEFAULT = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"

# 데이터 기간 (SERVICE.md 섹션 5)
DATA_START = "2023-10-23"
DATA_END = "2026-10-20"
NEWS_START = "2026-02-01"
NEWS_END = "2026-04-30"

# 데이터 저장 위치. data/raw 는 수집 후 수정하지 않는다
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# 비공식 출처 호출 사이 대기 시간(초)
REQUEST_PAUSE_SEC = 0.5

# 정답 기준 (SERVICE.md 섹션 4)
LABEL_HORIZON_DAYS = 5
LABEL_THRESHOLD = 0.03

# 워크포워드 구간 (SERVICE.md 섹션 8, 기본값). 처음 18개월은 학습에만 쓴다
TRAIN_END = "2025-04-22"
VALID_START = "2025-04-23"

# 신호 시각. 뉴스는 t-1일 이 시각부터 t일 이 시각 전까지를 t일 뉴스로 본다 (SERVICE.md 섹션 6)
SIGNAL_CUTOFF_KST = "15:15"
KST = "Asia/Seoul"

# 시장별 (시간대, 종가 확정 시각). 실시간 조회는 종가가 확정된 날까지만 받는다
MARKET_CLOSE = {"prices": ("Asia/Seoul", "15:30"), "hstech": ("Asia/Hong_Kong", "16:10"),
                "dxy": ("America/New_York", "17:15"), "wti_fut": ("America/New_York", "17:15")}  # ICE·NYMEX 선물은 17:00 ET 마감
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

# 기존 모델 특징 45개 (1~3일차 결과와 같은 모델을 다시 만들 수 있게 이름으로 고정한다)
FEATURE_SET_BASE = [
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
FEATURE_SETS = {
    "base": FEATURE_SET_BASE,
    "candidate": FEATURE_SET_BASE + FEATURE_SET_A + FEATURE_SET_B + FEATURE_SET_D,
}
SIGNALS_CANDIDATE_PATH = "data/processed/signals_candidate.parquet"

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

# LLM 비용 한도 (SERVICE.md 섹션 10). 채점과 judge 호출을 합산한다
COST_LIMIT_POC_USD = 80.0  # 개발 단계 한도
COST_LIMIT_MONTHLY_USD = 100.0  # 운영 단계 월 한도 (운영 코드에서 쓴다)
COST_LOG_PATH = "data/scores/cost_log.csv"

# 100만 토큰당 달러 (입력, 출력). 모델 ID 에 키 문자열이 들어 있으면 그 가격을 쓴다
LLM_PRICES_PER_MTOK = {
    "claude-sonnet-4-5": (3.0, 15.0),  # Bedrock 표준가 (리전에 따라 더 비쌀 수 있다)
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
