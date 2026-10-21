# judge 는 버전 비교의 기준이므로 상수로 고정한다. 바꾸면 이전 평가 결과와 비교할 수 없다
# 개발 중에는 채점 모델과 같은 Sonnet 4.5 를 쓰고, 최종 검증 때는 다른 모델로 바꾼다 (모델 미정)
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
JUDGE_REGION = "us-east-1"  # "us." 교차 리전 프로필은 미국 리전에서 호출한다
JUDGE_TEMPERATURE = 0

# 뉴스 채점 모델. .env 의 BEDROCK_MODEL_ID 가 없으면 이 값을 쓴다
SCORING_MODEL_DEFAULT = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

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

# Google 뉴스 RSS 검색을 신호 구간보다 며칠 앞에서 시작할지. 발행일을 하루 넘게 늦춰 배정하므로 넓게 잡는다
RSS_LOOKBACK_DAYS = 4

# 종목
SAMSUNG_CODE = "005930"
INVERSE_ETF_CANDIDATES = {"114800": "KODEX 인버스", "123310": "TIGER 인버스"}
INVERSE_ETF_CODE = "114800"  # D1-03: 삼성전자 수익률과 상관 -0.908 (TIGER 인버스 -0.901)
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
# 수익률 대신 수준과 차이로 특징을 만드는 시계열 (금리·변동성)
LEVEL_SERIES = {"us10y_yf", "vix_yf"}

# 개발 중 샘플 채점 요일 (SERVICE.md 섹션 7). 0=월 … 2=수. 휴장이면 다음 거래일
SAMPLE_WEEKDAY = 2

# 뉴스 수집 동시 다운로드 수 (GDELT GKG 원본 파일)
NEWS_DOWNLOAD_WORKERS = 4

# 뉴스 키워드
NEWS_KEYWORDS_KO = ["삼성전자", "반도체", "HBM", "파운드리", "유가", "금리", "연준", "관세", "중국 반도체", "엔비디아"]
NEWS_KEYWORDS_EN = ["Samsung Electronics", "semiconductor", "HBM", "foundry", "oil price", "Federal Reserve", "tariff", "China chip", "Nvidia"]

# 차트 지표 설정값 (사용자 확인, 2026-10-21)
MA_WINDOWS = [5, 20, 60, 120]
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
ICHIMOKU = {"tenkan": 9, "kijun": 26, "senkou_b": 52}
PSAR_STEP = 0.02
PSAR_MAX = 0.2
VOLUME_MA_WINDOW = 20

# 특징 계산 기간(거래일)
RETURN_WINDOWS = [1, 5, 20]  # 삼성전자 수익률
FLOW_WINDOW = 5  # 순매수 합계·보유비율 변화
FOREIGN_CHANGE_WINDOWS = [1, 5]  # 해외·거시 지표 변화

# LLM 비용 한도 (SERVICE.md 섹션 10). 채점과 judge 호출을 합산한다
COST_LIMIT_POC_USD = 80.0  # 개발 단계 한도
COST_LOG_PATH = "data/scores/cost_log.csv"

# 100만 토큰당 달러 (입력, 출력). 모델 ID 에 키 문자열이 들어 있으면 그 가격을 쓴다
LLM_PRICES_PER_MTOK = {
    "claude-sonnet-4-5": (3.0, 15.0),  # Bedrock 표준가 (리전에 따라 더 비쌀 수 있다)
    "claude-sonnet-5-5": (2.0, 10.0),  # Anthropic 공식가. Bedrock 가격을 확인하면 고친다
}
