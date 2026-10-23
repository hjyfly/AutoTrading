"""종목 레지스트리·역할 설정(10일차) 테스트: 종목을 더하거나 역할을 바꾸면 코드를 고치지 않고 따라오는지."""
import numpy as np
import pandas as pd
import pytest

from src import config, tools
from src.pipeline import experiment as ex
from src.pipeline import multi_ticker as mt

NEW = "005380"


@pytest.fixture
def fifth_ticker(monkeypatch):
    """레지스트리에 가짜 5번째 종목을 넣는다."""
    monkeypatch.setattr(config, "MULTI_TICKERS", {**config.MULTI_TICKERS, NEW: ("현대차", "자동차", None)})
    return NEW


def test_주문_종목은_매매_종목_역할을_따른다():
    assert config.KIS_ORDER_CODES == config.TRADE_TICKERS and config.MAIN_TICKER in config.MULTI_TICKERS
    assert set(config.NEWS_TICKERS) <= set(config.MULTI_TICKERS) and set(config.TRADE_TICKERS) <= set(config.MULTI_TICKERS)


def test_새_종목은_이름과_코드로_찾고_실시간_수집_목록에_들어간다(fifth_ticker):
    assert tools.ticker_code("현대차") == NEW and tools.ticker_code(NEW) == NEW
    assert tools.ticker_algorithm(NEW) == "S0"  # 분야 채택이 없으면 단독
    assert NEW in mt.live_tickers() and config.MAIN_TICKER not in mt.live_tickers()
    assert tools.ticker_code("") == config.MAIN_TICKER


def test_종목별_리포트_경로():
    assert ex.ticker_path("evaluation/reports/experiments_monitor.md", config.MAIN_TICKER).name == "experiments_monitor.md"
    assert ex.ticker_path("evaluation/reports/experiments_monitor.md", NEW).name == f"experiments_monitor_{NEW}.md"
    assert ex.ticker_path("data/experiments/monitor_history.csv", NEW).name == f"monitor_history_{NEW}.csv"


def test_다른_종목_실험_데이터는_그_종목_원본으로_만들고_현금_규칙만_계산한다(fifth_ticker, monkeypatch):
    days = pd.bdate_range("2024-01-01", "2025-10-31")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
    vol = np.where(np.arange(len(days)) == 50, 0, 1000.0)  # 거래정지 하루
    raw = {"ohlcv": pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": vol}, index=days),
           "flows": pd.DataFrame({"foreign_net": 0.0}, index=days)}
    monkeypatch.setattr(mt, "ticker_raw", lambda code, shared=None: raw)
    from src.pipeline import features
    monkeypatch.setattr(features, "build_features",
                        lambda raw: pd.DataFrame({c: rng.normal(size=len(raw["ohlcv"])) for c in config.FEATURE_SETS["base"]}, index=raw["ohlcv"].index))
    data = ex.Data(code=NEW)
    assert data.name() == "현대차" and data.scores is None and days[50] not in data.samsung.index  # 거래정지일은 뺀다
    table, _ = ex.compare([{**config.EXPERIMENT_BASE, "start": "2025-04-23"}], data)
    assert table.iloc[0]["days"] > 100 and table.iloc[0]["down_action"] == "cash"
    with pytest.raises(ValueError, match="인버스 ETF 가 없어"):  # 인버스 ETF 가 없는 종목
        ex.run_one({**config.EXPERIMENT_BASE, "down_action": "inverse", "start": "2025-04-23", "end": None}, data)
    with pytest.raises(ValueError, match="뉴스 점수가 없어"):
        ex.run_one({**config.EXPERIMENT_BASE, "news_mode": "adjust", "start": "2025-04-23", "end": None}, data)


def test_종목_추가와_과거_신호는_레지스트리_밖이나_분야_알고리즘_종목을_거부한다(monkeypatch):
    with pytest.raises(ValueError, match="레지스트리"):
        mt.add(NEW)
    with pytest.raises(ValueError, match="레지스트리"):
        mt.backfill(NEW)
    with pytest.raises(ValueError, match="sector_models"):
        mt.backfill("277810")  # 로봇 분야 S4 는 분야 전체로 만든다
    with pytest.raises(ValueError, match="운영 종목"):
        mt.add(config.MAIN_TICKER)


def test_종목_추가는_어제까지만_수집하고_과거_신호와_분석을_이어서_만든다(fifth_ticker, monkeypatch):
    calls = []
    monkeypatch.setattr(mt, "collect", lambda end=None, tickers=None: calls.append(("collect", end, tickers)) or pd.DataFrame())
    monkeypatch.setattr(mt, "backfill", lambda code: calls.append(("backfill", code)) or pd.DataFrame({"signal": ["상승"]}, index=[pd.Timestamp("2026-10-21")]))
    monkeypatch.setattr(mt, "analysis", lambda: calls.append(("analysis",)))
    out = mt.add(NEW)
    yesterday = str((pd.Timestamp.now(tz=config.KST).tz_localize(None).normalize() - pd.Timedelta(days=1)).date())
    assert calls == [("collect", yesterday, [NEW]), ("backfill", NEW), ("analysis",)] and out["signals"] == 1
