import numpy as np
import pandas as pd
import pytest

from src import config
from src.pipeline import news_signal as ns
from src.pipeline.labels import make_labels

DAYS = pd.bdate_range("2025-11-03", periods=200)


def articles(seed=0):
    """신호일마다 기사 4건의 가짜 피처."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in DAYS:
        for k in range(4):
            rows.append({"signal_date": d, "title": f"{d.date()} 기사 {k}", "url": "u", "published_kst": "", "source": "s",
                         "category": ["samsung_direct", "semi_ai", "macro", "irrelevant"][k], "event_type": "other",
                         "direction": int(rng.integers(-1, 2)), "magnitude": int(rng.integers(1, 4)), "relevance": float(rng.uniform(0.3, 1)),
                         "novelty": ["new", "follow_up", "rehash", "new"][k], "downside_risk": int(rng.integers(0, 4)),
                         "one_line_summary": "A사 요약", "reasoning": "근거", "cluster_size": int(rng.integers(1, 5))})
    return pd.DataFrame(rows)


def test_일별_피처_손계산():
    a = pd.DataFrame([
        {"signal_date": DAYS[0], "category": "samsung_direct", "direction": 1, "magnitude": 3, "relevance": 0.5, "novelty": "new", "downside_risk": 3},
        {"signal_date": DAYS[0], "category": "samsung_direct", "direction": -1, "magnitude": 1, "relevance": 1.0, "novelty": "rehash", "downside_risk": 1},
        {"signal_date": DAYS[0], "category": "irrelevant", "direction": 0, "magnitude": 1, "relevance": 0.1, "novelty": "new", "downside_risk": 0},
    ])
    d = ns.daily_features(a, volume=pd.Series([5.0], index=DAYS[:1])).iloc[0]
    assert d["f_samsung_direct"] == pytest.approx(1.5 - 1.0) and d["f_samsung_direct_new"] == pytest.approx(1.5)
    assert d["f_risk_max"] == 3 and d["f_risk_n2"] == 1 and d["f_risk_sum"] == pytest.approx(1.5 + 1.0)
    assert d["f_mag3"] == 1 and d["f_relevant"] == 2


def test_일별_피처는_다음_날_기사가_바뀌어도_같다():
    a = articles()
    vol = pd.Series(np.arange(len(DAYS), dtype=float), index=DAYS)
    before = ns.daily_features(a, vol)
    changed = a.copy()
    changed.loc[changed["signal_date"] > DAYS[100], "direction"] *= -1
    vol2 = vol.copy()
    vol2[vol2.index > DAYS[100]] *= 10
    after = ns.daily_features(changed, vol2)
    pd.testing.assert_frame_equal(before.loc[:DAYS[100]], after.loc[:DAYS[100]])


def test_로지스틱_워크포워드는_정답이_끝난_행만_학습하고_미래가_바뀌어도_같다():
    daily = ns.daily_features(articles(), pd.Series(10.0, index=DAYS))
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(DAYS)))), index=DAYS)
    out = ns.walk_forward_logit(daily[ns.NEWS_FEATURES], make_labels(close))
    assert len(out) and (out["train_last_future_date"] < out["retrained_on"]).all()
    assert (out["train_rows"] >= config.NEWS_MODEL_MIN_ROWS).all()
    cut = DAYS[150]
    close2 = close.copy()
    close2[close2.index > cut] *= 1.5
    daily2 = daily.copy()
    daily2.loc[daily2.index > cut] = 99.0
    out2 = ns.walk_forward_logit(daily2[ns.NEWS_FEATURES], make_labels(close2))
    pd.testing.assert_frame_equal(out.loc[:cut], out2.loc[:cut])


def _alert_articles(rows):
    base = {"title": "t", "url": "u", "published_kst": "", "source": "s", "event_type": "export_control", "direction": -1,
            "magnitude": 3, "category": "samsung_direct", "one_line_summary": "A사 규제", "reasoning": "r", "cluster_size": 2}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_경보는_새_고위험_기사면_HIGH_같은_사건은_쿨다운():
    rows = [{"signal_date": DAYS[i], "relevance": 0.9, "downside_risk": 3, "novelty": "new", "title": "미국 수출 규제 확대"} for i in (0, 1, 3)]
    rows.append({"signal_date": DAYS[2], "relevance": 0.2, "downside_risk": 0, "novelty": "new", "title": "평범"})  # 쿨다운은 신호일 순번으로 센다
    a = _alert_articles(rows)
    daily = ns.daily_features(a, pd.Series(1.0, index=DAYS[:4]))
    out = ns.risk_alerts(a, daily)
    assert list(out.index) == [DAYS[0], DAYS[3]]  # 다음 신호일은 같은 사건이라 쿨다운, 쿨다운이 지나면 다시 알림
    assert (out["level"] == "HIGH").all() and out.iloc[0]["evidence"][0]["title"] == "미국 수출 규제 확대"
    text = ns.render_alert(DAYS[0], out.iloc[0].to_dict(), "검증 미통과(참고)")
    assert "투자 권유가 아닙니다" in text and "삼성전자 규제" in text and "A사" not in text


def test_경보는_등급이_오르면_쿨다운_안이어도_다시_알리고_조건이_없으면_없다():
    rows = [{"signal_date": DAYS[0], "relevance": 0.9, "downside_risk": 3, "novelty": "follow_up", "title": "규제"},
            {"signal_date": DAYS[1], "relevance": 0.9, "downside_risk": 3, "novelty": "new", "title": "규제"},
            {"signal_date": DAYS[2], "relevance": 0.2, "downside_risk": 0, "novelty": "new", "title": "평범"}]
    a = _alert_articles(rows)
    daily = ns.daily_features(a, pd.Series(1.0, index=DAYS[:3]))
    consensus = pd.Series([config.ALERT_CONSENSUS_RISK, 0, 0], index=DAYS[:3])
    out = ns.risk_alerts(a, daily, consensus)
    assert list(out["level"]) == ["MEDIUM", "HIGH"] and DAYS[2] not in out.index


def test_경보는_뒷날_기사가_바뀌어도_앞날_결과가_같다():
    a = articles(2)
    vol = pd.Series(10.0, index=DAYS)
    before = ns.risk_alerts(a, ns.daily_features(a, vol))
    changed = a.copy()
    changed.loc[changed["signal_date"] > DAYS[120], ["downside_risk", "relevance"]] = [3, 1.0]
    after = ns.risk_alerts(changed, ns.daily_features(changed, vol))
    cut = DAYS[120]
    pd.testing.assert_frame_equal(before.loc[:cut], after.loc[:cut])


def test_V1_실시간_값은_과거_검증_값과_같다(monkeypatch):
    a = articles(3)
    monkeypatch.setattr(ns, "news_volume", lambda days: pd.Series(10.0, index=days))
    rng = np.random.default_rng(4)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(DAYS)))), index=DAYS)
    full = ns.walk_forward_logit(ns.daily_features(a)[ns.NEWS_FEATURES], make_labels(close))
    t = DAYS[170]
    live = ns.v1_signal(t, close, a)
    assert live["signal"] == full.loc[t, "signal"] and live["p_상승"] == pytest.approx(full.loc[t, "p_상승"])


def test_쿨다운은_새_사건이_섞이면_다시_알린다():
    rows = [{"signal_date": DAYS[0], "relevance": 0.9, "downside_risk": 3, "novelty": "new", "title": "규제 A"},
            {"signal_date": DAYS[1], "relevance": 0.9, "downside_risk": 3, "novelty": "new", "title": "규제 A"},
            {"signal_date": DAYS[1], "relevance": 0.9, "downside_risk": 3, "novelty": "new", "title": "다른 사건 B"}]
    a = _alert_articles(rows)
    out = ns.risk_alerts(a, ns.daily_features(a, pd.Series(1.0, index=DAYS[:2])))
    assert list(out.index) == [DAYS[0], DAYS[1]]  # 둘째 날은 새 사건 B 가 있어 다시 알린다
