from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# 특징으로 쓰면 안 되는 열 이름(정답·당일 종가). 이름에 들어 있기만 해도 막는다. 학습 직전 X 에도 다시 적용한다
FORBIDDEN_PATTERNS = ("label", "future_")
FORBIDDEN_EXACT = {"close", "is_dummy"}


def moving_average_features(close: pd.Series) -> pd.DataFrame:
    """종가와 이동평균의 괴리율(종가/이동평균 - 1)을 돌려준다."""
    return pd.DataFrame({f"ma{w}_gap": close / close.rolling(w).mean() - 1 for w in config.MA_WINDOWS})


def bollinger_features(close: pd.Series) -> pd.DataFrame:
    """볼린저 밴드 안에서 종가의 위치(%b)와 밴드 폭을 돌려준다."""
    mid = close.rolling(config.BOLLINGER_WINDOW).mean()
    std = close.rolling(config.BOLLINGER_WINDOW).std()
    upper, lower = mid + config.BOLLINGER_STD * std, mid - config.BOLLINGER_STD * std
    return pd.DataFrame({"bb_pctb": (close - lower) / (upper - lower), "bb_width": (upper - lower) / mid})


def ichimoku_features(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """일목균형표 특징. 구름은 차트에 그날 보이는 구름(26일 전에 계산된 선행스팬)을 쓴다.

    선행스팬을 shift(+기준선 기간)으로 옮기는 것은 과거 값을 가져오는 것이라 안전하다.
    후행스팬(종가를 26일 뒤로 옮긴 선)은 미래 종가와 비교하는 값이라 쓰지 않는다.
    """
    p = config.ICHIMOKU
    mid = lambda n: (high.rolling(n).max() + low.rolling(n).min()) / 2
    tenkan, kijun = mid(p["tenkan"]), mid(p["kijun"])
    span_a = ((tenkan + kijun) / 2).shift(p["kijun"])
    span_b = mid(p["senkou_b"]).shift(p["kijun"])
    cloud_top, cloud_bottom = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    return pd.DataFrame({
        "ichi_tenkan_gap": close / tenkan - 1,
        "ichi_kijun_gap": close / kijun - 1,
        "ichi_tenkan_kijun": tenkan / kijun - 1,
        "ichi_cloud_top_gap": close / cloud_top - 1,
        "ichi_cloud_bottom_gap": close / cloud_bottom - 1,
        "ichi_cloud_thickness": (span_a - span_b) / close,
    })


def parabolic_sar(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    """파라볼릭 SAR 값과 추세 방향(+1 상승, -1 하락)을 그날까지의 고가·저가만으로 계산한다."""
    step, max_af = config.PSAR_STEP, config.PSAR_MAX
    h, l = high.to_numpy(float), low.to_numpy(float)
    sar, trend = np.full(len(h), np.nan), np.zeros(len(h))
    if len(h) < 2:
        return pd.Series(sar, high.index), pd.Series(trend, high.index)
    up, af = True, step
    ep, sar[0], trend[0] = h[0], l[0], 1
    for i in range(1, len(h)):
        prev = sar[i - 1]
        if up:
            cur = min(prev + af * (ep - prev), l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < cur:
                up, cur, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_af)
        else:
            cur = max(prev + af * (ep - prev), h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > cur:
                up, cur, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_af)
        sar[i], trend[i] = cur, 1 if up else -1
    return pd.Series(sar, high.index), pd.Series(trend, high.index)


def domestic_features(ohlcv: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """국내 일봉·순매수로 그날 종가 기준 특징을 만든다. 신호에 쓰기 전에 반드시 하루 늦춘다(build_features)."""
    if ohlcv.index.has_duplicates:
        raise ValueError("일봉에 중복 날짜가 있다")
    ohlcv = ohlcv.sort_index()
    close, high, low, volume = ohlcv["Close"], ohlcv["High"], ohlcv["Low"], ohlcv["Volume"]
    sar, sar_trend = parabolic_sar(high, low)
    out = pd.concat([
        pd.DataFrame({
            **{f"ret_{w}d": close.pct_change(w) for w in config.RETURN_WINDOWS},
            "volume_ratio": volume / volume.rolling(config.VOLUME_MA_WINDOW).mean(),
            "psar_gap": close / sar - 1,
            "psar_trend": sar_trend,
        }),
        moving_average_features(close),
        bollinger_features(close),
        ichimoku_features(high, low, close),
    ], axis=1)
    flow = flows.reindex(ohlcv.index)
    out["foreign_net_ratio"] = flow["foreign_net"] / volume
    out["institution_net_ratio"] = flow["institution_net"] / volume
    w = config.FLOW_WINDOW
    out[f"foreign_net_ratio_{w}d"] = flow["foreign_net"].rolling(w).sum() / volume.rolling(w).sum()
    out[f"foreign_hold_change_{w}d"] = flow["foreign_hold_ratio"].diff(w)
    return out


def foreign_series_features(name: str, series: pd.DataFrame) -> pd.DataFrame:
    """해외·거시 시계열 하나의 변화율 특징을 현지 날짜 기준으로 만든다."""
    value = series["value"].dropna().sort_index()  # 결측 행이 '가장 최근 값'으로 붙지 않게 먼저 뺀다
    if name in config.LEVEL_SERIES:  # 금리·변동성은 수준과 차이를 쓴다
        df = pd.DataFrame({f"{name}_level": value,
                           **{f"{name}_chg_{w}d": value.diff(w) for w in config.FOREIGN_CHANGE_WINDOWS}})
    else:
        df = pd.DataFrame({f"{name}_ret_{w}d": value.pct_change(w) for w in config.FOREIGN_CHANGE_WINDOWS})
    df[f"{name}_src_date"] = df.index
    return df


def attach_foreign(kr_days: pd.DatetimeIndex, foreign: pd.DataFrame, name: str) -> pd.DataFrame:
    """한국 거래일 t 마다 현지 날짜가 t 보다 앞선(같은 날 제외) 가장 최근 값을 붙이고, 며칠 묵은 값인지 함께 둔다."""
    left = pd.DataFrame({"t": kr_days})
    right = foreign.reset_index(drop=True).sort_values(f"{name}_src_date")
    merged = pd.merge_asof(left, right, left_on="t", right_on=f"{name}_src_date",
                           direction="backward", allow_exact_matches=False)
    merged[f"{name}_age_days"] = (merged["t"] - merged[f"{name}_src_date"]).dt.days
    return merged.set_index("t")


def load_raw(kind: str, name: str) -> pd.DataFrame:
    """data/raw/<kind>/<name>.parquet 를 읽는다."""
    return pd.read_parquet(Path(config.RAW_DIR) / kind / f"{name}.parquet")


def build_features(as_of=None, raw: dict | None = None) -> pd.DataFrame:
    """한국 거래일 t 마다 t일 15:15 에 알 수 있던 값만으로 특징 행을 만든다.

    as_of 를 주면 as_of 이전 데이터만 읽고 as_of 까지의 행만 돌려준다.
    raw 를 주면 파일 대신 그 데이터를 쓴다(테스트용). 키: ohlcv, flows, 그리고 해외 시계열 이름.
    """
    raw = raw or {
        "ohlcv": load_raw("prices", "samsung_ohlcv"),
        "flows": load_raw("prices", "samsung_investor_flows"),
        **{name: load_raw("macro", name) for name in [*config.FRED_SERIES, *config.YF_TICKERS]},
    }
    days = pd.DatetimeIndex(raw["ohlcv"].index).sort_values()
    if as_of is None:
        cut = days.max() + pd.Timedelta(days=1)
    else:
        cut = pd.Timestamp(as_of)
        if cut.tzinfo is not None:
            cut = cut.tz_convert(config.KST).tz_localize(None)
        cut = cut.normalize()
    days = days[days <= cut]
    before = lambda df: df[df.index < cut]  # as_of 당일 이후 데이터는 읽지 않는다

    domestic = domestic_features(before(raw["ohlcv"]), before(raw["flows"]))
    domestic = domestic.reindex(days).shift(1)  # t 행에는 t-1 종가까지의 값만 둔다
    domestic["kr_src_date"] = pd.Series(days, index=days).shift(1)

    parts = [domestic]
    for name in [*config.FRED_SERIES, *config.YF_TICKERS]:
        feats = foreign_series_features(name, before(raw[name]))
        parts.append(attach_foreign(days, feats, name))
    table = pd.concat(parts, axis=1)
    table.index.name = "date"
    assert_no_lookahead(table)
    return table.drop(columns=[c for c in table.columns if c.endswith("_src_date")])


def assert_no_lookahead(table: pd.DataFrame):
    """모든 행에서 사용한 원천 데이터 날짜가 t 보다 앞서는지 확인한다. 어기면 예외를 던진다."""
    t = pd.Series(table.index, index=table.index)
    for col in [c for c in table.columns if c.endswith("_src_date")]:
        bad = table[col].notna() & (table[col] >= t)
        if bad.any():
            raise AssertionError(f"{col}: {int(bad.sum())}개 행이 t 이후 데이터를 쓴다 (예: {table.index[bad][0].date()})")
    leaked = [c for c in table.columns if c in FORBIDDEN_EXACT or any(p in c for p in FORBIDDEN_PATTERNS)]
    if leaked:
        raise AssertionError(f"정답 관련 열이 특징에 있다: {sorted(leaked)}")


def main():
    """특징 표를 만들어 data/processed/features.parquet 에 저장한다."""
    table = build_features()
    usable = table.dropna()
    table["is_dummy"] = False  # 저장용 표시. 학습 X 에서는 assert_no_lookahead 가 이 열을 막는다
    out = Path(config.PROCESSED_DIR) / "features.parquet"
    table.to_parquet(out)
    print(out, f"{len(table)}행 x {table.shape[1]}열, 결측 없는 행 {len(usable)} (첫 날짜 {usable.index.min().date()})")


if __name__ == "__main__":
    main()
