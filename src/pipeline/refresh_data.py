"""저장 가격·지표 데이터 주간 갱신.

이미 저장한 행은 바꾸지 않고, 종가가 확정된 새 날짜 행만 원본 끝에 덧붙인다(덧붙이기만 하는 갱신).
겹치는 날짜에서 제공처 값이 저장값과 다르면 저장값을 그대로 두고 건수만 기록한다(과거 학습 데이터가 바뀌지 않게).
덮어쓰기 전에 원본 파일을 data/backup/raw_<시각>/ 에 복사한다.
그 뒤 특징 표·정답·숫자 모델 신호·모의 매매 결과를 다시 만들고, 이미 있던 날짜의 신호가 바뀌지 않았는지 확인한다.
실행: .venv\\Scripts\\python -m src.pipeline.refresh_data
"""
import json
import shutil
from pathlib import Path

import pandas as pd

from src import config
from src.pipeline import backtest, collect_dart, collect_macro, collect_prices, features, labels, train

META_COLS = {"source", "collected_at", "is_dummy"}


def series_list() -> list[tuple[str, str]]:
    """갱신하는 원본 목록(학습·모의 매매에 쓰는 것만). 옛 FRED 원본과 고르지 않은 인버스 후보는 그대로 둔다."""
    prices = ["samsung_ohlcv", "samsung_investor_flows", f"inverse_{config.INVERSE_ETF_CODE}", *config.KR_EXTRA]
    macro = [*config.YF_TICKERS, *config.YF_TICKERS_EXT]
    return [("prices", n) for n in prices] + [("macro", n) for n in macro]


def fetch_series(kind: str, name: str, start: str, end: str) -> pd.DataFrame:
    """원본 하나를 제공처에서 기간만큼 받는다. 주간 갱신과 도구의 실시간 조회가 함께 쓴다."""
    if kind == "prices" and name == "samsung_ohlcv":
        return collect_prices.fetch_ohlcv(config.SAMSUNG_CODE, start, end)
    if kind == "prices" and name == "samsung_investor_flows":
        return collect_prices.fetch_foreign_net(config.SAMSUNG_CODE, start, end)
    if kind == "prices" and name == f"inverse_{config.INVERSE_ETF_CODE}":
        return collect_prices.fetch_ohlcv(config.INVERSE_ETF_CODE, start, end)
    if kind == "prices" and name in config.KR_EXTRA:
        return collect_prices.fetch_close(config.KR_EXTRA[name], start, end)
    if kind == "macro":
        return collect_macro.fetch_yf({**config.YF_TICKERS, **config.YF_TICKERS_EXT}[name], start, end)
    raise ValueError(f"받을 수 없는 데이터: {kind}/{name}")


def append_new(stored: pd.DataFrame, fresh: pd.DataFrame, end: pd.Timestamp) -> tuple[pd.DataFrame, int, int]:
    """저장본 뒤에 새 날짜 행만 붙인다. (합친 표, 추가한 행 수, 겹친 날짜 중 값이 다른 행 수)

    end 보다 뒤 날짜(장중 미확정 값)는 버린다. 겹치는 날짜는 저장값을 그대로 둔다.
    """
    fresh = fresh[fresh.index <= end]
    overlap = fresh.index.intersection(stored.index)
    cols = [c for c in stored.columns if c not in META_COLS and c in fresh.columns]
    mismatch = 0
    if len(overlap) and cols:
        a = stored.loc[overlap, cols].astype(float)
        b = fresh.loc[overlap, cols].astype(float)
        mismatch = int((~((a - b).abs() <= 1e-6 * b.abs().clip(lower=1)) & ~(a.isna() & b.isna())).any(axis=1).sum())
    new = fresh[fresh.index > stored.index.max()].reindex(columns=stored.columns)
    return pd.concat([stored, new]), len(new), mismatch


def last_closed_date(kind: str, name: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
    """그 시장에서 종가가 확정된 가장 최근 현지 날짜."""
    tz, close = config.MARKET_CLOSE.get(kind if kind == "prices" else name, config.US_MARKET_CLOSE)
    local = (now if now is not None else pd.Timestamp.now(tz="UTC")).tz_convert(tz).tz_localize(None)
    day = local.normalize()
    if kind == "prices":  # 수능일 등 늦게 마감하는 날
        close = config.KRX_LATE_CLOSE.get(str(day.date()), close)
    hour, minute = map(int, close.split(":"))
    return day if local - day >= pd.Timedelta(hours=hour, minutes=minute) else day - pd.Timedelta(days=1)


def refresh_cap(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """갱신으로 받을 수 있는 마지막 날짜 = 실행일(KST) 전날. 장 마감 직후의 잠정값(수급 등)을 영구 저장하지 않기 위해서다."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return now.tz_convert(config.KST).tz_localize(None).normalize() - pd.Timedelta(days=1)


def refresh_raw(now: pd.Timestamp | None = None, stamp: str | None = None) -> pd.DataFrame:
    """원본마다 마지막 저장일 뒤의 확정 행을 받아 덧붙인다.

    모두 받은 뒤에만 쓴다. 하나라도 실패하거나 국내 원본끼리 끝날이 다르면 아무 파일도 바꾸지 않는다.
    쓰기는 원본을 백업하고 임시 파일을 모두 만든 뒤 한꺼번에 바꾸며, 도중에 실패하면 백업으로 되돌린다. 결과 표의 written 열이 쓴 여부다.
    """
    stamp = stamp or pd.Timestamp.now(tz=config.KST).strftime("%Y%m%d_%H%M")
    backup_dir = Path(config.RAW_BACKUP_DIR) / f"raw_{stamp}"
    cap = refresh_cap(now)
    rows, pending = [], []
    for kind, name in series_list():
        path = Path(config.RAW_DIR) / kind / f"{name}.parquet"
        row = {"kind": kind, "name": name, "before_end": None, "after_end": None, "added": 0, "mismatch": 0, "note": "", "written": False}
        try:
            stored = pd.read_parquet(path)
            row["before_end"] = row["after_end"] = str(stored.index.max().date())
            end = min(last_closed_date(kind, name, now), cap)
            if stored.index.max() >= end:
                row["note"] = "새 확정 행 없음"
            else:
                start = stored.index.max() - pd.Timedelta(days=config.REFRESH_OVERLAP_DAYS)  # 겹치는 날로 제공처 값 변화를 점검한다
                fresh = fetch_series(kind, name, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                combined, added, mismatch = append_new(stored, fresh, end)
                row.update(added=added, mismatch=mismatch, after_end=str(combined.index.max().date()),
                           note=f"{added}일 추가" + (f", 겹친 날 값 차이 {mismatch}일(저장값 유지)" if mismatch else ""))
                if added:
                    pending.append((path, combined, row))
        except Exception as e:
            row["note"] = f"실패: {type(e).__name__}: {e}"
        rows.append(row)
    domestic_ends = {r["after_end"] for r in rows if r["kind"] == "prices" and not r["note"].startswith("실패")}
    if len(domestic_ends) > 1:  # 제공처가 예외 없이 빈 표·일부만 줘도 국내 원본끼리 끝날이 어긋나면 막는다
        for r in rows:
            if r["kind"] == "prices":
                r["note"] = f"실패: 국내 원본 끝날 불일치 {sorted(domestic_ends)} ({r['note']})"
    if any(r["note"].startswith("실패") for r in rows):
        for _, _, row in pending:
            row.update(after_end=row["before_end"], note=row["note"] + " (다른 원본 실패로 쓰지 않음)")
        return pd.DataFrame(rows)
    written = []
    try:
        for path, combined, _ in pending:  # 1) 백업과 임시 파일을 모두 만든다
            backup = backup_dir / path.parent.name / path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            combined.to_parquet(path.with_suffix(".tmp"))
        for path, _, row in pending:  # 2) 한꺼번에 바꾼다
            path.with_suffix(".tmp").replace(path)
            written.append(path)
            row["written"] = True
    except Exception:
        for path in written:  # 도중에 실패하면 바꾼 파일을 백업으로 되돌린다
            shutil.copy2(backup_dir / path.parent.name / path.name, path)
        for path, _, row in pending:
            path.with_suffix(".tmp").unlink(missing_ok=True)
            row["written"] = False
        raise
    return pd.DataFrame(rows)


def signal_changes(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """이미 있던 날짜 중 신호나 클래스 확률이 바뀐 날 수. 덧붙이기만 했으면 0 이어야 한다."""
    common = before.index.intersection(after.index)
    prob_cols = [c for c in before.columns if c.startswith("p_") and c in after.columns]
    changed = before.loc[common, "signal"] != after.loc[common, "signal"]
    if prob_cols:
        changed |= (before.loc[common, prob_cols] - after.loc[common, prob_cols]).abs().max(axis=1) > 1e-9
    return int(changed.sum()) + len(before.index.difference(after.index))  # 사라진 날짜도 변화로 센다


def feature_changes(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """이미 있던 날짜 중 특징 값이 바뀐 날 수(해외 원본이 늦게 채워지면 생길 수 있다). 경고용이다."""
    common = before.index.intersection(after.index)
    cols = [c for c in before.columns if c in after.columns and c != "is_dummy"]
    a, b = before.loc[common, cols].astype(float), after.loc[common, cols].astype(float)
    diff = ~(((a - b).abs() <= 1e-9) | (a.isna() & b.isna()))
    return int(diff.any(axis=1).sum())


def processed_outputs() -> list[Path]:
    """다시 만드는 산출물. 실패하면 이 파일들을 되돌린다."""
    names = ["features.parquet", "labels.parquet", "dart_index.parquet",
             *[f"equity_{m}_{r}.parquet" for m in ("numeric", "candidate") for r in config.DOWN_ACTIONS]]
    return [Path(config.PROCESSED_DIR) / n for n in names] + [Path(config.SIGNALS_NUMERIC_PATH), Path(config.SIGNALS_CANDIDATE_PATH)]


def rebuild() -> dict:
    """특징 표·정답·숫자 모델 신호(기존·후보)·모의 매매 결과를 다시 만든다.

    이미 있던 날짜의 신호·확률이 바뀌면 모의 매매 전에 멈춘다(예외). 되돌리기는 run() 이 한다.
    """
    before = {p: pd.read_parquet(p) for p in (config.SIGNALS_NUMERIC_PATH, config.SIGNALS_CANDIDATE_PATH) if Path(p).exists()}
    feat_path = Path(config.PROCESSED_DIR) / "features.parquet"
    old_features = pd.read_parquet(feat_path) if feat_path.exists() else None
    features.main()
    labels.main()
    train.main()
    changes = {Path(p).stem: signal_changes(df, pd.read_parquet(p)) for p, df in before.items()}
    if any(changes.values()):
        raise RuntimeError(f"이미 있던 날짜의 신호·확률이 바뀌었다 {changes}. 갱신을 되돌린다")
    backtest.main()
    return {"signal_changes": changes,
            "feature_changes": feature_changes(old_features, pd.read_parquet(feat_path)) if old_features is not None else 0}


def write_status(data_end: str, table: pd.DataFrame, checks: dict, dart_note: str, rag_stale: bool) -> dict:
    """갱신 결과를 data_status.json 에 남긴다. config.DATA_END 는 이 파일의 data_end 를 따른다."""
    status = {"data_end": data_end, "refreshed_at": pd.Timestamp.now(tz=config.KST).strftime("%Y-%m-%d %H:%M"),
              "series": table.to_dict("records"), "signal_changes": checks.get("signal_changes", {}),
              "feature_changes": checks.get("feature_changes", 0), "dart": dart_note, "rag_index_stale": rag_stale}
    path = config.data_status_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    return status


def restore(raw_backup: Path, processed_backup: Path):
    """백업해 둔 원본과 산출물로 되돌린다."""
    for src in raw_backup.rglob("*.parquet"):
        shutil.copy2(src, Path(config.RAW_DIR) / src.parent.name / src.name)
    backed_up = {src.name for src in processed_backup.glob("*.parquet")}
    for target in processed_outputs():
        if target.name in backed_up:
            shutil.copy2(processed_backup / target.name, target)
        else:
            target.unlink(missing_ok=True)  # 갱신 전에 없던 산출물은 지운다


def run(now: pd.Timestamp | None = None) -> dict:
    """주간 갱신 한 번. 삼성전자 일봉이 늘었을 때만 다시 만든다. 결과 요약(status)을 돌려준다.

    다시 만들다 실패하거나 이미 있던 날의 신호가 바뀌면 원본·산출물·DATA_END 를 모두 갱신 전으로 되돌리고 예외를 던진다.
    """
    stamp = pd.Timestamp.now(tz=config.KST).strftime("%Y%m%d_%H%M")
    table = refresh_raw(now, stamp)
    failed = table[table["note"].str.startswith("실패")]
    if len(failed):
        raise RuntimeError("원본 갱신 실패로 아무 파일도 바꾸지 않았다: " + "; ".join(f"{r['name']} {r['note']}" for _, r in failed.iterrows()))
    data_end = table.loc[table["name"] == "samsung_ohlcv", "after_end"].iloc[0]
    checks, dart_note, rag_stale = {}, "변화 없음", False
    if data_end != config.DATA_END:
        previous_end = config.DATA_END
        processed_backup = Path(config.RAW_BACKUP_DIR) / f"processed_{stamp}"
        processed_backup.mkdir(parents=True, exist_ok=True)
        for path in processed_outputs():
            if path.exists():
                shutil.copy2(path, processed_backup / path.name)
        dart_path = Path(config.PROCESSED_DIR) / "dart_index.parquet"
        dart_before = len(pd.read_parquet(dart_path)) if dart_path.exists() else 0
        config.DATA_END = data_end  # 같은 프로세스에서 뒤이어 도는 단계(정답·모니터링)가 새 끝날을 쓰게 한다
        try:
            try:
                collect_dart.main()  # 새 정기공시 목록과 원문(이미 받은 원문은 다시 받지 않는다)
                dart_after = len(pd.read_parquet(dart_path))
                rag_stale = dart_after > dart_before
                dart_note = f"공시 {dart_after - dart_before}건 추가" + (" — RAG 색인 재생성 필요" if rag_stale else "")
            except Exception as e:
                dart_note = f"실패: {type(e).__name__}"
            checks = rebuild()
        except Exception as e:
            config.DATA_END = previous_end
            restore(Path(config.RAW_BACKUP_DIR) / f"raw_{stamp}", processed_backup)
            raise RuntimeError(f"다시 만들기 실패로 원본·산출물을 되돌렸다({e}). 새로 받은 DART 원문 zip 은 남는다(목록은 되돌림)") from e
    return write_status(data_end, table, checks, dart_note, rag_stale)


def summary_lines(status: dict) -> list[str]:
    """알림 메일·화면용 짧은 요약."""
    added = [s for s in status["series"] if s["added"]]
    mismatch = [s for s in status["series"] if s["mismatch"]]
    lines = [f"저장 데이터 끝 {status['data_end']} (갱신 {status['refreshed_at']}) · 늘어난 원본 {len(added)}/{len(status['series'])}개"]
    if status["signal_changes"]:
        lines.append("이미 있던 날의 신호·확률 변화: 없음(과거 결과 그대로)")
    if status.get("feature_changes"):
        lines.append(f"경고: 이미 있던 날의 특징 값이 {status['feature_changes']}일 바뀌었다(신호는 그대로). 해외 원본이 늦게 채워진 경우다")
    if mismatch:
        lines.append("제공처 값이 저장값과 다른 날(저장값 유지): " + ", ".join(f"{s['name']} {s['mismatch']}일" for s in mismatch))
    if status.get("rag_index_stale"):
        lines.append(f"DART {status['dart']}")
    return lines


def main():
    """명령줄 실행."""
    status = run()
    print("\n".join(summary_lines(status)))
    print(pd.DataFrame(status["series"]).to_string(index=False))


if __name__ == "__main__":
    main()
