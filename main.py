#!/usr/bin/env python3
"""
異常検知システム
Isolation Forest + SHAP + PCA可視化 (Tableau連携)
版数: 2.0  作成日: 2026年3月

リファクタリング概要:
- Config dataclass による設定の一元管理
- モジュールレベルロガー（関数への引数渡し廃止）
- コンソール=INFO / ファイル=DEBUG の二層ログ
- timed デコレータによる処理時間の自動計測
- AnomalyDetectionError カスタム例外でエラーフロー統一
- ModelResults dataclass による結果の集約
- 各ステップで DataFrame 統計・カラム詳細を DEBUG 出力
"""

import argparse
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============================================================
# モジュールレベルロガー（setup_logging() で初期化）
# ============================================================
logger = logging.getLogger(__name__)


# ============================================================
# 設定
# ============================================================

@dataclass(frozen=True)
class Config:
    """実行パラメータの一元管理。frozen=True で不変オブジェクト。"""

    # --- Isolation Forest ---
    contamination: float = 0.05
    n_estimators: int = 200
    max_features: float = 1.0
    max_samples: float = 1.0            # 各木で使うサンプルの割合
    random_state: int = 42

    # --- 入力 ---
    input_encoding: str = "cp932"           # 入力CSVの文字コード（Shift-JIS=cp932, UTF-8=utf-8）

    # --- 前処理 ---
    missing_rate_threshold: float = 0.80    # 欠損率除外閾値
    date_parse_threshold: float = 0.80      # 日付判定の成功率閾値
    ohe_top_n: int = 10                     # OHE対象とする上位カテゴリ数
    ohe_coverage_threshold: float = 0.50    # 上位N件のカバレッジ閾値（超えたらOHE）
    variance_threshold: float = 0.01        # 低分散除外閾値
    corr_threshold: float = 0.95            # 高相関除外閾値

    # --- SHAP ---
    shap_all: bool = False                  # True: 全件 / False: 異常レコードのみ

    # --- PCA ---
    pca_variance_warning: float = 0.50      # 累積寄与率の警告閾値

    # --- Optunaチューニング ---
    label_col: Optional[str] = None         # ラベル列名（指定時にチューニング実行）
    label_anomaly_value: int = 1            # 異常を示すラベル値
    n_trials: int = 100                     # Optunaの試行回数

    # --- パス ---
    in_dir: Path = field(default_factory=lambda: Path("in"))
    out_dir: Path = field(default_factory=lambda: Path("out"))
    column_config_path: Optional[Path] = None   # カラム設定JSONのパス


# ============================================================
# カスタム例外
# ============================================================

class AnomalyDetectionError(Exception):
    """処理を中断すべき異常検知システム固有のエラー。"""


# ============================================================
# ユーティリティ
# ============================================================

def timed(step_name: str) -> Callable:
    """処理開始・終了・経過時間をログ出力するデコレータ。

    AnomalyDetectionError はそのまま再 raise し、
    それ以外の予期しない例外はフルトレースを DEBUG に出力してから再 raise する。
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"▶ {step_name} 開始")
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except AnomalyDetectionError:
                raise
            except Exception as e:
                logger.error(f"✗ {step_name} で予期しないエラー: {e}")
                logger.debug(traceback.format_exc())
                raise
            elapsed = time.perf_counter() - t0
            logger.info(f"◀ {step_name} 完了 ({elapsed:.2f}秒)")
            return result
        return wrapper
    return decorator


def _log_df_summary(df: pd.DataFrame, label: str) -> None:
    """DataFrameの統計サマリを DEBUG レベルで出力する。

    Args:
        df: 対象DataFrame
        label: ログ識別用ラベル
    """
    logger.debug(f"[{label}] shape={df.shape}")
    for col in df.columns:
        s = df[col]
        n_null = int(s.isnull().sum())
        null_pct = n_null / len(s) * 100 if len(s) > 0 else 0.0
        if pd.api.types.is_numeric_dtype(s):
            logger.debug(
                f"  {col}: dtype={s.dtype}, "
                f"null={n_null}({null_pct:.1f}%), "
                f"min={s.min():.4g}, max={s.max():.4g}, "
                f"mean={s.mean():.4g}, std={s.std():.4g}, "
                f"var={s.var():.4g}"
            )
        else:
            sample = s.dropna().iloc[:3].tolist()
            logger.debug(
                f"  {col}: dtype={s.dtype}, "
                f"null={n_null}({null_pct:.1f}%), "
                f"unique={s.nunique()}, sample={sample}"
            )


# ============================================================
# ロギング設定
# ============================================================

def setup_logging(out_dir: Path) -> None:
    """ロガーを設定する。

    - ファイル (processing.log): DEBUG 以上（詳細ログ、関数名・行番号付き）
    - コンソール (stdout):        INFO 以上（簡潔な進捗表示）

    Args:
        out_dir: ログファイルの出力先フォルダ
    """
    out_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ファイルハンドラ（DEBUG 以上 / 詳細フォーマット）
    fh = logging.FileHandler(out_dir / "processing.log", encoding="utf-8", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(funcName)s:%(lineno)d - %(message)s"
    ))

    # コンソールハンドラ（INFO 以上 / 簡潔フォーマット）
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))

    root.addHandler(fh)
    root.addHandler(ch)


# ============================================================
# データ読み込み
# ============================================================

@timed("データ読み込み")
def load_csvs(cfg: Config) -> pd.DataFrame:
    """in/ フォルダからCSVを全件読み込み、結合して返す。

    Args:
        cfg: 実行設定

    Returns:
        結合済みDataFrame

    Raises:
        AnomalyDetectionError: フォルダ不在 / CSV 0件 / 読み込み失敗
    """
    if not cfg.in_dir.exists():
        raise AnomalyDetectionError(
            f"入力フォルダ '{cfg.in_dir}' が存在しません。"
        )

    csv_files = sorted(cfg.in_dir.glob("*.csv"))
    if not csv_files:
        raise AnomalyDetectionError(
            f"'{cfg.in_dir}' フォルダにCSVファイルが存在しません。"
        )

    logger.debug(f"CSVファイル一覧: {[f.name for f in csv_files]}")

    dfs: list[pd.DataFrame] = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding=cfg.input_encoding)
        except Exception as e:
            raise AnomalyDetectionError(
                f"CSV読み込みエラー ({csv_file.name}): {e}"
            ) from e

        logger.info(
            f"  読み込み: {csv_file.name}  "
            f"{len(df)}件  {len(df.columns)}カラム"
        )
        logger.debug(f"  カラム: {df.columns.tolist()}")
        logger.debug(f"  dtypes:\n{df.dtypes.to_string()}")
        _log_df_summary(df, f"load:{csv_file.name}")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"結合後: {len(combined)}件, {len(combined.columns)}カラム")
    _log_df_summary(combined, "load_csvs 結合後")
    return combined


# ============================================================
# 前処理
# ============================================================

def _encode_column(
    df: pd.DataFrame,
    col: str,
    n_rows: int,
    cfg: Config,
    forced_encoding: Optional[str] = None,              # "ohe" | "frequency" | "auto" | None
    forced_ohe_top_n: Optional[int] = None,             # カラム個別の上位N件
    forced_ohe_coverage_threshold: Optional[float] = None,  # カラム個別のカバレッジ閾値
) -> tuple[pd.DataFrame, dict]:
    """単一の object 型カラムを型判定してエンコードする。

    encoding の優先順位:
      "ohe"       : カバレッジに関わらず強制OHE
      "frequency" : カバレッジに関わらず強制頻度エンコーディング
      "auto"      : top_n のカバレッジ >= threshold なら OHE、未満なら頻度
                    （日付判定はスキップ）
      None        : 日付 → OHE/頻度の自動判定（グローバル設定を使用）

    Args:
        df: 処理対象DataFrame
        col: 対象カラム名
        n_rows: 全レコード数
        cfg: 実行設定
        forced_encoding: エンコーディング方式（"ohe" / "frequency" / "auto" / None）
        forced_ohe_top_n: OHEの上位カテゴリ数（Noneの場合cfg.ohe_top_nを使用）
        forced_ohe_coverage_threshold: カバレッジ閾値（Noneの場合cfg.ohe_coverage_thresholdを使用）

    Returns:
        (エンコード処理後のDataFrame, エンコードサマリ dict)
    """
    summary: dict = {
        "column": col,
        "encoding": "",
        "n_unique": int(df[col].nunique()),
        "coverage_top_n": None,
        "other_count": None,
        "generated_columns": "",
    }

    # ---- 日付判定（forced_encoding 未指定時のみ / "auto" はスキップ）----
    if forced_encoding is None:
        parsed = pd.to_datetime(df[col], errors="coerce")
        parse_rate = parsed.notna().sum() / n_rows
        logger.debug(
            f"  日付パース率 [{col}]: {parse_rate:.3f} "
            f"(閾値={cfg.date_parse_threshold})"
        )

        if parse_rate >= cfg.date_parse_threshold:
            logger.info(f"  日付カラム → datetime分解: {col} (パース率={parse_rate:.3f})")
            df[col] = parsed
            date_parts = {
                "year": df[col].dt.year,
                "month": df[col].dt.month,
                "day": df[col].dt.day,
                "hour": df[col].dt.hour,
                "weekday": df[col].dt.weekday,
            }
            generated = []
            for part, values in date_parts.items():
                new_col = f"{col}_{part}"
                df[new_col] = values
                generated.append(new_col)
                logger.debug(f"    生成: {new_col}")
            df = df.drop(columns=[col])
            summary["encoding"] = "datetime"
            summary["generated_columns"] = ", ".join(generated)
            return df, summary

    # ---- 欠損補完（最頻値）----
    n_null = int(df[col].isnull().sum())
    mode_vals = df[col].mode()
    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
    if n_null > 0:
        df[col] = df[col].fillna(fill_val)
        logger.debug(f"  欠損補完 [{col}]: {n_null}件 → '{fill_val}'")

    # ---- OHE / 頻度エンコーディング の決定 ----
    top_n = forced_ohe_top_n if forced_ohe_top_n is not None else cfg.ohe_top_n
    threshold = (
        forced_ohe_coverage_threshold
        if forced_ohe_coverage_threshold is not None
        else cfg.ohe_coverage_threshold
    )
    freq = df[col].value_counts(normalize=True)
    top_n_cats = freq.nlargest(top_n).index.tolist()
    coverage = float(freq.nlargest(top_n).sum())

    logger.debug(
        f"  カバレッジ [{col}]: top{top_n}={coverage:.3f} "
        f"(閾値={threshold}), unique={df[col].nunique()}, "
        f"forced_encoding={forced_encoding}"
    )
    logger.debug(f"  top{top_n} カテゴリ: {top_n_cats}")

    summary["coverage_top_n"] = round(coverage, 4)

    use_ohe = (
        forced_encoding == "ohe"
        or (forced_encoding in (None, "auto") and coverage >= threshold)
    )

    if use_ohe:
        other_count = int((~df[col].isin(top_n_cats)).sum())
        df[col] = df[col].where(df[col].isin(top_n_cats), other="__other__")
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
        if forced_encoding == "ohe":
            mode_label = f"強制OHE top{top_n}"
        elif forced_encoding == "auto":
            mode_label = f"自動→OHE top{top_n} (coverage={coverage:.3f}>={threshold:.3f})"
        else:
            mode_label = f"自動→OHE top{top_n}"
        logger.info(
            f"  [{mode_label}] [{col}]: "
            f"(coverage={coverage:.3f}) "
            f"+ __other__({other_count}件) → {len(dummies.columns)}列生成"
        )
        logger.debug(f"  生成列: {dummies.columns.tolist()}")
        summary["encoding"] = "ohe"
        summary["other_count"] = other_count
        summary["generated_columns"] = ", ".join(dummies.columns.tolist())
    else:
        df[col] = df[col].map(freq)
        if forced_encoding == "frequency":
            mode_label = "強制頻度"
        elif forced_encoding == "auto":
            mode_label = f"自動→頻度 (coverage={coverage:.3f}<{threshold:.3f})"
        else:
            mode_label = "自動→頻度"
        logger.info(f"  [{mode_label}] {col} (coverage={coverage:.3f})")
        logger.debug(f"  頻度 top5:\n{freq.head().to_string()}")
        summary["encoding"] = "frequency"
        summary["generated_columns"] = col

    return df, summary


def _load_column_config(path: Path) -> dict:
    """カラム設定JSONを読み込む。

    フォーマット例:
    {
      "age":         {"use": true,  "type": "numeric"},
      "region":      {"use": true,  "type": "categorical", "encoding": "ohe", "ohe_top_n": 5},
      "customer_id": {"use": false, "type": "categorical", "encoding": "frequency"}
    }

    Args:
        path: JSONファイルのパス

    Returns:
        カラム名 → 設定dict のマッピング

    Raises:
        AnomalyDetectionError: ファイルが存在しない / JSON形式エラー
    """
    import json

    if not path.exists():
        raise AnomalyDetectionError(f"カラム設定ファイルが見つかりません: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise AnomalyDetectionError(
            f"カラム設定JSONの構文エラー: {e}\n"
            f"  ファイル: {path}"
        ) from e
    except OSError as e:
        raise AnomalyDetectionError(
            f"カラム設定JSONの読み込みエラー: {e}\n"
            f"  ファイル: {path}"
        ) from e

    logger.info(f"カラム設定JSON読み込み: {path} ({len(config)}カラム定義)")
    for col, spec in config.items():
        logger.debug(f"  {col}: {spec}")
    return config


@timed("前処理")
def preprocess(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """自動型判定・欠損補完・エンコーディングを行う前処理。

    Args:
        df: 入力DataFrame
        cfg: 実行設定

    Returns:
        (処理済みDataFrame, 除外カラムリスト, カテゴリエンコードサマリリスト)
    """
    excluded_cols: list[str] = []
    encode_summaries: list[dict] = []
    n_rows = len(df)
    logger.debug(f"前処理開始: {df.shape}")

    # ---- カラム設定JSON の適用 ----
    column_config: dict = {}
    if cfg.column_config_path is not None:
        column_config = _load_column_config(cfg.column_config_path)

        data_cols = set(df.columns)
        enabled_cols = [c for c, s in column_config.items() if s.get("use", True)]
        disabled_cols = [c for c, s in column_config.items() if not s.get("use", True)]

        # JSON定義カラムの一覧をINFOで表示
        logger.info("-" * 50)
        logger.info(f"【カラム設定JSON】定義数={len(column_config)}件")
        for col, spec in column_config.items():
            try:
                use = spec.get("use", True)
                ctype = spec.get("type", "-")
                enc = spec.get("encoding", "-")
                top_n = spec.get("ohe_top_n", "-")
                in_data = "✓" if col in data_cols else "✗(データになし)"
                flag = "ON " if use else "OFF"
                # :<N は s なしで int/str 両対応
                logger.info(
                    f"  [{flag}] {str(col):<30} type={str(ctype):<12} "
                    f"encoding={str(enc):<10} ohe_top_n={str(top_n):<5} data={in_data}"
                )
            except Exception as e:
                logger.warning(f"  カラム '{col}' のログ出力でエラー: {e} / spec={spec}")

        # データに存在するカラムとJSONのマッピングサマリ
        active_in_data  = [c for c in enabled_cols if c in data_cols]
        missing_in_data = [c for c in enabled_cols if c not in data_cols]
        unlisted        = [c for c in df.columns if c not in set(enabled_cols + disabled_cols)]

        logger.info("-" * 50)
        logger.info(f"【使用カラム（use=true かつデータに存在）】 {len(active_in_data)}件: {active_in_data}")
        if disabled_cols:
            disabled_in_data = [c for c in disabled_cols if c in data_cols]
            logger.info(f"【除外カラム（use=false）】 {len(disabled_in_data)}件: {disabled_in_data}")
        if missing_in_data:
            logger.error(f"【エラー】JSONでONに設定されているがデータに存在しないカラム: {missing_in_data}")
            raise AnomalyDetectionError(
                f"column_config に use=true で定義されているカラムがデータに存在しません: {missing_in_data}\n"
                f"  JSONファイル: {cfg.column_config_path}\n"
                f"  データのカラム一覧: {sorted(data_cols)}"
            )
        if unlisted:
            logger.info(f"【JSON未定義のため除外】 {len(unlisted)}件: {unlisted}")
        logger.info("-" * 50)

        # use=false のカラムを除外
        if disabled_cols:
            excluded_cols.extend([c for c in disabled_cols if c in data_cols])
            df = df.drop(columns=[c for c in disabled_cols if c in data_cols])

        # JSON に記載のないカラムを除外
        if unlisted:
            excluded_cols.extend(unlisted)
            df = df.drop(columns=unlisted)

        logger.info(f"column_config 適用後の使用カラム ({len(df.columns)}件): {df.columns.tolist()}")

    # ---- 欠損率チェック ----
    missing_ratio = df.isnull().mean()
    nonzero_missing = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
    if not nonzero_missing.empty:
        logger.debug(f"欠損率（非ゼロのみ）:\n{nonzero_missing.to_string()}")

    high_missing = missing_ratio[
        missing_ratio > cfg.missing_rate_threshold
    ].index.tolist()
    for col in high_missing:
        logger.warning(
            f"  欠損率 {missing_ratio[col]:.1%} 超のため除外: {col}"
        )
    if high_missing:
        excluded_cols.extend(high_missing)
        df = df.drop(columns=high_missing)

    # ---- カラムごとに型判定・処理 ----
    cols_to_process = list(df.columns)

    # dtype別に分類してINFOで表示
    dtype_groups: dict[str, list[str]] = {"numeric": [], "object": [], "bool": [], "other": []}
    for col in cols_to_process:
        dtype = df[col].dtype
        if dtype == bool:
            dtype_groups["bool"].append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            dtype_groups["numeric"].append(col)
        elif dtype == object:
            dtype_groups["object"].append(col)
        else:
            dtype_groups["other"].append(col)

    logger.info(
        f"dtype別カラム数: 数値={len(dtype_groups['numeric'])}, "
        f"カテゴリ(object)={len(dtype_groups['object'])}, "
        f"bool={len(dtype_groups['bool'])}, "
        f"その他={len(dtype_groups['other'])}"
    )
    if dtype_groups["object"]:
        logger.info(f"カテゴリカラム一覧: {dtype_groups['object']}")
    else:
        logger.warning("カテゴリカラム(object型)が検出されませんでした。"
                       "CSVの文字列カラムが数値として読み込まれている可能性があります。")
    logger.debug(f"型判定対象カラム ({len(cols_to_process)}件): {cols_to_process}")

    for col in cols_to_process:
        if col not in df.columns:
            continue  # 日付分解で削除済み

        dtype = df[col].dtype
        spec = column_config.get(col, {})
        forced_type      = spec.get("type")                    # "numeric" | "date" | "categorical" | None
        forced_enc       = spec.get("encoding")               # "ohe" | "frequency" | "auto" | None
        forced_topn      = spec.get("ohe_top_n")              # int | None
        forced_threshold = spec.get("ohe_coverage_threshold") # float | None
        logger.debug(
            f"カラム処理: {col} (dtype={dtype}, "
            f"forced_type={forced_type}, forced_enc={forced_enc}, "
            f"forced_topn={forced_topn}, forced_threshold={forced_threshold})"
        )

        # JSON で type=numeric が指定された場合は強制数値変換
        if forced_type == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
            logger.debug(f"  [JSON] numeric 強制変換: {col}")

        # JSON で type=date が指定された場合は強制日付分解
        elif forced_type == "date":
            parsed = pd.to_datetime(df[col], errors="coerce")
            df[col] = parsed
            date_parts = {
                "year": df[col].dt.year, "month": df[col].dt.month,
                "day": df[col].dt.day, "hour": df[col].dt.hour,
                "weekday": df[col].dt.weekday,
            }
            generated = []
            for part, values in date_parts.items():
                new_col = f"{col}_{part}"
                df[new_col] = values
                generated.append(new_col)
            df = df.drop(columns=[col])
            encode_summaries.append({
                "column": col, "encoding": "datetime",
                "n_unique": None, "coverage_top_n": None,
                "other_count": None, "generated_columns": ", ".join(generated),
            })
            logger.info(f"  [JSON] date 強制分解: {col} → {generated}")

        # JSON で type=categorical が指定、または dtype==object の場合
        elif forced_type == "categorical" or (forced_type is None and dtype == object):
            df, enc_summary = _encode_column(
                df, col, n_rows, cfg,
                forced_encoding=forced_enc,
                forced_ohe_top_n=forced_topn,
                forced_ohe_coverage_threshold=forced_threshold,
            )
            encode_summaries.append(enc_summary)

        elif dtype == bool:
            df[col] = df[col].astype(int)
            logger.debug(f"  bool → int: {col}")

        elif pd.api.types.is_numeric_dtype(dtype):
            n_null = int(df[col].isnull().sum())
            if n_null > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(
                    f"  数値欠損補完 [{col}]: {n_null}件 → 中央値={median_val:.4g}"
                )

        else:
            logger.debug(f"  未知dtype({dtype}) → 数値変換試行: {col}")
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                n_null = int(df[col].isnull().sum())
                if n_null > 0:
                    df[col] = df[col].fillna(df[col].median())
                    logger.debug(f"  変換後欠損補完 [{col}]: {n_null}件")
            except Exception as e:
                logger.warning(f"  変換不可のため除外 [{col}]: {e}")
                logger.debug(traceback.format_exc())
                excluded_cols.append(col)
                df = df.drop(columns=[col])

    # ---- 非数値カラムの最終除外 ----
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(f"数値変換できないカラムを除外: {non_numeric}")
        excluded_cols.extend(non_numeric)
        df = df.select_dtypes(include=[np.number])

    logger.info(
        f"前処理後: {len(df.columns)}カラム "
        f"(除外済み累計: {len(excluded_cols)}件, "
        f"カテゴリ処理: {len(encode_summaries)}件)"
    )
    _log_df_summary(df, "前処理後")
    return df, excluded_cols, encode_summaries


# ============================================================
# 特徴量選択
# ============================================================

@timed("特徴量選択")
def select_features(
    df: pd.DataFrame, excluded_cols: list[str], cfg: Config
) -> tuple[pd.DataFrame, list[str]]:
    """低分散・高相関カラムを除外する特徴量選択。

    Args:
        df: 前処理済みDataFrame
        excluded_cols: これまでの除外カラムリスト（破壊的に追記）
        cfg: 実行設定

    Returns:
        (選択後DataFrame, 更新された除外カラムリスト)
    """
    # ---- 分散フィルタ ----
    variances = df.var()
    logger.debug(f"全カラムの分散（昇順）:\n{variances.sort_values().to_string()}")

    low_var_cols = variances[variances < cfg.variance_threshold].index.tolist()
    if low_var_cols:
        for col in low_var_cols:
            logger.debug(
                f"  低分散除外: {col} (var={variances[col]:.6g} < {cfg.variance_threshold})"
            )
        logger.info(f"低分散カラムを除外 ({len(low_var_cols)}件): {low_var_cols}")
        excluded_cols.extend(low_var_cols)
        df = df.drop(columns=low_var_cols)

    if df.shape[1] == 0:
        return df, excluded_cols

    # ---- 相関フィルタ ----
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # 高相関ペアを全件 DEBUG 出力
    high_corr_pairs = [
        (col_a, col_b, float(upper.loc[col_b, col_a]))
        for col_b in upper.index
        for col_a in upper.columns
        if pd.notna(upper.loc[col_b, col_a])
        and upper.loc[col_b, col_a] > cfg.corr_threshold
    ]
    if high_corr_pairs:
        logger.debug(f"高相関ペア (>{cfg.corr_threshold}):")
        for col_a, col_b, r in sorted(high_corr_pairs, key=lambda x: -x[2]):
            logger.debug(f"  {col_a} ↔ {col_b}: {r:.4f}")

    high_corr_cols = [
        col for col in upper.columns if any(upper[col] > cfg.corr_threshold)
    ]
    if high_corr_cols:
        logger.info(
            f"高相関カラムを除外 ({len(high_corr_cols)}件): {high_corr_cols}"
        )
        excluded_cols.extend(high_corr_cols)
        df = df.drop(columns=high_corr_cols)

    logger.info(f"特徴量選択後: {len(df.columns)}カラム")
    logger.debug(f"選択後カラム: {df.columns.tolist()}")
    return df, excluded_cols


# ============================================================
# ラベル付き異常のパーセンタイル確認
# ============================================================

def _log_labeled_percentile(
    anomaly_score: np.ndarray,
    labeled_idx: np.ndarray,
    label: str = "",
) -> None:
    """ラベル付き異常の異常スコア中央値が全体の上位何%にいるかをINFOで出力する。

    Args:
        anomaly_score: 全レコードの異常スコア配列
        labeled_idx: ラベル付き異常のインデックス配列
        label: ログの識別ラベル
    """
    labeled_scores = anomaly_score[labeled_idx]
    median_score   = float(np.median(labeled_scores))
    # 全体の中で median_score より低いスコアの割合 = 下位X% → 上位(100-X)%
    percentile_from_bottom = float((anomaly_score < median_score).mean() * 100)
    top_pct = 100.0 - percentile_from_bottom

    prefix = f"[{label}] " if label else ""
    logger.info(
        f"{prefix}ラベル付き異常スコアの中央値: {median_score:.6f}  "
        f"→ 全体の上位 {top_pct:.2f}% "
        f"(N={len(labeled_idx)}件, "
        f"min={labeled_scores.min():.4f}, max={labeled_scores.max():.4f})"
    )


# ============================================================
# Optunaハイパーパラメータチューニング
# ============================================================

@timed("Optunaチューニング")
def tune_hyperparams(
    X_scaled: np.ndarray,
    original_df: pd.DataFrame,
    cfg: Config,
) -> Config:
    """Optunaでハイパーパラメータを探索し、最適なConfigを返す。

    目的関数: ラベル付き異常レコードの anomaly_score 中央値を最大化。
    探索対象: contamination / max_features / max_samples

    Args:
        X_scaled: 標準化済み特徴量行列
        original_df: 元データ（ラベル列を含む）
        cfg: 現在の設定

    Returns:
        best_params で上書きした新しい Config

    Raises:
        AnomalyDetectionError: ラベル列が存在しない / 異常レコードが0件
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise AnomalyDetectionError(
            "optunaがインストールされていません。"
            " pip install optuna を実行してください。"
        )

    if cfg.label_col not in original_df.columns:
        raise AnomalyDetectionError(
            f"ラベル列 '{cfg.label_col}' がデータに存在しません。"
            f" データのカラム: {original_df.columns.tolist()}"
        )

    labeled_idx = np.where(
        original_df[cfg.label_col].values == cfg.label_anomaly_value
    )[0]
    if len(labeled_idx) == 0:
        raise AnomalyDetectionError(
            f"ラベル列 '{cfg.label_col}' に異常値 '{cfg.label_anomaly_value}' が"
            f" 1件も存在しません。"
        )

    logger.info(
        f"チューニング開始: ラベル列={cfg.label_col}, "
        f"異常値={cfg.label_anomaly_value}, "
        f"ラベル付き異常件数={len(labeled_idx)}, "
        f"試行回数={cfg.n_trials}"
    )

    def objective(trial: "optuna.Trial") -> float:
        contamination = trial.suggest_float("contamination", 0.001, 0.10, log=True)
        max_features  = trial.suggest_float("max_features",  0.3,   1.0)
        max_samples   = trial.suggest_float("max_samples",   0.1,   1.0)

        model = IsolationForest(
            n_estimators=cfg.n_estimators,
            contamination=contamination,
            max_features=max_features,
            max_samples=max_samples,
            random_state=cfg.random_state,
        )
        model.fit(X_scaled)
        scores = -model.score_samples(X_scaled)
        return float(np.median(scores[labeled_idx]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

    best = study.best_params
    best_value = study.best_value
    logger.info(
        f"チューニング完了: best_value(中央値)={best_value:.6f}\n"
        f"  contamination={best['contamination']:.6f}\n"
        f"  max_features={best['max_features']:.4f}\n"
        f"  max_samples={best['max_samples']:.4f}"
    )

    # best_params でのスコアをパーセンタイルで確認
    best_model = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=best["contamination"],
        max_features=best["max_features"],
        max_samples=best["max_samples"],
        random_state=cfg.random_state,
    )
    best_model.fit(X_scaled)
    best_scores = -best_model.score_samples(X_scaled)
    _log_labeled_percentile(best_scores, labeled_idx, "チューニング最良モデル")

    # チューニング結果をCSVに保存
    trials_df = study.trials_dataframe()[
        ["number", "value", "params_contamination", "params_max_features", "params_max_samples"]
    ].sort_values("value", ascending=False)
    out_path = cfg.out_dir / "tuning_results.csv"
    trials_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"出力: tuning_results.csv ({len(trials_df)}試行)")

    # best_params を Config に反映して返す
    return Config(
        contamination=best["contamination"],
        max_features=best["max_features"],
        max_samples=best["max_samples"],
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        missing_rate_threshold=cfg.missing_rate_threshold,
        date_parse_threshold=cfg.date_parse_threshold,
        ohe_top_n=cfg.ohe_top_n,
        ohe_coverage_threshold=cfg.ohe_coverage_threshold,
        variance_threshold=cfg.variance_threshold,
        corr_threshold=cfg.corr_threshold,
        shap_all=cfg.shap_all,
        pca_variance_warning=cfg.pca_variance_warning,
        label_col=cfg.label_col,
        label_anomaly_value=cfg.label_anomaly_value,
        n_trials=cfg.n_trials,
        in_dir=cfg.in_dir,
        out_dir=cfg.out_dir,
        column_config_path=cfg.column_config_path,
    )


# ============================================================
# Isolation Forest
# ============================================================

@timed("Isolation Forest")
def run_isolation_forest(
    X_scaled: np.ndarray, cfg: Config
) -> tuple[IsolationForest, np.ndarray, np.ndarray]:
    """Isolation Forestで異常スコアと異常フラグを算出する。

    Args:
        X_scaled: 標準化済み特徴量行列
        cfg: 実行設定

    Returns:
        (学習済みモデル, anomaly_score, is_anomaly)
    """
    logger.info(
        f"パラメータ: n_estimators={cfg.n_estimators}, "
        f"contamination={cfg.contamination}, "
        f"max_features={cfg.max_features}, "
        f"max_samples={cfg.max_samples}, "
        f"random_state={cfg.random_state}"
    )
    logger.debug(f"入力行列: shape={X_scaled.shape}")

    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        max_features=cfg.max_features,
        max_samples=cfg.max_samples,
        random_state=cfg.random_state,
    )
    model.fit(X_scaled)
    logger.debug("モデル学習完了")

    # score_samples() の出力をマイナス反転（高いほど異常）
    anomaly_score = -model.score_samples(X_scaled)
    # predict(): -1=異常 → 1, 1=正常 → 0
    is_anomaly = (model.predict(X_scaled) == -1).astype(int)

    n_anomaly = int(is_anomaly.sum())
    n_total = len(is_anomaly)
    logger.info(
        f"異常件数: {n_anomaly} / {n_total} 件 ({n_anomaly / n_total:.1%})"
    )
    logger.debug(
        f"anomaly_score: min={anomaly_score.min():.4f}, "
        f"max={anomaly_score.max():.4f}, "
        f"mean={anomaly_score.mean():.4f}, "
        f"std={anomaly_score.std():.4f}, "
        f"p95={float(np.percentile(anomaly_score, 95)):.4f}"
    )
    threshold_score = float(
        np.percentile(anomaly_score, 100 * (1 - cfg.contamination))
    )
    logger.debug(f"異常判定閾値スコア (上位{cfg.contamination:.0%}): {threshold_score:.4f}")

    return model, anomaly_score, is_anomaly


# ============================================================
# SHAP
# ============================================================

@timed("SHAP分析")
def run_shap(
    model: IsolationForest,
    X_scaled: np.ndarray,
    is_anomaly: np.ndarray,
    feature_names: list[str],
    cfg: Config,
) -> tuple[Optional[pd.DataFrame], np.ndarray, np.ndarray]:
    """SHAP値を算出し、各レコードの最大寄与特徴量を特定する。

    計算対象は cfg.shap_all=False の場合は is_anomaly=1 のレコードのみ。
    エラー発生時は警告ログを出力しスキップ（result.csv は出力継続）。

    Args:
        model: 学習済みIsolation Forestモデル
        X_scaled: 標準化済み特徴量行列
        is_anomaly: 異常フラグ配列
        feature_names: 特徴量名リスト
        cfg: 実行設定

    Returns:
        (shap_df or None, top_feature_arr, top_shap_value_arr)
    """
    n = len(X_scaled)
    top_feature_arr = np.full(n, "", dtype=object)
    top_shap_value_arr = np.zeros(n)

    try:
        logger.debug("TreeExplainer 初期化中...")
        explainer = shap.TreeExplainer(model)

        if cfg.shap_all:
            indices = np.arange(n)
            logger.info(f"SHAP: 全件計算 ({n}件)")
        else:
            indices = np.where(is_anomaly == 1)[0]
            logger.info(f"SHAP: 異常レコードのみ計算 ({len(indices)} / {n}件)")

        if len(indices) == 0:
            logger.warning("SHAP計算対象レコードがありません。")
            return None, top_feature_arr, top_shap_value_arr

        X_target = X_scaled[indices]
        logger.debug(f"SHAP計算対象行列: shape={X_target.shape}")

        shap_values = explainer.shap_values(X_target)

        if isinstance(shap_values, list):
            logger.debug("shap_values が list 形式 → [0] を使用")
            shap_values = shap_values[0]

        # score_samples ベースの SHAP を反転
        # → 正の値が異常方向への寄与、負の値が正常方向への寄与
        shap_values = -shap_values

        logger.debug(
            f"SHAP値行列（反転後）: shape={shap_values.shape}, "
            f"min={shap_values.min():.4f}, max={shap_values.max():.4f}, "
            f"mean|SHAP|={np.abs(shap_values).mean():.4f}"
        )

        # 特徴量別 平均|SHAP| ランキング（上位10件を DEBUG 出力）
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        ranking = sorted(
            zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True
        )
        logger.debug("特徴量別 平均|SHAP| ランキング (上位10):")
        for rank, (feat, val) in enumerate(ranking[:10], 1):
            logger.debug(f"  {rank:2d}. {feat}: {val:.4f}")

        shap_df = pd.DataFrame(
            shap_values, columns=feature_names, index=indices
        )

        # 最大寄与特徴量の特定
        abs_shap = np.abs(shap_values)
        top_idx = np.argmax(abs_shap, axis=1)
        for i, orig_idx in enumerate(indices):
            top_feature_arr[orig_idx] = feature_names[top_idx[i]]
            top_shap_value_arr[orig_idx] = abs_shap[i, top_idx[i]]

        logger.info("SHAP計算完了")
        return shap_df, top_feature_arr, top_shap_value_arr

    except Exception as e:
        logger.warning(f"SHAP計算エラーのためスキップ: {e}")
        logger.debug(traceback.format_exc())
        return None, top_feature_arr, top_shap_value_arr


# ============================================================
# PCA
# ============================================================

@timed("PCA")
def run_pca(X_scaled: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """PCAで2次元に圧縮し、主成分座標と寄与率を返す。

    Args:
        X_scaled: 標準化済み特徴量行列
        cfg: 実行設定

    Returns:
        (pca_coords [n×2], explained_variance_ratio)
    """
    n_components = min(2, X_scaled.shape[1])
    logger.debug(
        f"PCA: n_components={n_components}, input_shape={X_scaled.shape}"
    )

    pca = PCA(n_components=n_components, random_state=cfg.random_state)
    coords = pca.fit_transform(X_scaled)

    variance_ratio = pca.explained_variance_ratio_
    cumulative = float(np.sum(variance_ratio))
    pc1_var = float(variance_ratio[0])
    pc2_var = float(variance_ratio[1]) if n_components > 1 else 0.0

    logger.info(
        f"PCA寄与率: PC1={pc1_var:.3f} ({pc1_var*100:.1f}%), "
        f"PC2={pc2_var:.3f} ({pc2_var*100:.1f}%), "
        f"累積={cumulative:.3f} ({cumulative*100:.1f}%)"
    )
    logger.debug(f"固有値: {pca.explained_variance_.tolist()}")
    logger.debug(
        f"PC1スコア範囲: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}]"
    )
    if n_components > 1:
        logger.debug(
            f"PC2スコア範囲: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]"
        )

    if cumulative < cfg.pca_variance_warning:
        logger.warning(
            f"PCA累積寄与率 ({cumulative:.3f}) が "
            f"警告閾値 ({cfg.pca_variance_warning}) を下回っています。"
        )

    # 常に2列確保（特徴量が1次元しかない場合）
    if n_components == 1:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])
        variance_ratio = np.append(variance_ratio, 0.0)

    return coords, variance_ratio


# ============================================================
# 出力
# ============================================================

@dataclass
class ModelResults:
    """モデリング・分析の結果を集約するデータクラス。"""
    anomaly_score: np.ndarray
    is_anomaly: np.ndarray
    shap_df: Optional[pd.DataFrame]
    top_feature_arr: np.ndarray
    top_shap_value_arr: np.ndarray
    pca_coords: np.ndarray
    pca_variance: np.ndarray


@timed("結果出力")
def save_outputs(
    original_df: pd.DataFrame,
    results: ModelResults,
    excluded_cols: list[str],
    encode_summaries: list[dict],
    cfg: Config,
) -> None:
    """結果ファイル一式を out/ フォルダに保存する。

    Args:
        original_df: 元の入力DataFrame（変換前）
        results: モデリング結果
        excluded_cols: 前処理・特徴量選択で除外したカラム名リスト
        encode_summaries: カテゴリ変数のエンコードサマリ
        cfg: 実行設定
    """
    cfg.out_dir.mkdir(exist_ok=True)

    # ---- result.csv ----
    result_df = original_df.copy()
    result_df["anomaly_score"] = results.anomaly_score
    result_df["is_anomaly"] = results.is_anomaly
    result_df["top_feature"] = results.top_feature_arr
    result_df["top_shap_value"] = results.top_shap_value_arr
    out_path = cfg.out_dir / "result.csv"
    result_df.to_csv(out_path, index=True, index_label="index", encoding="utf-8-sig")
    logger.info(
        f"出力: result.csv ({len(result_df)}件, {len(result_df.columns)}カラム)"
    )
    logger.debug(f"  → {out_path.resolve()}")

    # ---- shap_summary.csv ----
    if results.shap_df is not None:
        out_path = cfg.out_dir / "shap_summary.csv"
        results.shap_df.to_csv(out_path, encoding="utf-8-sig")
        logger.info(f"出力: shap_summary.csv ({len(results.shap_df)}件)")
        logger.debug(f"  → {out_path.resolve()}")
    else:
        logger.info("SHAP出力スキップ")

    # ---- pca_2d.csv（PC1, PC2, 異常情報, 元データ全列）----
    pca_df = pd.DataFrame({
        "PC1": results.pca_coords[:, 0],
        "PC2": results.pca_coords[:, 1],
        "anomaly_score": results.anomaly_score,
        "is_anomaly": results.is_anomaly,
        "top_feature": results.top_feature_arr,
    })
    for col in original_df.columns:
        pca_df[col] = original_df[col].values
    out_path = cfg.out_dir / "pca_2d.csv"
    pca_df.to_csv(out_path, index=True, index_label="index", encoding="utf-8-sig")
    logger.info(f"出力: pca_2d.csv ({len(pca_df)}件, {len(pca_df.columns)}カラム)")
    logger.debug(f"  → {out_path.resolve()}")

    # ---- pca_variance.csv ----
    n_comp = len(results.pca_variance)
    variance_df = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(n_comp)],
        "explained_variance_ratio": results.pca_variance,
        "cumulative_variance_ratio": np.cumsum(results.pca_variance),
    })
    out_path = cfg.out_dir / "pca_variance.csv"
    variance_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("出力: pca_variance.csv")

    # ---- excluded_columns.txt ----
    content = "\n".join(excluded_cols) if excluded_cols else "(除外カラムなし)"
    out_path = cfg.out_dir / "excluded_columns.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"出力: excluded_columns.txt ({len(excluded_cols)}件)")
    logger.debug(f"除外カラム一覧: {excluded_cols}")

    # ---- encoding_summary.csv ----
    if encode_summaries:
        enc_df = pd.DataFrame(encode_summaries, columns=[
            "column", "encoding", "n_unique",
            "coverage_top_n", "other_count", "generated_columns",
        ])
        out_path = cfg.out_dir / "encoding_summary.csv"
        enc_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"出力: encoding_summary.csv ({len(enc_df)}件)")
        logger.debug(f"\n{enc_df.to_string(index=False)}")
    else:
        logger.info("カテゴリ変数なし: encoding_summary.csv はスキップ")


# ============================================================
# メイン処理
# ============================================================

def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="異常検知システム (Isolation Forest + SHAP + PCA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- パス ---
    parser.add_argument("--in-dir", type=Path, default=Path("in"),
                        help="入力CSVフォルダのパス")
    parser.add_argument("--out-dir", type=Path, default=Path("out"),
                        help="出力フォルダのパス")
    parser.add_argument("--encoding", type=str, default="cp932",
                        help="入力CSVの文字コード (cp932=Shift-JIS, utf-8 など)")

    # --- Isolation Forest ---
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="異常割合の想定値 (0.0〜0.5)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Isolation Forestの木の本数")
    parser.add_argument("--max-samples", type=float, default=1.0,
                        help="各木で使うサンプルの割合 (0.0〜1.0)")

    # --- Optunaチューニング ---
    parser.add_argument("--label-col", type=str, default=None,
                        help="ラベル列名。指定するとOptunaチューニングを実行")
    parser.add_argument("--label-anomaly-value", type=int, default=1,
                        help="異常を示すラベル値")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Optunaの試行回数")

    # --- 前処理 ---
    parser.add_argument("--missing-rate-threshold", type=float, default=0.80,
                        help="欠損率がこの値を超えるカラムを除外")
    parser.add_argument("--ohe-top-n", type=int, default=10,
                        help="OHE対象とする上位カテゴリ数")
    parser.add_argument("--ohe-coverage-threshold", type=float, default=0.50,
                        help="上位N件のカバレッジがこの値以上ならOHE")
    parser.add_argument("--variance-threshold", type=float, default=0.01,
                        help="分散がこの値未満のカラムを除外")
    parser.add_argument("--corr-threshold", type=float, default=0.95,
                        help="相関係数がこの値を超えるペアの片方を除外")

    # --- SHAP ---
    parser.add_argument("--shap-all", action=argparse.BooleanOptionalAction, default=False,
                        help="全件SHAP計算。--no-shap-all で異常レコードのみに切り替え")

    # --- PCA ---
    parser.add_argument("--pca-variance-warning", type=float, default=0.50,
                        help="PCA累積寄与率がこの値を下回ると警告")

    # --- カラム設定 ---
    parser.add_argument("--column-config", type=Path, default=None,
                        help="カラム設定JSONファイルのパス (例: column_config.json)")

    return parser.parse_args()


def main() -> None:
    """異常検知システムのメイン処理を実行する。"""
    args = _parse_args()

    cfg = Config(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        input_encoding=args.encoding,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        missing_rate_threshold=args.missing_rate_threshold,
        ohe_top_n=args.ohe_top_n,
        ohe_coverage_threshold=args.ohe_coverage_threshold,
        variance_threshold=args.variance_threshold,
        corr_threshold=args.corr_threshold,
        shap_all=args.shap_all,
        pca_variance_warning=args.pca_variance_warning,
        label_col=args.label_col,
        label_anomaly_value=args.label_anomaly_value,
        n_trials=args.n_trials,
        column_config_path=args.column_config,
    )

    setup_logging(cfg.out_dir)

    logger.info("=" * 60)
    logger.info("異常検知システム 開始")
    logger.info(
        f"設定: contamination={cfg.contamination}, "
        f"n_estimators={cfg.n_estimators}, "
        f"ohe_top_n={cfg.ohe_top_n}, "
        f"ohe_coverage_threshold={cfg.ohe_coverage_threshold}, "
        f"missing_rate_threshold={cfg.missing_rate_threshold}, "
        f"variance_threshold={cfg.variance_threshold}, "
        f"corr_threshold={cfg.corr_threshold}, "
        f"shap_all={cfg.shap_all}"
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()

    try:
        # 1. データ読み込み
        original_df = load_csvs(cfg)

        # 2. 前処理
        processed_df, excluded_cols, encode_summaries = preprocess(original_df.copy(), cfg)
        if processed_df.shape[1] == 0:
            raise AnomalyDetectionError("前処理後に有効なカラムが0件です。")

        # 3. 特徴量選択
        feature_df, excluded_cols = select_features(processed_df, excluded_cols, cfg)
        if feature_df.shape[1] == 0:
            raise AnomalyDetectionError("特徴量選択後に有効なカラムが0件です。")

        # 4. 標準化前の NaN 最終チェック・補完
        nan_counts = feature_df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
            logger.warning(
                f"標準化前に NaN を検出。中央値で補完します:\n{nan_cols.to_string()}"
            )
            feature_df = feature_df.fillna(feature_df.median())

        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df.values)

        # 標準化後の NaN チェック（無限大・数値異常も検出）
        if not np.isfinite(X_scaled).all():
            n_nan = int(np.isnan(X_scaled).sum())
            n_inf = int(np.isinf(X_scaled).sum())
            logger.warning(
                f"標準化後に非有限値を検出 (NaN={n_nan}, Inf={n_inf})。0 で補完します。"
            )
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"標準化完了: shape={X_scaled.shape}")
        logger.debug(
            f"標準化後行列: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}"
        )

        feature_names = feature_df.columns.tolist()
        logger.debug(f"特徴量名 ({len(feature_names)}件): {feature_names}")

        # 5. Optunaチューニング（--label-col 指定時のみ）
        if cfg.label_col is not None:
            cfg = tune_hyperparams(X_scaled, original_df, cfg)

        # 6. Isolation Forest
        model, anomaly_score, is_anomaly = run_isolation_forest(X_scaled, cfg)

        # ラベル付き異常のパーセンタイル確認（--label-col 指定時のみ）
        if cfg.label_col is not None:
            labeled_idx = np.where(
                original_df[cfg.label_col].values == cfg.label_anomaly_value
            )[0]
            _log_labeled_percentile(anomaly_score, labeled_idx, "最終モデル")

        # 7. SHAP
        shap_df, top_feature_arr, top_shap_value_arr = run_shap(
            model, X_scaled, is_anomaly, feature_names, cfg
        )

        # 8. PCA
        pca_coords, pca_variance = run_pca(X_scaled, cfg)

        # 9. 出力
        results = ModelResults(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            shap_df=shap_df,
            top_feature_arr=top_feature_arr,
            top_shap_value_arr=top_shap_value_arr,
            pca_coords=pca_coords,
            pca_variance=pca_variance,
        )
        save_outputs(original_df, results, excluded_cols, encode_summaries, cfg)

    except AnomalyDetectionError as e:
        logger.error(f"処理中断: {e}")
        sys.exit(1)

    total_elapsed = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info(f"異常検知システム 完了  (合計: {total_elapsed:.2f}秒)")
    logger.info(f"結果フォルダ: {cfg.out_dir.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
