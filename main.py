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
import pickle
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

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
    label_col: Optional[str] = None         # ラベル列名（パーセンタイル表示・チューニングに使用）
    label_anomaly_value: int = 1            # 異常を示すラベル値
    tune: bool = False                      # True: Optunaチューニングを実行
    n_trials: int = 100                     # Optunaの試行回数

    # --- モデル手法 ---
    method: str = "if"                      # "if": Isolation Forest / "lof": Local Outlier Factor
    lof_n_neighbors: int = 20               # LOFの近傍数
    lof_tune_weights: bool = True           # LOF+Optuna時に特徴量重みをチューニングするか

    # --- パス ---
    in_dir: Path = field(default_factory=lambda: Path("in"))
    out_dir: Path = field(default_factory=lambda: Path("out"))
    model_dir: Path = field(default_factory=lambda: Path("models"))
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
    prefix_n: Optional[int] = None,                     # 先頭N文字に切り詰め（Noneは無効）
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
        prefix_n: 先頭N文字に切り詰めてからエンコード（Noneは無効）

    Returns:
        (エンコード処理後のDataFrame, エンコードサマリ dict)
    """
    # ---- 先頭N文字への切り詰め ----
    if prefix_n is not None:
        n_unique_before = df[col].nunique()
        df[col] = df[col].astype(str).str[:prefix_n]
        n_unique_after = df[col].nunique()
        logger.info(
            f"  先頭{prefix_n}文字に切り詰め [{col}]: "
            f"ユニーク数 {n_unique_before} → {n_unique_after}"
        )

    summary: dict = {
        "column": col,
        "encoding": "",
        "n_unique": int(df[col].nunique()),
        "coverage_top_n": None,
        "other_count": None,
        "generated_columns": "",
        "prefix_n": prefix_n,
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
        prefix_n         = spec.get("prefix_n")               # int | None
        log_transform    = bool(spec.get("log_transform", False))  # True: log1p変換
        logger.debug(
            f"カラム処理: {col} (dtype={dtype}, "
            f"forced_type={forced_type}, forced_enc={forced_enc}, "
            f"forced_topn={forced_topn}, forced_threshold={forced_threshold}, "
            f"prefix_n={prefix_n}, log_transform={log_transform})"
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
                prefix_n=prefix_n,
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

        # ---- 対数変換（log_transform=true かつ数値カラムのみ）----
        if log_transform and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            min_val = float(df[col].min())
            if min_val < 0:
                logger.warning(
                    f"  log変換スキップ [{col}]: 負の値あり (min={min_val:.4g})"
                )
            else:
                before_mean = float(df[col].mean())
                df[col] = np.log1p(df[col])
                logger.info(
                    f"  対数変換 log1p [{col}]: "
                    f"変換前 mean={before_mean:.4g} → 変換後 mean={df[col].mean():.4g}"
                )

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
    df: pd.DataFrame,
    excluded_cols: list[str],
    cfg: Config,
    encode_summaries: list[dict],
) -> tuple[pd.DataFrame, list[str]]:
    """低分散・高相関カラムを除外する特徴量選択。

    OHEで展開されたカラムは相関フィルタの除外対象から保護する。
    OHEグループの一部だけが消えると元カラムの情報が欠落するため。

    Args:
        df: 前処理済みDataFrame
        excluded_cols: これまでの除外カラムリスト（破壊的に追記）
        cfg: 実行設定
        encode_summaries: preprocess() が返すエンコードサマリ（OHE列特定に使用）

    Returns:
        (選択後DataFrame, 更新された除外カラムリスト)
    """
    # OHEで生成されたカラム名のセットを構築（相関フィルタから保護）
    ohe_protected: set[str] = set()
    for s in encode_summaries:
        if s.get("encoding") == "ohe":
            for col in s.get("generated_columns", "").split(","):
                col = col.strip()
                if col and col in df.columns:
                    ohe_protected.add(col)
    if ohe_protected:
        logger.info(
            f"相関フィルタ保護（OHE列）: {len(ohe_protected)}件 "
            f"— OHEグループの部分削除を防ぐため除外対象から除く"
        )
        logger.debug(f"  保護対象: {sorted(ohe_protected)}")

    # ---- 分散フィルタ（OHE列は保護）----
    variances = df.var()
    logger.debug(f"全カラムの分散（昇順）:\n{variances.sort_values().to_string()}")

    low_var_cols = [
        col for col, var in variances.items()
        if var < cfg.variance_threshold and col not in ohe_protected
    ]
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

    # OHE列は除外対象から保護
    high_corr_cols = [
        col for col in upper.columns
        if col not in ohe_protected and any(upper[col] > cfg.corr_threshold)
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
# チューニング結果の保存・ロード
# ============================================================

def save_best_params(
    cfg: Config,
    lof_weights: Optional[np.ndarray],
) -> None:
    """チューニング結果（ハイパーパラメータ + LOF重み）をpklで保存する。

    Args:
        cfg: チューニング後の Config
        lof_weights: LOF用重み配列（IFの場合は None）
    """
    pkl_path = cfg.model_dir / "best_params.pkl"
    cfg.model_dir.mkdir(exist_ok=True)
    data = {
        "method":          cfg.method,
        "contamination":   cfg.contamination,
        "lof_n_neighbors": cfg.lof_n_neighbors,
        "max_features":    cfg.max_features,
        "max_samples":     cfg.max_samples,
        "lof_weights":     lof_weights,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"チューニング結果を保存: {pkl_path.resolve()}")
    logger.info(
        f"  method={data['method']}, contamination={data['contamination']:.6f}"
        + (f", n_neighbors={data['lof_n_neighbors']}" if cfg.method == "lof" else
           f", max_features={data['max_features']:.4f}, max_samples={data['max_samples']:.4f}")
    )


def load_best_params(
    cfg: Config,
) -> tuple[Config, Optional[np.ndarray]]:
    """pklからチューニング結果を読み込み、Config と LOF重み配列を返す。

    Args:
        cfg: 現在の Config（pkl の値で上書きする）

    Returns:
        (更新された Config, LOF用重み配列 or None)

    Raises:
        AnomalyDetectionError: ファイルの読み込み・形式エラー
    """
    pkl_path = cfg.model_dir / "best_params.pkl"
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise AnomalyDetectionError(
            f"チューニング結果pklの読み込みエラー: {e}\n  ファイル: {pkl_path}"
        ) from e

    lof_weights: Optional[np.ndarray] = data.get("lof_weights")

    new_cfg = replace(
        cfg,
        contamination=data["contamination"],
        lof_n_neighbors=data.get("lof_n_neighbors", cfg.lof_n_neighbors),
        max_features=data.get("max_features", cfg.max_features),
        max_samples=data.get("max_samples", cfg.max_samples),
    )

    logger.info(f"チューニング結果をロード: {pkl_path.resolve()}")
    logger.info(
        f"  method={data.get('method', '?')}, contamination={new_cfg.contamination:.6f}"
        + (f", n_neighbors={new_cfg.lof_n_neighbors}" if cfg.method == "lof" else
           f", max_features={new_cfg.max_features:.4f}, max_samples={new_cfg.max_samples:.4f}")
    )
    if lof_weights is not None:
        logger.info(
            f"  重みづけ: min={lof_weights.min():.4f}, "
            f"max={lof_weights.max():.4f}, mean={lof_weights.mean():.4f}"
        )
    return new_cfg, lof_weights


# ============================================================
# カラムグループ構築（OHE展開前の元カラム → 展開後カラムのマッピング）
# ============================================================

def build_column_groups(
    feature_names: list[str],
    encode_summaries: list[dict],
) -> dict[str, list[str]]:
    """元カラム名をキーに、展開後の特徴量名リストをまとめた辞書を返す。

    OHEで複数列に展開されたカラムは元カラム名にまとめられる。
    日付分解されたカラムも元カラム名でグループ化される。
    数値カラムは1対1（元カラム名 = 特徴量名）。

    Args:
        feature_names: 前処理・特徴量選択後の特徴量名リスト
        encode_summaries: preprocess() が返すエンコードサマリリスト

    Returns:
        {元カラム名: [特徴量名, ...]} の辞書（feature_names の順序を保持）
    """
    # 展開後カラム → 元カラム名 の逆引き辞書を構築
    generated_to_original: dict[str, str] = {}
    for s in encode_summaries:
        orig = s["column"]
        generated = s.get("generated_columns") or ""
        for gen_col in [c.strip() for c in generated.split(",") if c.strip()]:
            generated_to_original[gen_col] = orig

    # feature_names を元カラム名でグループ化
    groups: dict[str, list[str]] = {}
    for feat in feature_names:
        orig = generated_to_original.get(feat, feat)  # 変換なし数値列は自己マッピング
        groups.setdefault(orig, []).append(feat)

    logger.debug(f"column_groups: {len(groups)}グループ / {len(feature_names)}特徴量")
    for orig, feats in groups.items():
        logger.debug(f"  {orig}: {feats}")
    return groups


# ============================================================
# Optunaハイパーパラメータチューニング
# ============================================================

@timed("Optunaチューニング")
def tune_hyperparams(
    X_scaled: np.ndarray,
    original_df: pd.DataFrame,
    cfg: Config,
    feature_names: list[str],
    column_groups: dict[str, list[str]],
) -> tuple[Config, Optional[np.ndarray]]:
    """Optunaでハイパーパラメータを探索し、最適な Config と重み配列を返す。

    目的関数: ラベル付き異常レコードの anomaly_score 中央値を最大化。

    IF 探索対象: contamination / max_features / max_samples
    LOF 探索対象: n_neighbors / contamination / 元カラムごとの特徴量重み

    LOF の重みは元カラム単位（OHE展開前）で決定し、展開後の全列に適用する。
    これにより「region に高い重みを与える」という直感的な操作が可能になる。

    Args:
        X_scaled: 標準化済み特徴量行列
        original_df: 元データ（ラベル列を含む）
        cfg: 現在の設定
        feature_names: 前処理後の特徴量名リスト
        column_groups: build_column_groups() の出力（元カラム → 特徴量名リスト）

    Returns:
        (best_params で上書きした Config, LOF用重み配列 or None)

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

    def _calc_top_pct(scores: np.ndarray) -> float:
        """ラベル付き異常の中央値スコアが全体の上位何%かを返す。（低いほど良い）"""
        median_score = float(np.median(scores[labeled_idx]))
        return 100.0 - float((scores < median_score).mean() * 100)

    def _objective_value(scores: np.ndarray) -> float:
        """最適化目的関数値: 上位%を下げる = percentile_from_bottom を最大化。"""
        top_pct = _calc_top_pct(scores)
        return 100.0 - top_pct  # percentile_from_bottom（大きいほど良い）

    # LOF用: 特徴量インデックスマップ（重み配列の生成に使用）
    feat_idx: dict[str, int] = {f: i for i, f in enumerate(feature_names)}

    def _build_weight_array(trial_params: dict) -> np.ndarray:
        """trial.params から元カラム単位の重みを展開した配列を生成する。"""
        weights = np.ones(len(feature_names))
        for orig_col, feat_cols in column_groups.items():
            w = trial_params.get(f"w_{orig_col}", 1.0)
            for fc in feat_cols:
                if fc in feat_idx:
                    weights[feat_idx[fc]] = w
        return weights

    if cfg.method == "lof":
        def objective(trial: "optuna.Trial") -> float:
            n_neighbors   = trial.suggest_int("n_neighbors", 5, 50)
            contamination = trial.suggest_float("contamination", 0.001, 0.10, log=True)
            # 重みチューニングが有効な場合のみ元カラム単位で重みを探索
            if cfg.lof_tune_weights:
                for orig_col in column_groups:
                    trial.suggest_float(f"w_{orig_col}", 0.1, 3.0)
            weights = _build_weight_array(trial.params) if cfg.lof_tune_weights else None
            X_weighted = X_scaled * weights if weights is not None else X_scaled
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=False,
            )
            model.fit_predict(X_weighted)
            scores = -model.negative_outlier_factor_
            trial.set_user_attr("top_pct", _calc_top_pct(scores))
            return _objective_value(scores)
    else:
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
            trial.set_user_attr("top_pct", _calc_top_pct(scores))
            return _objective_value(scores)

    def _log_callback(study: "optuna.Study", trial: "optuna.Trial") -> None:
        n = trial.number + 1
        val = trial.value if trial.value is not None else float("nan")
        top_pct = trial.user_attrs.get("top_pct", float("nan"))
        is_best = val == study.best_value
        best_label = " ★best更新" if is_best else ""
        params_str = ""
        if is_best:
            struct = {k: v for k, v in trial.params.items() if not k.startswith("w_")}
            params_str = "  (" + ", ".join(f"{k}={v:.4g}" for k, v in struct.items()) + ")"
            # 重みチューニング有効時は値のみをリスト表示
            if cfg.lof_tune_weights:
                w_vals = [v for k, v in trial.params.items() if k.startswith("w_")]
                if w_vals:
                    params_str += "  w=[" + ", ".join(f"{w:.2f}" for w in w_vals) + "]"
        logger.info(
            f"  trial {n:>4d}/{cfg.n_trials}  上位{top_pct:.2f}%{best_label}{params_str}"
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.n_trials, callbacks=[_log_callback])

    best = study.best_params
    best_value = study.best_value

    # 構造パラメータのみサマリ出力（重みは別途出力）
    struct_params = {k: v for k, v in best.items() if not k.startswith("w_")}
    params_summary = "\n".join(f"  {k}={v:.6g}" for k, v in struct_params.items())
    best_top_pct = 100.0 - best_value
    logger.info(
        f"チューニング完了: 上位{best_top_pct:.2f}%\n{params_summary}"
    )

    # LOF: 最良重みを構築してログ出力
    lof_weights: Optional[np.ndarray] = None
    if cfg.method == "lof":
        lof_weights = _build_weight_array(best)
        weight_by_orig = {
            orig: float(best.get(f"w_{orig}", 1.0))
            for orig in column_groups
        }
        sorted_weights = sorted(weight_by_orig.items(), key=lambda x: -x[1])
        logger.info("最良重み（元カラム単位、降順）:")
        for orig_col, w in sorted_weights:
            logger.info(f"  {orig_col:<30} {w:.4f}")

    # best_params でのスコアをパーセンタイルで確認
    if cfg.method == "lof":
        best_model = LocalOutlierFactor(
            n_neighbors=best["n_neighbors"],
            contamination=best["contamination"],
            novelty=False,
        )
        X_best = X_scaled * lof_weights if lof_weights is not None else X_scaled
        best_model.fit_predict(X_best)
        best_scores = -best_model.negative_outlier_factor_
    else:
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

    # チューニング結果をCSVに保存（重みカラムは w_ プレフィックス付きで含む）
    all_cols = study.trials_dataframe().columns.tolist()
    param_cols = [c for c in all_cols if c.startswith("params_")]
    trials_df = study.trials_dataframe()[
        ["number", "value"] + param_cols
    ].sort_values("value", ascending=False)
    out_path = cfg.out_dir / "tuning_results.csv"
    trials_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"出力: tuning_results.csv ({len(trials_df)}試行)")

    # best_params を Config に反映して返す
    common_kwargs = dict(
        contamination=best["contamination"],
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        input_encoding=cfg.input_encoding,
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
        method=cfg.method,
        lof_n_neighbors=cfg.lof_n_neighbors,
        lof_tune_weights=cfg.lof_tune_weights,
        in_dir=cfg.in_dir,
        out_dir=cfg.out_dir,
        model_dir=cfg.model_dir,
        column_config_path=cfg.column_config_path,
    )
    if cfg.method == "lof":
        common_kwargs["lof_n_neighbors"] = best["n_neighbors"]
    else:
        common_kwargs["max_features"] = best["max_features"]
        common_kwargs["max_samples"]  = best["max_samples"]

    return Config(**common_kwargs), lof_weights


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
# Local Outlier Factor
# ============================================================

@timed("Local Outlier Factor")
def run_lof(
    X_scaled: np.ndarray,
    cfg: Config,
    weights: Optional[np.ndarray] = None,
) -> tuple[LocalOutlierFactor, np.ndarray, np.ndarray]:
    """Local Outlier Factorで異常スコアと異常フラグを算出する。

    Args:
        X_scaled: 標準化済み特徴量行列
        cfg: 実行設定
        weights: 特徴量重み配列（Noneの場合は重みなし）。
                 チューニング結果の重みを渡すと距離計算に反映される。

    Returns:
        (学習済みモデル, anomaly_score, is_anomaly)
    """
    logger.info(
        f"パラメータ: n_neighbors={cfg.lof_n_neighbors}, "
        f"contamination={cfg.contamination}"
    )
    if weights is not None:
        logger.info(
            f"特徴量重みづけ適用: min={weights.min():.4f}, "
            f"max={weights.max():.4f}, mean={weights.mean():.4f}"
        )
    logger.debug(f"入力行列: shape={X_scaled.shape}")

    X_input = X_scaled * weights if weights is not None else X_scaled

    model = LocalOutlierFactor(
        n_neighbors=cfg.lof_n_neighbors,
        contamination=cfg.contamination,
        novelty=False,
    )
    is_anomaly_raw = model.fit_predict(X_input)

    # negative_outlier_factor_ はマイナス値（より負 = より異常）
    # 反転して「高いほど異常」なスコアにする
    anomaly_score = -model.negative_outlier_factor_
    is_anomaly = (is_anomaly_raw == -1).astype(int)

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
    model: Any,
    X_scaled: np.ndarray,
    is_anomaly: np.ndarray,
    feature_names: list[str],
    cfg: Config,
    shap_indices: Optional[np.ndarray] = None,
) -> tuple[Optional[pd.DataFrame], np.ndarray, np.ndarray]:
    """SHAP値を算出し、各レコードの最大寄与特徴量を特定する。

    計算対象の優先順位:
      1. shap_indices が指定されていればそのインデックス（label_col==1 など）
      2. cfg.shap_all=True なら全件
      3. それ以外は is_anomaly=1 のレコードのみ
    LOFモデルの場合はSHAPをスキップする。
    エラー発生時は警告ログを出力しスキップ（result.csv は出力継続）。

    Args:
        model: 学習済みモデル（IsolationForest / LocalOutlierFactor）
        X_scaled: 標準化済み特徴量行列
        is_anomaly: 異常フラグ配列
        feature_names: 特徴量名リスト
        cfg: 実行設定
        shap_indices: 計算対象レコードのインデックス配列（None の場合は自動決定）

    Returns:
        (shap_df or None, top_feature_arr, top_shap_value_arr)
    """
    n = len(X_scaled)
    top_feature_arr = np.full(n, "", dtype=object)
    top_shap_value_arr = np.zeros(n)

    # LOFはSHAP TreeExplainerに非対応のためスキップ
    if isinstance(model, LocalOutlierFactor):
        logger.info("SHAP: LOFモデルはSHAP非対応のためスキップします。")
        return None, top_feature_arr, top_shap_value_arr

    try:
        logger.debug("TreeExplainer 初期化中...")
        explainer = shap.TreeExplainer(model)

        if cfg.shap_all:
            indices = np.arange(n)
            logger.info(f"SHAP: 全件計算 ({n}件)")
        elif shap_indices is not None:
            indices = shap_indices
            logger.info(f"SHAP: ラベル付き異常レコードのみ計算 ({len(indices)} / {n}件)")
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
# LOF 距離ベース特徴量寄与度分析
# ============================================================

@timed("LOF特徴量寄与度分析")
def run_lof_contributions(
    X_input: np.ndarray,
    model: LocalOutlierFactor,
    is_anomaly: np.ndarray,
    feature_names: list[str],
    shap_indices: Optional[np.ndarray],
    cfg: Config,
) -> tuple[Optional[pd.DataFrame], np.ndarray, np.ndarray]:
    """LOF用距離ベース特徴量寄与度分析。

    各レコードと近傍の二乗距離を特徴量ごとに分解し、
    LOFスコアへの寄与度（合計1の比率）を算出する。

    計算対象の優先順位:
      1. shap_indices が指定されていればそのインデックス（label_col==1 など）
      2. cfg.shap_all=True なら全件
      3. それ以外は is_anomaly=1 のレコードのみ

    Args:
        X_input: LOFに入力した特徴量行列（重みづけ・標準化済み）
        model: 学習済みLocalOutlierFactorモデル
        is_anomaly: 異常フラグ配列
        feature_names: 特徴量名リスト
        shap_indices: 計算対象インデックス（Noneの場合は自動決定）
        cfg: 実行設定

    Returns:
        (contrib_df or None, top_feature_arr, top_contrib_arr)
        contrib_df: 各レコードの特徴量別寄与比率 (行=対象レコード, 列=特徴量)
    """
    n = len(X_input)
    top_feature_arr = np.full(n, "", dtype=object)
    top_contrib_arr = np.zeros(n)

    # 計算対象インデックスの決定
    if cfg.shap_all:
        target_indices = np.arange(n)
        logger.info(f"LOF寄与度: 全件計算 ({n}件)")
    elif shap_indices is not None:
        target_indices = shap_indices
        logger.info(
            f"LOF寄与度: ラベル付き異常レコードのみ計算 ({len(target_indices)} / {n}件)"
        )
    else:
        target_indices = np.where(is_anomaly == 1)[0]
        logger.info(
            f"LOF寄与度: 異常レコードのみ計算 ({len(target_indices)} / {n}件)"
        )

    if len(target_indices) == 0:
        logger.warning("LOF寄与度計算対象レコードがありません。")
        return None, top_feature_arr, top_contrib_arr

    try:
        # LOFの内部kNNから近傍インデックスを取得（自己参照を除く）
        n_neighbors = model.n_neighbors_
        _, nbr_indices_all = model.kneighbors(
            X_input, n_neighbors=n_neighbors + 1
        )
        nbr_indices_all = nbr_indices_all[:, 1:]  # 先頭は自己参照 → 除外

        contributions = []
        for i in target_indices:
            nbr_X = X_input[nbr_indices_all[i]]        # (k, p)
            diffs_sq = (X_input[i] - nbr_X) ** 2       # (k, p) 二乗距離
            feat_contrib = diffs_sq.mean(axis=0)        # (p,) 近傍平均
            total = float(feat_contrib.sum())
            if total > 0:
                feat_contrib_norm = feat_contrib / total  # 合計=1 に正規化
            else:
                feat_contrib_norm = np.ones(len(feature_names)) / len(feature_names)

            contributions.append(feat_contrib_norm)
            top_j = int(np.argmax(feat_contrib_norm))
            top_feature_arr[i] = feature_names[top_j]
            top_contrib_arr[i] = float(feat_contrib_norm[top_j])

        contrib_df = pd.DataFrame(
            contributions, columns=feature_names, index=target_indices
        )

        # 特徴量別 平均寄与度ランキング（上位10件をINFO出力）
        mean_contrib = contrib_df.mean(axis=0).sort_values(ascending=False)
        logger.info("LOF寄与度ランキング (上位10):")
        for rank, (feat, val) in enumerate(mean_contrib.head(10).items(), 1):
            logger.info(f"  {rank:2d}. {feat}: {val:.4f}")

        logger.info("LOF寄与度計算完了")
        return contrib_df, top_feature_arr, top_contrib_arr

    except Exception as e:
        logger.warning(f"LOF寄与度計算エラーのためスキップ: {e}")
        logger.debug(traceback.format_exc())
        return None, top_feature_arr, top_contrib_arr


# ============================================================
# PCA / MCA
# ============================================================

def _run_mca_first_component(X_ohe: np.ndarray) -> np.ndarray:
    """OHEインジケータ行列に対応分析(MCA)を適用し、第1成分スコアを返す。

    指示行列の対応分析（Correspondence Analysis）を利用。
    各行は元カテゴリカラム数だけ 1 を持つ 0/1 行列を想定。

    Args:
        X_ohe: OHEインジケータ行列 (n×p)、値は 0 または 1

    Returns:
        第1成分の行スコア配列 (n,)
    """
    X = X_ohe.astype(float)
    N = X.sum()
    if N == 0:
        return np.zeros(X.shape[0])

    Z = X / N                                   # 比率行列
    r = Z.sum(axis=1)                           # 行マス (n,)
    c = Z.sum(axis=0)                           # 列マス (p,)

    # ゼロ除算回避
    r_inv_sqrt = np.where(r > 0, 1.0 / np.sqrt(r), 0.0)
    c_inv_sqrt = np.where(c > 0, 1.0 / np.sqrt(c), 0.0)

    # 残差行列（独立からの偏差をスケーリング）
    S = r_inv_sqrt[:, np.newaxis] * (Z - np.outer(r, c)) * c_inv_sqrt[np.newaxis, :]

    n_comp = min(2, min(S.shape) - 1)
    if n_comp < 1:
        return np.zeros(X.shape[0])

    U, sigma, _ = randomized_svd(S, n_components=n_comp, random_state=42)

    # 行座標（第1成分）
    scores = r_inv_sqrt * U[:, 0] * sigma[0]
    return scores


@timed("PCA/MCA")
def run_pca_mca(
    X_scaled: np.ndarray,
    feature_df: pd.DataFrame,
    feature_names: list[str],
    encode_summaries: list[dict],
    cfg: Config,
) -> tuple[np.ndarray, dict]:
    """数値カラム→PCA第1軸、OHEカラム→MCA第1軸 のハイブリッド2次元座標を返す。

    カラム種別ごとの処理:
      - 数値 & OHE 両方あり: DIM1=PCA-PC1(数値), DIM2=MCA-DIM1(OHE)
      - 数値のみ: 標準PCA 2成分
      - OHEのみ: MCA 第1成分 + ゼロ埋め第2成分

    MCA に使うカラムは特徴量選択を通過したものだけ（feature_names 内）。

    Args:
        X_scaled: 標準化済み特徴量行列
        feature_df: 特徴量選択後の未スケールDataFrame（OHEの0/1値を取得するため）
        feature_names: 前処理・特徴量選択後の特徴量名リスト
        encode_summaries: preprocess() が返すエンコードサマリ
        cfg: 実行設定

    Returns:
        (coords [n×2], dim_info dict)
        dim_info には "dim1" / "dim2" キーで各軸の説明文字列を格納。
    """
    # 特徴量選択後に残っているOHEカラムを特定
    ohe_cols_in_features: set[str] = set()
    for s in encode_summaries:
        if s.get("encoding") == "ohe":
            for col in s.get("generated_columns", "").split(","):
                col = col.strip()
                if col and col in feature_names:
                    ohe_cols_in_features.add(col)

    numeric_col_list = [c for c in feature_names if c not in ohe_cols_in_features]
    ohe_col_list     = [c for c in feature_names if c in ohe_cols_in_features]

    has_numeric = len(numeric_col_list) > 0
    has_ohe     = len(ohe_col_list) > 0

    logger.info(
        f"PCA/MCA: 数値カラム={len(numeric_col_list)}件, "
        f"OHEカラム={len(ohe_col_list)}件"
    )
    logger.debug(f"  数値カラム: {numeric_col_list}")
    logger.debug(f"  OHEカラム: {ohe_col_list}")

    feat_idx = {f: i for i, f in enumerate(feature_names)}

    if has_numeric and has_ohe:
        # --- DIM1: 数値カラムのPCA第1成分 ---
        num_indices = [feat_idx[c] for c in numeric_col_list]
        X_num = X_scaled[:, num_indices]
        pca = PCA(n_components=1, random_state=cfg.random_state)
        axis1 = pca.fit_transform(X_num)[:, 0]
        var1 = float(pca.explained_variance_ratio_[0])

        # --- DIM2: OHEカラムのMCA第1成分（未スケール0/1値を使用）---
        X_ohe = feature_df[ohe_col_list].values
        axis2 = _run_mca_first_component(X_ohe)

        coords = np.column_stack([axis1, axis2])
        dim_info = {
            "dim1": f"PCA-PC1 (数値 {len(numeric_col_list)}列, var={var1:.3f})",
            "dim2": f"MCA-DIM1 (OHE {len(ohe_col_list)}列)",
        }
        logger.info(
            f"DIM1=PCA-PC1 (数値 {len(numeric_col_list)}列, 寄与率={var1:.3f}), "
            f"DIM2=MCA-DIM1 (OHE {len(ohe_col_list)}列)"
        )

    elif has_numeric:
        # 数値カラムのみ → 標準PCA 2成分
        num_indices = [feat_idx[c] for c in numeric_col_list]
        X_num = X_scaled[:, num_indices]
        n_comp = min(2, X_num.shape[1])
        pca = PCA(n_components=n_comp, random_state=cfg.random_state)
        raw = pca.fit_transform(X_num)
        vr = pca.explained_variance_ratio_
        cumulative = float(np.sum(vr))
        if cumulative < cfg.pca_variance_warning:
            logger.warning(
                f"PCA累積寄与率 ({cumulative:.3f}) が "
                f"警告閾値 ({cfg.pca_variance_warning}) を下回っています。"
            )
        if n_comp == 1:
            raw = np.hstack([raw, np.zeros((len(raw), 1))])
            vr = np.append(vr, 0.0)
        coords = raw
        dim_info = {
            "dim1": f"PCA-PC1 (数値 {len(numeric_col_list)}列, var={vr[0]:.3f})",
            "dim2": f"PCA-PC2 (数値 {len(numeric_col_list)}列, var={vr[1]:.3f})",
        }
        logger.info(
            f"DIM1=PCA-PC1 (var={vr[0]:.3f}), DIM2=PCA-PC2 (var={vr[1]:.3f}), "
            f"累積={cumulative:.3f}"
        )

    elif has_ohe:
        # OHEカラムのみ → MCA 第1成分 + ゼロ埋め
        X_ohe = feature_df[ohe_col_list].values
        axis1 = _run_mca_first_component(X_ohe)
        coords = np.column_stack([axis1, np.zeros(len(axis1))])
        dim_info = {
            "dim1": f"MCA-DIM1 (OHE {len(ohe_col_list)}列)",
            "dim2": "zeros (OHEのみのためMCA第2成分省略)",
        }
        logger.info(f"DIM1=MCA-DIM1 (OHE {len(ohe_col_list)}列), DIM2=zeros")

    else:
        coords = np.zeros((X_scaled.shape[0], 2))
        dim_info = {"dim1": "zeros", "dim2": "zeros"}
        logger.warning("PCA/MCA: 有効なカラムがありません。ゼロ座標を使用します。")

    logger.debug(
        f"DIM1スコア範囲: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}]"
    )
    logger.debug(
        f"DIM2スコア範囲: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]"
    )

    return coords, dim_info


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
    pca_dim_info: dict          # DIM1/DIM2 の説明（PCA寄与率 or MCA軸情報）


@timed("結果出力")
def save_outputs(
    original_df: pd.DataFrame,
    results: ModelResults,
    excluded_cols: list[str],
    encode_summaries: list[dict],
    cfg: Config,
    column_groups: Optional[dict] = None,
) -> None:
    """結果ファイル一式を out/ フォルダに保存する。

    Args:
        original_df: 元の入力DataFrame（変換前）
        results: モデリング結果
        excluded_cols: 前処理・特徴量選択で除外したカラム名リスト
        encode_summaries: カテゴリ変数のエンコードサマリ
        cfg: 実行設定
        column_groups: build_column_groups() の出力（元カラム → 展開後特徴量リスト）
    """
    cfg.out_dir.mkdir(exist_ok=True)

    # ---- result.csv ----
    result_df = original_df.copy()
    result_df["anomaly_score"] = results.anomaly_score

    # 相対スコア: 最低=0.0, 最高=1.0, 中央値≒0.5（順位ベース正規化）
    n_rec = len(results.anomaly_score)
    if n_rec > 1:
        ranks = pd.Series(results.anomaly_score).rank(method="average") - 1
        result_df["anomaly_score_relative"] = (ranks / (n_rec - 1)).clip(0.0, 1.0).values
    else:
        result_df["anomaly_score_relative"] = np.zeros(n_rec)

    result_df["is_anomaly"] = results.is_anomaly
    result_df["top_feature"] = results.top_feature_arr
    result_df["top_shap_value"] = results.top_shap_value_arr

    # 元カラム名への逆引きマップ（OHE展開後 → 元カラム名）
    feat_to_orig: dict[str, str] = {}
    if column_groups:
        for orig, feats in column_groups.items():
            for f in feats:
                feat_to_orig[f] = orig
    result_df["top_feature_original"] = [
        feat_to_orig.get(f, f) for f in results.top_feature_arr
    ]
    out_path = cfg.out_dir / "result.csv"
    result_df.to_csv(out_path, index=True, index_label="index", encoding="utf-8-sig")
    logger.info(
        f"出力: result.csv ({len(result_df)}件, {len(result_df.columns)}カラム)"
    )
    logger.debug(f"  → {out_path.resolve()}")

    # ---- feature_contribution.csv（SHAP値 or LOF距離寄与度）----
    if results.shap_df is not None:
        out_path = cfg.out_dir / "feature_contribution.csv"
        results.shap_df.to_csv(out_path, encoding="utf-8-sig")
        logger.info(f"出力: feature_contribution.csv ({len(results.shap_df)}件)")
        logger.debug(f"  → {out_path.resolve()}")

        # ---- feature_contribution_original.csv（元カラム単位に集約）----
        if column_groups:
            orig_rows: dict[str, pd.Series] = {}
            for orig, feats in column_groups.items():
                cols_in_df = [f for f in feats if f in results.shap_df.columns]
                if cols_in_df:
                    orig_rows[orig] = results.shap_df[cols_in_df].sum(axis=1)
            if orig_rows:
                orig_contrib_df = pd.DataFrame(orig_rows, index=results.shap_df.index)
                out_path = cfg.out_dir / "feature_contribution_original.csv"
                orig_contrib_df.to_csv(out_path, encoding="utf-8-sig")
                logger.info(
                    f"出力: feature_contribution_original.csv "
                    f"({len(orig_contrib_df)}件, {len(orig_contrib_df.columns)}カラム)"
                )
    else:
        logger.info("特徴量寄与度出力スキップ")

    # ---- pca_2d.csv（DIM1, DIM2, 異常情報, 元データ全列）----
    pca_df = pd.DataFrame({
        "DIM1": results.pca_coords[:, 0],
        "DIM2": results.pca_coords[:, 1],
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

    # ---- dim_info.csv（各次元の説明）----
    dim_df = pd.DataFrame([
        {"dimension": "DIM1", "description": results.pca_dim_info.get("dim1", "")},
        {"dimension": "DIM2", "description": results.pca_dim_info.get("dim2", "")},
    ])
    out_path = cfg.out_dir / "dim_info.csv"
    dim_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("出力: dim_info.csv")

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
            "coverage_top_n", "other_count", "generated_columns", "prefix_n",
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
        description="異常検知システム (Isolation Forest / LOF + SHAP + PCA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- パス ---
    parser.add_argument("--in-dir", type=Path, default=Path("in"),
                        help="入力CSVフォルダのパス")
    parser.add_argument("--out-dir", type=Path, default=Path("out"),
                        help="出力フォルダのパス")
    parser.add_argument("--model-dir", type=Path, default=Path("models"),
                        help="チューニング結果pkl の保存・読み込みフォルダ (デフォルト: models)")
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
                        help="ラベル列名（パーセンタイル表示・チューニングに使用）")
    parser.add_argument("--label-anomaly-value", type=int, default=1,
                        help="異常を示すラベル値")
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=False,
                        help="Optunaチューニングを実行（--label-col 必須）")
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

    # --- モデル手法 ---
    parser.add_argument("--method", type=str, choices=["if", "lof"], default="if",
                        help="異常検知手法: if=Isolation Forest, lof=Local Outlier Factor")
    parser.add_argument("--lof-n-neighbors", type=int, default=20,
                        help="LOFの近傍数 (--method lof 時に使用)")
    parser.add_argument("--lof-tune-weights", action=argparse.BooleanOptionalAction, default=True,
                        help="LOF+Optuna時に特徴量重みをチューニングする。--no-lof-tune-weights で無効化")

    # --- カラム設定 ---
    parser.add_argument("--column-config", type=Path, default=Path("column_config.json"),
                        help="カラム設定JSONファイルのパス (デフォルト: column_config.json)")

    return parser.parse_args()


def main() -> None:
    """異常検知システムのメイン処理を実行する。"""
    args = _parse_args()

    cfg = Config(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
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
        tune=args.tune,
        n_trials=args.n_trials,
        method=args.method,
        lof_n_neighbors=args.lof_n_neighbors,
        lof_tune_weights=args.lof_tune_weights,
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
        feature_df, excluded_cols = select_features(
            processed_df, excluded_cols, cfg, encode_summaries
        )
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

        # 元カラム → 展開後特徴量のグループマップを構築
        column_groups = build_column_groups(feature_names, encode_summaries)
        logger.info(
            f"特徴量グループ: {len(column_groups)}元カラム / {len(feature_names)}特徴量"
        )

        # 5. チューニング / pkl ロード
        lof_weights: Optional[np.ndarray] = None
        pkl_path = cfg.model_dir / "best_params.pkl"

        if cfg.tune:
            if cfg.label_col is None:
                raise AnomalyDetectionError(
                    "--tune を使用するには --label-col でラベル列名を指定してください。"
                )
            cfg, lof_weights = tune_hyperparams(
                X_scaled, original_df, cfg, feature_names, column_groups
            )
            save_best_params(cfg, lof_weights)
        elif pkl_path.exists():
            logger.info(f"pklファイルを検出。チューニング結果をロードします: {pkl_path}")
            cfg, lof_weights = load_best_params(cfg)
        else:
            logger.info(
                f"チューニングなし・pklなし ({pkl_path}): デフォルトパラメータで実行します。"
            )

        # 6. 異常検知モデル（IF または LOF）
        if cfg.method == "lof":
            model, anomaly_score, is_anomaly = run_lof(X_scaled, cfg, weights=lof_weights)
        else:
            model, anomaly_score, is_anomaly = run_isolation_forest(X_scaled, cfg)

        # ラベル付き異常のパーセンタイル確認（--label-col 指定時のみ）
        if cfg.label_col is not None:
            labeled_idx = np.where(
                original_df[cfg.label_col].values == cfg.label_anomaly_value
            )[0]
            _log_labeled_percentile(anomaly_score, labeled_idx, "最終モデル")

        # 7. SHAP (IF) / 距離ベース寄与度 (LOF)
        # デフォルト対象: label_col指定時はlabel_col==label_anomaly_valueのレコード
        if not cfg.shap_all and cfg.label_col is not None:
            shap_indices = np.where(
                original_df[cfg.label_col].values == cfg.label_anomaly_value
            )[0]
        else:
            shap_indices = None

        if cfg.method == "lof":
            # LOF: 近傍との二乗距離を特徴量ごとに分解して寄与度を算出
            X_lof_input = X_scaled * lof_weights if lof_weights is not None else X_scaled
            shap_df, top_feature_arr, top_shap_value_arr = run_lof_contributions(
                X_lof_input, model, is_anomaly, feature_names, shap_indices, cfg
            )
        else:
            # IF: TreeExplainer による SHAP 値
            shap_df, top_feature_arr, top_shap_value_arr = run_shap(
                model, X_scaled, is_anomaly, feature_names, cfg,
                shap_indices=shap_indices,
            )

        # 8. PCA/MCA ハイブリッド（数値軸=PCA, OHE軸=MCA）
        pca_coords, pca_dim_info = run_pca_mca(
            X_scaled, feature_df, feature_names, encode_summaries, cfg
        )

        # 9. 出力
        results = ModelResults(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            shap_df=shap_df,
            top_feature_arr=top_feature_arr,
            top_shap_value_arr=top_shap_value_arr,
            pca_coords=pca_coords,
            pca_dim_info=pca_dim_info,
        )
        save_outputs(original_df, results, excluded_cols, encode_summaries, cfg, column_groups)

        # 最終サマリ：flag付きデータの異常スコア位置
        if cfg.label_col is not None:
            labeled_idx = np.where(
                original_df[cfg.label_col].values == cfg.label_anomaly_value
            )[0]
            logger.info("=" * 60)
            _log_labeled_percentile(anomaly_score, labeled_idx, "最終サマリ")
            logger.info("=" * 60)

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
