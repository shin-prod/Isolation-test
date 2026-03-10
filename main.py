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
    random_state: int = 42

    # --- 前処理 ---
    missing_rate_threshold: float = 0.50    # 欠損率除外閾値
    date_parse_threshold: float = 0.80      # 日付判定の成功率閾値
    ohe_top_n: int = 10                     # OHE対象とする上位カテゴリ数
    ohe_coverage_threshold: float = 0.50    # 上位N件のカバレッジ閾値（超えたらOHE）
    variance_threshold: float = 0.01        # 低分散除外閾値
    corr_threshold: float = 0.95            # 高相関除外閾値

    # --- SHAP ---
    shap_all: bool = True                   # True: 全件 / False: 異常レコードのみ

    # --- PCA ---
    pca_variance_warning: float = 0.50      # 累積寄与率の警告閾値

    # --- パス ---
    in_dir: Path = field(default_factory=lambda: Path("in"))
    out_dir: Path = field(default_factory=lambda: Path("out"))


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
            df = pd.read_csv(csv_file, encoding="utf-8")
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
    df: pd.DataFrame, col: str, n_rows: int, cfg: Config
) -> pd.DataFrame:
    """単一の object 型カラムを型判定してエンコードする。

    優先順位:
      1. 日付パース成功率 >= date_parse_threshold
             → datetime 分解（年/月/日/時/曜日）
      2. 上位 ohe_top_n カテゴリのカバレッジ >= ohe_coverage_threshold
             → 上位 N カテゴリを One-Hot Encoding、残りを "__other__"
      3. それ以外（高カーディナリティ）
             → 頻度エンコーディング

    Args:
        df: 処理対象DataFrame
        col: 対象カラム名
        n_rows: 全レコード数
        cfg: 実行設定

    Returns:
        エンコード処理後のDataFrame
    """
    # ---- 日付判定 ----
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
        for part, values in date_parts.items():
            new_col = f"{col}_{part}"
            df[new_col] = values
            logger.debug(f"    生成: {new_col}")
        df = df.drop(columns=[col])
        return df

    # ---- 欠損補完（最頻値）----
    n_null = int(df[col].isnull().sum())
    mode_vals = df[col].mode()
    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
    if n_null > 0:
        df[col] = df[col].fillna(fill_val)
        logger.debug(f"  欠損補完 [{col}]: {n_null}件 → '{fill_val}'")

    # ---- カバレッジ判定 ----
    freq = df[col].value_counts(normalize=True)
    top_n_cats = freq.nlargest(cfg.ohe_top_n).index.tolist()
    coverage = float(freq.nlargest(cfg.ohe_top_n).sum())
    logger.debug(
        f"  カバレッジ [{col}]: top{cfg.ohe_top_n}={coverage:.3f} "
        f"(閾値={cfg.ohe_coverage_threshold}), unique={df[col].nunique()}"
    )
    logger.debug(f"  top{cfg.ohe_top_n} カテゴリ: {top_n_cats}")

    if coverage >= cfg.ohe_coverage_threshold:
        # 上位N件以外を "__other__" に置き換えてから OHE
        other_count = int((~df[col].isin(top_n_cats)).sum())
        df[col] = df[col].where(df[col].isin(top_n_cats), other="__other__")
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
        logger.info(
            f"  OHE [{col}]: top{cfg.ohe_top_n}カテゴリ + __other__({other_count}件) "
            f"→ {len(dummies.columns)}列生成"
        )
        logger.debug(f"  生成列: {dummies.columns.tolist()}")
    else:
        # 高カーディナリティ → 頻度エンコーディング
        df[col] = df[col].map(freq)
        logger.info(
            f"  高カーディナリティ → 頻度エンコード: {col} "
            f"(coverage={coverage:.3f} < {cfg.ohe_coverage_threshold})"
        )
        logger.debug(f"  頻度 top5:\n{freq.head().to_string()}")

    return df


@timed("前処理")
def preprocess(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    """自動型判定・欠損補完・エンコーディングを行う前処理。

    Args:
        df: 入力DataFrame
        cfg: 実行設定

    Returns:
        (処理済みDataFrame, 除外カラムリスト)
    """
    excluded_cols: list[str] = []
    n_rows = len(df)
    logger.debug(f"前処理開始: {df.shape}")

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
    logger.debug(
        f"型判定対象カラム ({len(cols_to_process)}件): {cols_to_process}"
    )

    for col in cols_to_process:
        if col not in df.columns:
            continue  # 日付分解で削除済み

        dtype = df[col].dtype
        logger.debug(f"カラム処理: {col} (dtype={dtype})")

        if dtype == bool:
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

        elif dtype == object:
            df = _encode_column(df, col, n_rows, cfg)

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
        f"(除外済み累計: {len(excluded_cols)}件)"
    )
    _log_df_summary(df, "前処理後")
    return df, excluded_cols


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
        f"random_state={cfg.random_state}"
    )
    logger.debug(f"入力行列: shape={X_scaled.shape}")

    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        max_features=cfg.max_features,
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
    cfg: Config,
) -> None:
    """結果ファイル一式を out/ フォルダに保存する。

    Args:
        original_df: 元の入力DataFrame（変換前）
        results: モデリング結果
        excluded_cols: 前処理・特徴量選択で除外したカラム名リスト
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
    pca_df.to_csv(out_path, index=False, encoding="utf-8-sig")
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


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    """異常検知システムのメイン処理を実行する。"""
    cfg = Config()
    setup_logging(cfg.out_dir)

    logger.info("=" * 60)
    logger.info("異常検知システム 開始")
    logger.info(
        f"設定: contamination={cfg.contamination}, "
        f"n_estimators={cfg.n_estimators}, "
        f"shap_all={cfg.shap_all}"
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()

    try:
        # 1. データ読み込み
        original_df = load_csvs(cfg)

        # 2. 前処理
        processed_df, excluded_cols = preprocess(original_df.copy(), cfg)
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

        # 5. Isolation Forest
        model, anomaly_score, is_anomaly = run_isolation_forest(X_scaled, cfg)

        # 6. SHAP
        shap_df, top_feature_arr, top_shap_value_arr = run_shap(
            model, X_scaled, is_anomaly, feature_names, cfg
        )

        # 7. PCA
        pca_coords, pca_variance = run_pca(X_scaled, cfg)

        # 8. 出力
        results = ModelResults(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            shap_df=shap_df,
            top_feature_arr=top_feature_arr,
            top_shap_value_arr=top_shap_value_arr,
            pca_coords=pca_coords,
            pca_variance=pca_variance,
        )
        save_outputs(original_df, results, excluded_cols, cfg)

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
