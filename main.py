"""
異常検知システム: Isolation Forest + SHAP + PCA可視化 (Tableau連携)
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import chardet
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── 設定パラメータ ──────────────────────────────────────────
CONTAMINATION = 0.05          # 異常割合の想定値
N_ESTIMATORS = 200            # Isolation Forestの木の本数
VARIANCE_THRESHOLD = 0.01     # 低分散カラム除外の閾値
CORR_THRESHOLD = 0.95         # 高相関カラム除外の閾値
SHAP_ALL = False              # True: 全件SHAP計算 / False: 異常レコードのみ
PCA_VARIANCE_WARNING = 0.50   # 累積寄与率の警告閾値
MAX_FEATURES = 1.0            # 各木で使用する特徴量の割合
RANDOM_STATE = 42             # 再現性のための乱数シード

# ── パス設定 ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
IN_DIR = BASE_DIR / "in"
OUT_DIR = BASE_DIR / "out"

# ── ログ設定 ────────────────────────────────────────────────
OUT_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("anomaly_detection")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

file_handler = logging.FileHandler(OUT_DIR / "processing.log", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# ── 1. CSV読み込み ──────────────────────────────────────────
def detect_encoding(filepath: Path) -> str:
    """ファイルの文字コードを自動判定する。"""
    with open(filepath, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]


def load_csv_files(in_dir: Path) -> pd.DataFrame:
    """in/ フォルダ内の全CSVを読み込み・結合する。"""
    if not in_dir.exists():
        logger.error(f"入力フォルダが存在しません: {in_dir}")
        sys.exit(1)

    csv_files = sorted(in_dir.glob("*.csv"))
    if len(csv_files) == 0:
        logger.error(f"入力フォルダにCSVファイルがありません: {in_dir}")
        sys.exit(1)

    frames = []
    for fp in csv_files:
        enc = detect_encoding(fp)
        logger.info(f"読み込み: {fp.name}  (encoding={enc})")
        df = pd.read_csv(fp, encoding=enc)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"読み込み完了: {len(csv_files)} ファイル, {len(combined)} 行, {len(combined.columns)} カラム")
    return combined


# ── 2. 前処理 ──────────────────────────────────────────────
def try_parse_date(series: pd.Series) -> bool:
    """シリーズが日付としてパースできるか判定する。"""
    try:
        parsed = pd.to_datetime(series.dropna().head(100), infer_datetime_format=True)
        return parsed.notna().mean() > 0.8
    except (ValueError, TypeError):
        return False


def expand_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """日付カラムを年/月/日/時/曜日に展開する。"""
    dt = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_year"] = dt.dt.year
    df[f"{col}_month"] = dt.dt.month
    df[f"{col}_day"] = dt.dt.day
    df[f"{col}_hour"] = dt.dt.hour
    df[f"{col}_dayofweek"] = dt.dt.dayofweek
    df.drop(columns=[col], inplace=True)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    自動型判定・前処理を行い、(処理済み特徴量DF, 元データDF, 除外カラムリスト) を返す。
    """
    excluded_columns: list[str] = []
    original_df = df.copy()
    n_rows = len(df)

    # --- 欠損率50%超のカラムを除外 ---
    missing_ratio = df.isnull().mean()
    high_missing = missing_ratio[missing_ratio > 0.50].index.tolist()
    if high_missing:
        logger.warning(f"欠損率50%超で除外: {high_missing}")
        excluded_columns.extend(high_missing)
        df.drop(columns=high_missing, inplace=True)

    # --- 自動型判定・変換 ---
    for col in list(df.columns):
        dtype = df[col].dtype

        # bool型 → 0/1
        if dtype == bool or df[col].dropna().isin([True, False]).all():
            df[col] = df[col].astype(int)
            continue

        # 数値型はそのまま
        if pd.api.types.is_numeric_dtype(dtype):
            continue

        # object型: 日付 → datetime展開
        if dtype == object and try_parse_date(df[col]):
            logger.info(f"日付カラム検出・展開: {col}")
            df = expand_datetime(df, col)
            continue

        # object型: カテゴリ判定
        if dtype == object:
            n_unique = df[col].nunique()
            ratio = n_unique / n_rows if n_rows > 0 else 0

            if ratio <= 0.50:
                # Label Encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                logger.info(f"Label Encoding: {col} (ユニーク数={n_unique})")
            else:
                # 頻度エンコーディング
                freq = df[col].value_counts(normalize=True)
                df[col] = df[col].map(freq).fillna(0)
                logger.info(f"頻度エンコーディング: {col} (ユニーク数={n_unique})")

    # --- 欠損値補完 ---
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "UNKNOWN", inplace=True)

    # --- 特徴量選択: 低分散除去 ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    variances = df[numeric_cols].var()
    low_var = variances[variances < VARIANCE_THRESHOLD].index.tolist()
    if low_var:
        logger.info(f"低分散で除外: {low_var}")
        excluded_columns.extend(low_var)
        df.drop(columns=low_var, inplace=True)
        numeric_cols = [c for c in numeric_cols if c not in low_var]

    # --- 特徴量選択: 高相関除去 ---
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
        if high_corr:
            logger.info(f"高相関で除外: {high_corr}")
            excluded_columns.extend(high_corr)
            df.drop(columns=high_corr, inplace=True)

    # 数値カラムのみ残す
    df = df.select_dtypes(include=[np.number])

    if df.shape[1] == 0:
        logger.error("前処理後に有効なカラムが0件です。除外理由をログで確認してください。")
        sys.exit(1)

    logger.info(f"前処理完了: {df.shape[1]} 特徴量, {df.shape[0]} 行")
    return df, original_df, excluded_columns


# ── 3. Isolation Forest ───────────────────────────────────
def run_isolation_forest(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isolation Forest を実行し (anomaly_score, is_anomaly) を返す。"""
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
    )
    model.fit(X_scaled)

    # スコア: score_samples()をマイナス反転（高いほど異常）
    anomaly_score = -model.score_samples(X_scaled)

    # ラベル: -1(異常)→1, 1(正常)→0
    raw_pred = model.predict(X_scaled)
    is_anomaly = (raw_pred == -1).astype(int)

    logger.info(f"Isolation Forest完了: 異常={is_anomaly.sum()} 件 / 全体={len(is_anomaly)} 件")
    return anomaly_score, is_anomaly, model


# ── 4. SHAP 分析 ──────────────────────────────────────────
def run_shap_analysis(
    model: IsolationForest,
    X_scaled: np.ndarray,
    feature_names: list[str],
    is_anomaly: np.ndarray,
) -> tuple[pd.DataFrame | None, np.ndarray | None, list[str] | None]:
    """
    SHAP値を算出し、(shap_summary_df, shap_values_full, top_features) を返す。
    エラー時は (None, None, None) を返す。
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)

        if SHAP_ALL:
            idx = np.arange(len(X_scaled))
        else:
            idx = np.where(is_anomaly == 1)[0]

        if len(idx) == 0:
            logger.warning("異常レコードが0件のため、SHAP計算をスキップします。")
            return None, None, None

        logger.info(f"SHAP計算開始: {len(idx)} 件")
        shap_values = explainer.shap_values(X_scaled[idx])

        # shap_summary_df: 対象レコード × 特徴量
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.insert(0, "row_index", idx)

        # 全レコード用の top_feature / top_shap_value
        top_features_all = [None] * len(X_scaled)
        top_shap_values_all = [None] * len(X_scaled)

        abs_shap = np.abs(shap_values)
        for i, row_idx in enumerate(idx):
            max_col = int(abs_shap[i].argmax())
            top_features_all[row_idx] = feature_names[max_col]
            top_shap_values_all[row_idx] = float(abs_shap[i, max_col])

        logger.info("SHAP計算完了")
        return shap_df, top_features_all, top_shap_values_all

    except Exception as e:
        logger.warning(f"SHAP計算エラー（スキップ）: {e}")
        return None, None, None


# ── 5. PCA ────────────────────────────────────────────────
def run_pca(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA 2次元射影を実行し、(座標, 寄与率配列) を返す。"""
    n_components = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    variance_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance_ratio)
    logger.info(f"PCA寄与率: {variance_ratio}  累積: {cumulative}")

    if len(cumulative) >= 2 and cumulative[1] < PCA_VARIANCE_WARNING:
        logger.warning(
            f"PCA累積寄与率が{PCA_VARIANCE_WARNING:.0%}未満です "
            f"(累積={cumulative[1]:.2%})。可視化の精度に注意してください。"
        )

    # 1次元の場合はPC2を0埋め
    if n_components == 1:
        coords = np.column_stack([coords, np.zeros(len(coords))])
        variance_ratio = np.append(variance_ratio, 0.0)

    return coords, variance_ratio


# ── 6. 出力 ───────────────────────────────────────────────
def save_results(
    original_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    anomaly_score: np.ndarray,
    is_anomaly: np.ndarray,
    top_features: list | None,
    top_shap_values: list | None,
    shap_df: pd.DataFrame | None,
    pca_coords: np.ndarray,
    variance_ratio: np.ndarray,
    excluded_columns: list[str],
) -> None:
    """結果ファイルを out/ に出力する。"""
    OUT_DIR.mkdir(exist_ok=True)

    # --- result.csv ---
    result = original_df.copy()
    result["anomaly_score"] = anomaly_score
    result["is_anomaly"] = is_anomaly
    if top_features is not None:
        result["top_feature"] = top_features
    if top_shap_values is not None:
        result["top_shap_value"] = top_shap_values
    result.to_csv(OUT_DIR / "result.csv", index=False, encoding="utf-8-sig")
    logger.info(f"出力: result.csv ({len(result)} 行)")

    # --- shap_summary.csv ---
    if shap_df is not None:
        shap_df.to_csv(OUT_DIR / "shap_summary.csv", index=False, encoding="utf-8-sig")
        logger.info(f"出力: shap_summary.csv ({len(shap_df)} 行)")

    # --- pca_2d.csv ---
    pca_df = original_df.copy()
    pca_df.insert(0, "PC1", pca_coords[:, 0])
    pca_df.insert(1, "PC2", pca_coords[:, 1])
    pca_df["anomaly_score"] = anomaly_score
    pca_df["is_anomaly"] = is_anomaly
    if top_features is not None:
        pca_df["top_feature"] = top_features
    pca_df.to_csv(OUT_DIR / "pca_2d.csv", index=False, encoding="utf-8-sig")
    logger.info(f"出力: pca_2d.csv ({len(pca_df)} 行)")

    # --- pca_variance.csv ---
    n_components = len(variance_ratio)
    var_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(n_components)],
        "explained_variance_ratio": variance_ratio,
        "cumulative_variance_ratio": np.cumsum(variance_ratio),
    })
    var_df.to_csv(OUT_DIR / "pca_variance.csv", index=False, encoding="utf-8-sig")
    logger.info("出力: pca_variance.csv")

    # --- excluded_columns.txt ---
    with open(OUT_DIR / "excluded_columns.txt", "w", encoding="utf-8") as f:
        for col in excluded_columns:
            f.write(col + "\n")
    logger.info(f"出力: excluded_columns.txt ({len(excluded_columns)} カラム)")


# ── メイン ─────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 60)
    logger.info("異常検知システム 開始")
    logger.info("=" * 60)

    # 1. CSV読み込み
    raw_df = load_csv_files(IN_DIR)

    # 2. 前処理
    feature_df, original_df, excluded_columns = preprocess(raw_df)

    # 3. スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)
    feature_names = feature_df.columns.tolist()

    # 4. Isolation Forest
    anomaly_score, is_anomaly, model = run_isolation_forest(X_scaled)

    # 5. SHAP
    shap_df, top_features, top_shap_values = run_shap_analysis(
        model, X_scaled, feature_names, is_anomaly
    )

    # 6. PCA
    pca_coords, variance_ratio = run_pca(X_scaled)

    # 7. 出力
    save_results(
        original_df, feature_df, anomaly_score, is_anomaly,
        top_features, top_shap_values, shap_df,
        pca_coords, variance_ratio, excluded_columns,
    )

    logger.info("=" * 60)
    logger.info("異常検知システム 完了")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
