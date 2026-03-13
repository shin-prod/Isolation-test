# 異常検知システム

**Isolation Forest / LOF + SHAP + PCA可視化（Tableau連携）**

CSVデータを入力するだけで、機械学習による異常検知・原因分析・可視化データ生成を自動実行します。

---

## 概要

| 項目 | 内容 |
|------|------|
| 入力 | `in/` フォルダのCSVファイル（Shift-JIS/cp932、複数可） |
| 出力 | `out/` フォルダに結果CSV一式を自動生成 |
| 異常検知 | Isolation Forest または Local Outlier Factor（切り替え可） |
| 原因分析 | SHAP（TreeExplainer、IFのみ） |
| 可視化用途 | PCA 2次元座標 → Tableau連携 |
| チューニング | Optuna によるハイパーパラメータ自動最適化 |

---

## ディレクトリ構成

```
isolation_test/
├── in/                          # 入力データ（CSVをここに置く）
├── out/                         # 出力フォルダ（自動生成）
│   ├── result.csv               # 異常スコア・フラグ付き結果
│   ├── shap_summary.csv         # SHAP値マトリクス（IFのみ）
│   ├── pca_2d.csv               # PCA座標 + 元データ（Tableau用メイン）
│   ├── pca_variance.csv         # 主成分の寄与率
│   ├── encoding_summary.csv     # カテゴリ変数のエンコード結果サマリ
│   ├── excluded_columns.txt     # 前処理で除外されたカラム一覧
│   ├── tuning_results.csv       # Optunaチューニング結果（--tune 時のみ）
│   └── processing.log           # 詳細処理ログ（DEBUG含む）
├── main.py                      # メインスクリプト
├── dedup.py                     # 重複削除スクリプト
├── check_coverage.py            # カテゴリカバレッジ確認スクリプト
├── column_config.json           # カラム設定（デフォルトで自動読み込み）
├── column_config.example.json   # カラム設定サンプル
├── run.sh                       # 実行シェルスクリプト
├── requirements.txt             # 依存パッケージ
└── README.md                    # 本ファイル
```

---

## セットアップ

### 前提条件

- Python 3.9 以上

### 手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/shin-prod/Isolation-test.git
cd Isolation-test

# 2. 仮想環境を作成・有効化
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. 依存パッケージをインストール
pip install -r requirements.txt

# 4. カラム設定を準備
cp column_config.example.json column_config.json
# column_config.json を編集して使用するカラムを設定

# 5. 入力データを配置
cp /path/to/your/data.csv in/

# 6. 実行
python main.py
```

---

## 実行方法

### 基本実行

```bash
python main.py
```

### LOF で実行

```bash
python main.py --method lof
```

### Optunaチューニングあり（LOF + 重みづけ）

```bash
python main.py --method lof --tune --label-col flag --n-trials 100
```

### チューニングあり（重みづけなし）

```bash
python main.py --method lof --tune --label-col flag --no-lof-tune-weights
```

---

## コマンドライン引数一覧

### 基本設定

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--in-dir` | `in` | 入力CSVフォルダのパス |
| `--out-dir` | `out` | 出力フォルダのパス |
| `--encoding` | `cp932` | 入力CSVの文字コード（cp932=Shift-JIS、utf-8 など） |
| `--column-config` | `column_config.json` | カラム設定JSONファイルのパス |

### モデル手法

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--method` | `if` | 異常検知手法: `if`=Isolation Forest / `lof`=Local Outlier Factor |

### Isolation Forest パラメータ

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--contamination` | 0.05 | 異常割合の想定値（0.0〜0.5） |
| `--n-estimators` | 200 | 木の本数 |
| `--max-samples` | 1.0 | 各木で使うサンプルの割合（0.0〜1.0） |

### LOF パラメータ

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--lof-n-neighbors` | 20 | 近傍数 |

### Optunaチューニング

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--tune` / `--no-tune` | False | Optunaチューニングを実行する |
| `--label-col` | なし | ラベル列名（`--tune` に必須） |
| `--label-anomaly-value` | 1 | 異常を示すラベル値 |
| `--n-trials` | 100 | Optunaの試行回数 |
| `--lof-tune-weights` / `--no-lof-tune-weights` | True | LOF+Optuna時に特徴量重みをチューニングするか |

> **目的関数**: ラベル付き異常データの異常スコア中央値が全体の上位何%かを最小化（= より異常なスコアを付けられるパラメータを探索）

### 前処理

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--missing-rate-threshold` | 0.80 | 欠損率がこの値を超えるカラムを除外 |
| `--ohe-top-n` | 10 | OHE対象とする上位カテゴリ数（グローバル設定） |
| `--ohe-coverage-threshold` | 0.50 | 上位N件のカバレッジがこの値以上ならOHE |
| `--variance-threshold` | 0.01 | 分散がこの値未満のカラムを除外 |
| `--corr-threshold` | 0.95 | 相関係数がこの値を超えるペアの片方を除外 |

### 出力

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--shap-all` / `--no-shap-all` | False | 全件SHAP計算 / 異常レコードのみ |
| `--pca-variance-warning` | 0.50 | PCA累積寄与率の警告閾値 |

---

## カラム設定JSON（column_config.json）

`column_config.json` でカラムごとに型・エンコーディング方法を指定します。
**JSONに記載されていないカラムは処理から除外されます。**

```json
{
  "age":          {"use": true,  "type": "numeric"},
  "created_at":   {"use": true,  "type": "date"},
  "region":       {"use": true,  "type": "categorical", "encoding": "ohe",       "ohe_top_n": 5},
  "description":  {"use": true,  "type": "categorical", "encoding": "frequency"},
  "status":       {"use": true,  "type": "categorical", "encoding": "auto",
                   "ohe_top_n": 10, "ohe_coverage_threshold": 0.7},
  "product_code": {"use": true,  "type": "categorical", "encoding": "ohe",
                   "ohe_top_n": 20, "prefix_n": 3},
  "customer_id":  {"use": false, "type": "categorical", "encoding": "frequency"}
}
```

### フィールド説明

| フィールド | 値 | 説明 |
|-----------|-----|------|
| `use` | `true` / `false` | `false` のカラムは処理から除外 |
| `type` | `numeric` / `date` / `categorical` | 型を強制指定 |
| `encoding` | `ohe` / `frequency` / `auto` | エンコーディング方式 |
| `ohe_top_n` | 整数 | OHE対象とする上位カテゴリ数（カラム個別設定） |
| `ohe_coverage_threshold` | 0〜1 | `auto` モード時のOHE判定閾値 |
| `prefix_n` | 整数 | 先頭N文字だけでエンコード（例: `"ABC-001"` → `"ABC"`） |

### エンコーディング方式

| 方式 | 動作 |
|------|------|
| `ohe` | 上位 `ohe_top_n` カテゴリをOne-Hot Encoding、残りは `__other__` にまとめる |
| `frequency` | カテゴリを出現頻度（比率）に変換 |
| `auto` | 上位 `ohe_top_n` のカバレッジ ≥ 閾値 → OHE、未満 → 頻度エンコーディング |

---

## 前処理の流れ

```
CSV読み込み（Shift-JIS対応）
  ↓
column_config.json の適用（use=false カラム除外 / 未記載カラム除外）
  ↓
欠損率チェック（閾値超えは除外）
  ↓
型別処理:
  数値カラム  → 欠損補完（中央値）
  日付カラム  → 年/月/日/時/曜日 に分解
  カテゴリカラム → OHE / 頻度エンコーディング
  bool カラム → 0/1 変換
  ↓
低分散カラム除外（variance_threshold）
  ↓
高相関カラム除外（corr_threshold）
  ↓
StandardScaler（全カラム一括標準化）
  ↓
異常検知モデル（IF または LOF）
```

---

## LOF の特徴量重みづけ

LOF は距離ベースの手法のため、**各特徴量のスケールが距離計算に直接影響**します。
Optuna チューニング時に `--lof-tune-weights` を有効にすると、元カラム単位で重みを自動最適化します。

```
X_weighted = X_scaled × weights   # 重みで各軸を引き伸ばす
```

**ポイント**: OHEで展開されたカラム群（例: `region_東京`, `region_大阪`...）には**元カラム（`region`）に対して1つの重み**を設定し、全展開列に一括適用します。日付分解カラムも同様です。

チューニング完了後、最良重みは元カラム単位でログ出力されます。

---

## 出力ファイル詳細

### `out/result.csv`

元データに以下の列を追加したファイルです。

| カラム名 | 型 | 説明 |
|----------|----|------|
| `anomaly_score` | float | 異常スコア（高いほど異常） |
| `is_anomaly` | int | 異常フラグ（1: 異常, 0: 正常） |
| `top_feature` | str | 最も異常に寄与した特徴量名（SHAP由来、IFのみ） |
| `top_shap_value` | float | その特徴量のSHAP値（IFのみ） |

### `out/shap_summary.csv`

各レコード × 各特徴量の SHAP 値マトリクスです（IFのみ）。
デフォルトでは **is_anomaly=1 のレコードのみ**出力されます。

### `out/pca_2d.csv`（Tableau連携メイン）

| カラム名 | 説明 |
|----------|------|
| `PC1` | 第1主成分スコア |
| `PC2` | 第2主成分スコア |
| `anomaly_score` | 異常スコア |
| `is_anomaly` | 異常フラグ |
| `top_feature` | 最大寄与特徴量名 |
| （元データ全列） | Tableauでのドリルダウン・フィルタ用 |

### `out/encoding_summary.csv`

カテゴリ変数のエンコード結果サマリです。

| カラム名 | 説明 |
|----------|------|
| `column` | 元カラム名 |
| `encoding` | 採用されたエンコーディング方式 |
| `n_unique` | ユニーク値数 |
| `coverage_top_n` | 上位N件のカバレッジ率 |
| `other_count` | `__other__` に集約されたレコード数 |
| `generated_columns` | 生成されたカラム名 |
| `prefix_n` | 使用した先頭N文字数 |

### `out/tuning_results.csv`（`--tune` 時のみ）

Optuna の全試行結果を最良スコア順に出力します。

### `out/processing.log`

- **コンソール出力**: INFO 以上（簡潔な進捗）
- **ファイル出力**: DEBUG 以上（DataFrame統計・カラム別詳細・SHAPランキング・相関ペアなど）

---

## ユーティリティスクリプト

### `dedup.py` — 重複削除

全カラムの値が同じレコードを削除します。

```bash
python dedup.py \
  --sum-cols 支払金額 税額 \      # 重複時に合算する数値カラム
  --ignore-cols 備考 更新日時 \   # 重複判定に使わず max を取るカラム
  --encoding cp932
```

| 引数 | 説明 |
|------|------|
| `--sum-cols` | 重複グループ内で合算する数値カラム |
| `--ignore-cols` | 重複判定から除外し max を取るカラム |
| `--in-dir` | 入力CSVフォルダ（デフォルト: `in`） |
| `--out-dir` | 出力フォルダ（デフォルト: `out`） |
| `--encoding` | 文字コード（デフォルト: `cp932`） |

### `check_coverage.py` — カテゴリカバレッジ確認

カテゴリ変数の上位10件・上位20件のカバレッジ率を一覧表示します。
`column_config.json` の `ohe_top_n` 設定の参考にしてください。

```bash
python check_coverage.py --encoding cp932
```

出力: `out/coverage_check.csv`

---

## Tableau 可視化ガイド

### 推奨ダッシュボード構成

| シート名 | 入力ファイル | X軸 | Y軸 | 色 / サイズ |
|---------|------------|-----|-----|------------|
| PCA散布図 | pca_2d.csv | PC1 | PC2 | is_anomaly（色）/ anomaly_score（サイズ） |
| 異常スコア分布 | result.csv | anomaly_score | 件数 | is_anomaly（色） |
| SHAP寄与度ランキング | shap_summary.csv | 特徴量名 | 平均\|SHAP\| | 特徴量（色） |
| 異常レコード詳細 | result.csv | レコードID | 各特徴量値 | top_feature（色） |

### 推奨フィルタ

- `is_anomaly`: 正常/異常の切り替え
- `anomaly_score`: スライダーによる閾値調整
- `top_feature`: 原因特徴量での絞り込み

---

## エラー対処

| エラーメッセージ | 原因と対処 |
|----------------|-----------|
| `入力フォルダ 'in' が存在しません` | `in/` フォルダを作成してCSVを配置してください |
| `CSVファイルが存在しません` | `in/` にCSVファイルがあるか確認してください |
| `カラム設定ファイルが見つかりません` | `column_config.json` を作成してください（`column_config.example.json` を参考に） |
| `JSONでONに設定されているがデータに存在しないカラム` | `column_config.json` のカラム名とCSVのカラム名が一致しているか確認してください |
| `前処理後に有効なカラムが0件` | 有効なカラムがありません。`excluded_columns.txt` を確認してください |
| `特徴量選択後に有効なカラムが0件` | 全カラムが低分散・高相関として除外されました。`processing.log` を確認してください |
| `--tune を使用するには --label-col が必須` | `--label-col` でラベル列名を指定してください |
| SHAP計算エラー（警告ログ） | LOF使用時はSHAPをスキップします。IFでエラーの場合も処理は継続します |

詳細なデバッグは `out/processing.log` を参照してください。

---

## 依存ライブラリ

```
pandas
scikit-learn
shap
optuna
```

---

## ライセンス

MIT License
