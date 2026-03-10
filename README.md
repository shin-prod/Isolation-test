# 異常検知システム

**Isolation Forest + SHAP + PCA可視化（Tableau連携）**

CSVデータを入力するだけで、機械学習による異常検知・原因分析・可視化データ生成を自動実行します。

---

## 概要

| 項目 | 内容 |
|------|------|
| 入力 | `in/` フォルダのCSVファイル（UTF-8、複数可） |
| 出力 | `out/` フォルダに結果CSV一式を自動生成 |
| 異常検知 | Isolation Forest（scikit-learn） |
| 原因分析 | SHAP（TreeExplainer） |
| 可視化用途 | PCA 2次元座標 → Tableau連携 |

---

## ディレクトリ構成

```
isolation_test/
├── in/                      # 入力データ（CSVをここに置く）
├── out/                     # 出力フォルダ（自動生成）
│   ├── result.csv           # 異常スコア・フラグ付き結果
│   ├── shap_summary.csv     # SHAP値マトリクス
│   ├── pca_2d.csv           # PCA座標 + 元データ（Tableau用メイン）
│   ├── pca_variance.csv     # 主成分の寄与率
│   ├── excluded_columns.txt # 前処理で除外されたカラム一覧
│   └── processing.log       # 詳細処理ログ（DEBUG含む）
├── main.py                  # メインスクリプト
├── run.sh                   # 実行シェルスクリプト
├── requirements.txt         # 依存パッケージ
├── .env                     # 仮想環境パスの設定
└── README.md                # 本ファイル
```

---

## セットアップ

### 前提条件

- Python 3.9 以上
- bash が使えること（macOS / Linux）

### 手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/shin-prod/Isolation-test.git
cd Isolation-test

# 2. .env で仮想環境の保存先を確認・変更（任意）
cat .env
# VENV_PATH=./.venv  ← 変更したい場合はここを編集

# 3. 入力データを in/ フォルダに配置
cp /path/to/your/data.csv in/

# 4. 実行スクリプトを起動（初回は仮想環境を自動作成）
./run.sh
```

> **初回実行時**は `pip install` が走るため数分かかる場合があります。

---

## 実行方法

### シェルスクリプト（推奨）

```bash
./run.sh
```

スクリプトは以下を自動処理します：

1. `.env` から仮想環境パスを読み込む
2. `in/` フォルダとCSVファイルの存在チェック
3. `out/` フォルダを自動作成
4. venv が未作成なら自動作成（初回のみ）
5. `requirements.txt` のパッケージをインストール
6. `python main.py` を実行

### 直接実行（venv 有効化済みの場合）

```bash
source .venv/bin/activate
python main.py
```

---

## 入力データ仕様

| 項目 | 仕様 |
|------|------|
| フォーマット | CSV（ヘッダー行あり） |
| 文字コード | **UTF-8**（BOMなし推奨） |
| 複数ファイル | `in/` 内の全CSVを結合して処理 |
| 数値変数 | int / float 型（自動検出） |
| カテゴリ変数 | str / object 型（自動エンコーディング） |
| 日付変数 | 自動検出 → 年/月/日/時/曜日に分解 |
| 欠損値 | 自動補完（数値: 中央値、カテゴリ: 最頻値） |

### 前処理の動作

| 状況 | 処理内容 |
|------|---------|
| 欠損率 > 50% のカラム | 自動除外 → `excluded_columns.txt` に記録 |
| bool 型 | 0/1 に変換 |
| object 型（日付として解釈できる） | datetime 分解（年・月・日・時・曜日） |
| object 型（ユニーク比率 ≤ 50%） | Label Encoding |
| object 型（ユニーク比率 > 50%） | 頻度エンコーディング |
| 分散 < 0.01 のカラム | 低分散として除外 |
| 相関係数 > 0.95 のペア | 片方を除外 |

---

## 出力ファイル詳細

### `out/result.csv`

元データに以下の列を追加したファイルです。

| カラム名 | 型 | 説明 |
|----------|----|------|
| anomaly_score | float | 異常スコア（高いほど異常） |
| is_anomaly | int | 異常フラグ（1: 異常, 0: 正常） |
| top_feature | str | 最も異常に寄与した特徴量名（SHAP由来） |
| top_shap_value | float | その特徴量のSHAP値（絶対値） |

### `out/shap_summary.csv`

各レコード × 各特徴量の SHAP 値マトリクスです。
デフォルトでは **is_anomaly=1 のレコードのみ**出力されます。

### `out/pca_2d.csv`（Tableau連携メイン）

| カラム名 | 説明 |
|----------|------|
| PC1 | 第1主成分スコア |
| PC2 | 第2主成分スコア |
| anomaly_score | 異常スコア |
| is_anomaly | 異常フラグ |
| top_feature | 最大寄与特徴量名 |
| （元データ全列） | Tableauでのドリルダウン・フィルタ用 |

### `out/pca_variance.csv`

各主成分の寄与率・累積寄与率を記録します。

### `out/processing.log`

- **コンソール出力**: INFO 以上（簡潔な進捗）
- **ファイル出力**: DEBUG 以上（DataFrame 統計・カラム別詳細・SHAP ランキング・相関ペアなど）

デバッグ時は `processing.log` を確認してください。

---

## パラメータ設定

`main.py` の `Config` クラスで調整できます。

```python
@dataclass(frozen=True)
class Config:
    contamination: float = 0.05      # 異常割合の想定値（0.0〜0.5）
    n_estimators: int = 200          # 木の本数（多いほど安定・低速）
    max_features: float = 1.0        # 各木で使用する特徴量の割合
    random_state: int = 42           # 乱数シード（再現性）

    missing_rate_threshold: float = 0.50   # 欠損率除外閾値
    date_parse_threshold: float = 0.80     # 日付判定の成功率閾値
    high_cardinality_threshold: float = 0.50  # 高カーディナリティ判定閾値
    variance_threshold: float = 0.01       # 低分散除外閾値
    corr_threshold: float = 0.95           # 高相関除外閾値

    shap_all: bool = False           # True: 全件SHAP計算 / False: 異常レコードのみ
    pca_variance_warning: float = 0.50     # PCA累積寄与率の警告閾値
```

---

## Tableau 可視化ガイド

`out/` フォルダのCSVを Tableau Desktop / Public で開いてください。

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
| `前処理後に有効なカラムが0件` | 数値変換できるカラムがありません。`excluded_columns.txt` を確認してください |
| `特徴量選択後に有効なカラムが0件` | 全カラムが低分散・高相関として除外されました。`processing.log` を確認してください |
| SHAP計算エラー（警告ログ） | SHAP計算に失敗しましたが処理は継続します。`result.csv` は正常に出力されます |

詳細なデバッグは `out/processing.log` を参照してください。

---

## 依存ライブラリ

```
pandas
scikit-learn
shap
```

---

## ライセンス

MIT License
