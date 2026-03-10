# 異常検知システム 設計仕様書
**Isolation Forest + SHAP + PCA可視化 (Tableau連携)**
版数: 1.0　作成日: 2026年3月

---

## 1. システム概要

本システムは、カテゴリ変数・数値変数が混在するCSVデータを `in/` フォルダから入力し、自動的な前処理・型変換を行った後、Isolation Forestによる異常検知、SHAPによる異常原因の追及、PCAによる2次元可視化データの生成を行い、結果を `out/` フォルダに出力する。可視化はTableauで実施する。

| 項目 | 内容 |
|------|------|
| 入力フォルダ | `in/`（CSVファイルを配置） |
| 出力フォルダ | `out/`（結果CSVを自動生成） |
| 主要手法 | Isolation Forest |
| 原因分析 | SHAP（TreeExplainer） |
| 可視化データ出力 | PCA 2次元座標 → Tableau連携 |

---

## 2. フォルダ・ファイル構成

### 2.1 ディレクトリ構成

```
project/
  in/                      # 入力データフォルダ
    *.csv                  # 処理対象CSVファイル（複数可）
  out/                     # 出力フォルダ（自動生成）
    result.csv             # 異常スコア・ラベル付き結果
    shap_summary.csv       # SHAP値サマリ
    pca_2d.csv             # PCA 2次元座標（Tableau用）
    pca_variance.csv       # 主成分の寄与率
    excluded_columns.txt   # 除外カラム一覧
    processing.log         # 処理ログ
  main.py                  # メイン実行スクリプト
  requirements.txt         # 依存ライブラリ
```

### 2.2 入力ファイル仕様

| 項目 | 仕様 |
|------|------|
| フォーマット | CSV（ヘッダー行あり） |
| 文字コード | UTF-8 / Shift-JIS（自動判定） |
| 複数ファイル | `in/` 内の全CSVを結合して処理 |
| 数値変数 | int / float 型（自動検出） |
| カテゴリ変数 | str / object 型（自動検出） |
| 日付変数 | 自動検出 → datetime変換 → 数値特徴量化 |
| 欠損値 | 自動補完（数値: 中央値、カテゴリ: 最頻値） |

---

## 3. 処理フロー

### 3.1 全体フロー

1. `in/` フォルダからCSVを読み込み・結合
2. 自動型判定・前処理（型変換・欠損補完・エンコーディング）
3. 特徴量選択（低分散除去・高相関除去）
4. StandardScalerで標準化
5. Isolation Forest で異常スコア算出
6. SHAP で各レコードの異常原因を特徴量ごとに分解
7. PCA で2次元に圧縮
8. `out/` フォルダへ結果CSV一式を出力

---

## 4. 前処理仕様

### 4.1 自動型判定ロジック

pandasのdtypeおよびユニーク数比率をもとに以下のルールで型を判定する。

| 判定条件 | 処理内容 |
|----------|----------|
| dtype = int64 / float64 | 数値変数としてそのまま使用 |
| dtype = object かつ日付パース成功 | datetime変換 → 年/月/日/時/曜日に分解 |
| dtype = object かつユニーク数 <= 全件の50% | カテゴリ変数 → Label Encoding |
| dtype = object かつユニーク数 > 全件の50% | 高カーディナリティ → 頻度エンコーディング |
| bool 型 | 0/1 に変換 |

### 4.2 欠損値補完

- 数値変数: 中央値（median）で補完
- カテゴリ変数: 最頻値（mode）で補完
- 欠損率 50% 超のカラムは自動除外し、`excluded_columns.txt` に記録

### 4.3 特徴量選択（不要カラム除去）

- 分散フィルタ: 分散 < 0.01 のカラムを除外
- 相関フィルタ: 相関係数 > 0.95 のペアから片方を除外
- 除外されたカラム名を `out/excluded_columns.txt` に記録

### 4.4 スケーリング

StandardScalerで全特徴量を標準化（平均0、標準偏差1）してからIsolation Forestに投入する。SHAPおよびPCAにも同じスケール済みデータを使用する。

---

## 5. Isolation Forest 仕様

### 5.1 モデルパラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| n_estimators | 200 | 木の本数（多いほど安定） |
| contamination | 0.05 | 異常データの想定割合（5%） |
| max_features | 1.0 | 各木で使用する特徴量の割合 |
| random_state | 42 | 再現性のための乱数シード |

### 5.2 出力スコア

- `score_samples()` の出力をマイナス反転して `anomaly_score` カラムとして付与（高いほど異常）
- `predict()` の出力（-1: 異常, 1: 正常）を `is_anomaly` カラムとして付与（1: 異常, 0: 正常に変換）

---

## 6. SHAP 異常原因分析仕様

### 6.1 処理概要

TreeExplainerを使用してIsolation Forestの各予測に対するSHAP値を算出し、どの特徴量が異常スコアに寄与しているかを定量化する。

### 6.2 出力内容

| 出力先 | 内容 |
|--------|------|
| `out/shap_summary.csv` | 各レコード × 各特徴量のSHAP値（全件 or 異常レコードのみ） |
| `out/result.csv`（top_feature列） | そのレコードで最も異常に寄与した特徴量名 |
| `out/result.csv`（top_shap_value列） | 最大寄与特徴量のSHAP値（絶対値） |

### 6.3 計算対象の設定

- デフォルト: `is_anomaly=1` のレコードのみSHAP計算（高速）
- 全件計算が必要な場合は `main.py` の `SHAP_ALL = True` に変更

---

## 7. PCA 2次元可視化仕様

### 7.1 処理概要

StandardScaler適用済みの特徴量行列に対してPCAを適用し、第1・第2主成分への射影座標をTableau連携用CSVとして出力する。

### 7.2 出力ファイル仕様（out/pca_2d.csv）

| カラム名 | 型 | 説明 |
|----------|----|------|
| PC1 | float | 第1主成分スコア |
| PC2 | float | 第2主成分スコア |
| anomaly_score | float | Isolation Forestの異常スコア |
| is_anomaly | int | 異常フラグ（1: 異常, 0: 正常） |
| top_feature | str | 最大寄与特徴量名（SHAP由来） |
| （元データの全カラム） | - | Tableauでのドリルダウン・フィルタ用 |

### 7.3 寄与率の記録

PCAの各主成分の寄与率を `out/pca_variance.csv` に出力する。第1・第2主成分の累積寄与率が50%を下回る場合は警告ログを出力する（処理は継続）。

---

## 8. 出力ファイル一覧（out/）

| ファイル名 | 説明 | Tableau使用 |
|-----------|------|------------|
| result.csv | 元データ + anomaly_score + is_anomaly + top_feature + top_shap_value | 〇 |
| shap_summary.csv | レコード × 特徴量のSHAP値マトリクス | 〇（原因分析） |
| pca_2d.csv | PC1/PC2座標 + 異常情報 + 元データ全列 | 〇（メイン） |
| pca_variance.csv | 各主成分の寄与率・累積寄与率 | - |
| excluded_columns.txt | 前処理で除外されたカラム名一覧 | - |
| processing.log | 処理ログ（件数・除外カラム・警告等） | - |

---

## 9. Tableau 可視化設計

### 9.1 推奨ダッシュボード構成

| シート名 | 入力ファイル | X軸 | Y軸 | 色 / サイズ |
|---------|------------|-----|-----|------------|
| PCA散布図 | pca_2d.csv | PC1 | PC2 | is_anomaly（色）/ anomaly_score（サイズ） |
| 異常スコア分布 | result.csv | anomaly_score | 件数（Bin） | is_anomaly（色） |
| SHAP寄与度ランキング | shap_summary.csv | 特徴量名 | 平均\|SHAP値\| | 特徴量（色） |
| 異常レコード詳細 | result.csv | レコードID | 各特徴量値 | top_feature（色） |

### 9.2 推奨フィルタ設定

- `is_anomaly`: 0/1 切り替えフィルタ（正常/異常の切り替え）
- `anomaly_score`: スライダーによる閾値調整
- `top_feature`: 原因特徴量での絞り込み

---

## 10. 実行方法

### 10.1 依存ライブラリ（requirements.txt）

```
pandas
scikit-learn
shap
chardet
```

### 10.2 実行手順

1. `in/` フォルダに処理対象CSVを配置する
2. `python main.py` を実行する
3. `out/` フォルダに生成されたCSVをTableauで開く

### 10.3 主要設定パラメータ（main.py）

| 設定キー | デフォルト | 説明 |
|---------|----------|------|
| CONTAMINATION | 0.05 | 異常割合の想定値 |
| N_ESTIMATORS | 200 | Isolation Forestの木の本数 |
| VARIANCE_THRESHOLD | 0.01 | 低分散カラム除外の閾値 |
| CORR_THRESHOLD | 0.95 | 高相関カラム除外の閾値 |
| SHAP_ALL | False | True: 全件SHAP計算 / False: 異常レコードのみ |
| PCA_VARIANCE_WARNING | 0.50 | 累積寄与率の警告閾値 |

---

## 11. エラー処理・ログ

| 状況 | 処理内容 |
|------|---------|
| `in/` フォルダが存在しない | エラー終了・メッセージ表示 |
| `in/` 内にCSVが0件 | エラー終了・メッセージ表示 |
| 前処理後に有効カラムが0件 | エラー終了・除外理由をログ出力 |
| 欠損率50%超のカラム検出 | 警告ログ出力後にそのカラムを除外 |
| PCA累積寄与率50%未満 | 警告ログ出力（処理は継続） |
| SHAP計算エラー | 警告ログ出力・SHAP出力スキップ（result.csvは出力） |

---

*異常検知システム設計仕様書 v1.0*
