#!/usr/bin/env python3
"""
カテゴリカラムのカバレッジ確認スクリプト

in/ フォルダのCSVを読み込み、数値以外のカラムについて
上位10件・上位20件のカバレッジ率を一覧表示する。
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    in_dir = Path("in")
    if not in_dir.exists():
        print(f"[エラー] 入力フォルダ '{in_dir}' が存在しません。")
        return

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        print(f"[エラー] '{in_dir}' にCSVファイルがありません。")
        return

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, encoding="utf-8")
        print(f"読み込み: {f.name}  {len(df)}件  {len(df.columns)}カラム")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n結合後: {len(combined)}件  {len(combined.columns)}カラム\n")

    # 数値に見えない（object型）カラムを抽出
    obj_cols = combined.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        print("数値以外のカラムが見つかりませんでした。")
        return

    print(f"対象カラム数: {len(obj_cols)}件\n")

    rows = []
    for col in obj_cols:
        s = combined[col].dropna()
        n_total = len(s)
        n_unique = s.nunique()
        freq = s.value_counts(normalize=True)
        cov10 = float(freq.nlargest(10).sum())
        cov20 = float(freq.nlargest(20).sum())
        rows.append({
            "カラム名":     col,
            "ユニーク数":   n_unique,
            "top10カバレッジ": round(cov10, 4),
            "top20カバレッジ": round(cov20, 4),
        })

    result = pd.DataFrame(rows).sort_values("top10カバレッジ", ascending=False)

    # コンソール表示
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print(result.to_string(index=False))

    # CSV出力
    out_path = Path("out/coverage_check.csv")
    out_path.parent.mkdir(exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n→ {out_path} に保存しました。")


if __name__ == "__main__":
    main()
