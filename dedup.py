#!/usr/bin/env python3
"""
重複削除スクリプト

非数値カラムが全て一致するレコードを重複とみなし、
数値カラムを合算して1レコードにまとめる。

使い方:
    python dedup.py
    python dedup.py --in-dir in --out-dir out
"""

import argparse
from pathlib import Path

import pandas as pd


def load_csvs(in_dir: Path) -> pd.DataFrame:
    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"[エラー] '{in_dir}' にCSVファイルがありません。")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, encoding="utf-8")
        print(f"読み込み: {f.name}  {len(df):,}件  {len(df.columns)}カラム")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"結合後: {len(combined):,}件\n")
    return combined


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    # 数値カラムと非数値カラムに自動分類
    num_cols = df.select_dtypes(include="number").columns.tolist()
    key_cols = [c for c in df.columns if c not in num_cols]

    print(f"キーカラム（非数値・重複判定に使用） {len(key_cols)}件: {key_cols}")
    print(f"合計カラム（数値）                  {len(num_cols)}件: {num_cols}")

    if not key_cols:
        raise SystemExit("[エラー] 非数値カラムが存在しないため重複判定できません。")

    n_before = len(df)

    if not num_cols:
        # 数値カラムなし → 単純重複削除
        result = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    else:
        # 非数値カラムでグループ化し、数値カラムを合算
        agg = {col: "sum" for col in num_cols}
        result = (
            df.groupby(key_cols, dropna=False, sort=False)
            .agg(agg)
            .reset_index()
        )
        # 元のカラム順に並べ直す
        result = result[df.columns]

    n_removed = n_before - len(result)
    print(f"\n削除件数: {n_removed:,}件  ({n_before:,} → {len(result):,}件)")

    # 数値カラムの合計値チェック（前後一致するはず）
    if num_cols:
        print("\n【数値カラムの合計値チェック（前後一致するはず）】")
        for col in num_cols:
            before = df[col].sum()
            after = result[col].sum()
            match = "✓" if abs(before - after) < 1e-6 else "✗ 不一致!"
            print(f"  {col}: {before:,.4f} → {after:,.4f}  {match}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="重複削除スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in-dir",  type=Path, default=Path("in"),  help="入力CSVフォルダ")
    parser.add_argument("--out-dir", type=Path, default=Path("out"), help="出力フォルダ")
    args = parser.parse_args()

    if not args.in_dir.exists():
        raise SystemExit(f"[エラー] 入力フォルダ '{args.in_dir}' が存在しません。")

    df = load_csvs(args.in_dir)
    result = dedup(df)

    args.out_dir.mkdir(exist_ok=True)
    out_path = args.out_dir / "dedup.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n→ {out_path} に保存しました。")


if __name__ == "__main__":
    main()
