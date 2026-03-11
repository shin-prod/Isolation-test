#!/usr/bin/env python3
"""
重複削除スクリプト

全カラムが一致する完全重複レコードを削除する。
金額カラムなど集計が必要なカラムはコマンドライン引数で指定すると、
重複グループ内で合算した上で1レコードにまとめる。

使い方:
    # 完全重複を単純削除（合計なし）
    python dedup.py

    # 金額カラムを合算しながら重複削除
    python dedup.py --sum-cols 支払金額合計 税額 消費税額

    # 入出力フォルダを変える場合
    python dedup.py --sum-cols 支払金額合計 --in-dir in --out-dir out
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


def dedup(df: pd.DataFrame, sum_cols: list[str]) -> pd.DataFrame:
    n_before = len(df)

    # 指定した sum_cols がデータに存在するか確認
    missing = [c for c in sum_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[エラー] 指定した合計カラムがデータに存在しません: {missing}")

    if not sum_cols:
        # 合計カラム未指定 → 全カラム完全一致の重複を単純削除
        result = df.drop_duplicates()
        n_removed = n_before - len(result)
        print(f"完全重複を削除: {n_removed:,}件削除 ({n_before:,} → {len(result):,}件)")
        return result.reset_index(drop=True)

    # キーカラム = sum_cols 以外の全カラム
    key_cols = [c for c in df.columns if c not in sum_cols]
    print(f"キーカラム ({len(key_cols)}件): {key_cols}")
    print(f"合計カラム ({len(sum_cols)}件): {sum_cols}")

    # キーカラムが全て一致するグループ内で sum_cols を合算
    agg = {col: "sum" for col in sum_cols}
    result = df.groupby(key_cols, dropna=False, sort=False).agg(agg).reset_index()

    # 元のカラム順に並べ直す
    result = result[df.columns]

    n_removed = n_before - len(result)
    print(f"\n重複削除（合計あり）: {n_removed:,}件削除 ({n_before:,} → {len(result):,}件)")

    # 合計カラムの変化を確認
    print("\n【合計カラムの合計値チェック（前後一致するはず）】")
    for col in sum_cols:
        before = df[col].sum()
        after = result[col].sum()
        match = "✓" if abs(before - after) < 1e-6 else "✗ 不一致!"
        print(f"  {col}: {before:,.2f} → {after:,.2f}  {match}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="重複削除スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in-dir", type=Path, default=Path("in"),
                        help="入力CSVフォルダ")
    parser.add_argument("--out-dir", type=Path, default=Path("out"),
                        help="出力フォルダ")
    parser.add_argument("--sum-cols", nargs="*", default=[],
                        metavar="カラム名",
                        help="重複時に合算するカラム名（スペース区切りで複数指定可）")
    args = parser.parse_args()

    if not args.in_dir.exists():
        raise SystemExit(f"[エラー] 入力フォルダ '{args.in_dir}' が存在しません。")

    df = load_csvs(args.in_dir)
    result = dedup(df, args.sum_cols)

    args.out_dir.mkdir(exist_ok=True)
    out_path = args.out_dir / "dedup.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n→ {out_path} に保存しました。")


if __name__ == "__main__":
    main()
