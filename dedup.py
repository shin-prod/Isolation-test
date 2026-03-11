#!/usr/bin/env python3
"""
重複削除スクリプト

--sum-cols で指定したカラムを合算対象とし、
残りのカラムが全て一致するレコードを重複とみなして1レコードにまとめる。

使い方:
    python dedup.py --sum-cols 支払金額合計 税額 消費税額
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
    # 指定カラムの存在確認
    missing = [c for c in sum_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[エラー] 指定した合計カラムがデータに存在しません: {missing}")

    key_cols = [c for c in df.columns if c not in sum_cols]

    print(f"キーカラム（重複判定に使用） {len(key_cols)}件: {key_cols}")
    print(f"合計カラム                  {len(sum_cols)}件: {sum_cols}")

    if not key_cols:
        raise SystemExit("[エラー] キーカラムが0件です。全カラムを --sum-cols に指定しないでください。")

    n_before = len(df)

    if not sum_cols:
        result = df.drop_duplicates().reset_index(drop=True)
    else:
        agg = {col: "sum" for col in sum_cols}
        result = (
            df.groupby(key_cols, dropna=False, sort=False)
            .agg(agg)
            .reset_index()
        )
        result = result[df.columns]

    n_removed = n_before - len(result)
    print(f"\n削除件数: {n_removed:,}件  ({n_before:,} → {len(result):,}件)")

    if sum_cols:
        print("\n【合計カラムの合計値チェック（前後一致するはず）】")
        for col in sum_cols:
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
    parser.add_argument("--sum-cols", nargs="*", default=[],
                        metavar="カラム名",
                        help="重複時に合算するカラム名（スペース区切りで複数指定）")
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
