#!/usr/bin/env python3
"""
重複削除スクリプト

カラムを3種類に分類して重複集約する。

  --sum-cols    : 数値カラム。重複グループ内で合算する。
  --ignore-cols : 無視カラム（文字列想定）。重複判定に使わず max を取る。
  残り          : キーカラム。これら全てが一致するレコードを重複とみなす。

使い方:
    python dedup.py --sum-cols 支払金額合計 税額 --ignore-cols 備考 更新日時
"""

import argparse
from pathlib import Path

import pandas as pd


def load_csvs(in_dir: Path, encoding: str = "cp932") -> pd.DataFrame:
    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"[エラー] '{in_dir}' にCSVファイルがありません。")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, encoding=encoding)
        print(f"読み込み: {f.name}  {len(df):,}件  {len(df.columns)}カラム")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"結合後: {len(combined):,}件\n")
    return combined


def dedup(
    df: pd.DataFrame,
    sum_cols: list[str],
    ignore_cols: list[str],
) -> pd.DataFrame:
    # 存在確認
    for label, cols in [("--sum-cols", sum_cols), ("--ignore-cols", ignore_cols)]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"[エラー] {label} に存在しないカラムが指定されています: {missing}")

    # 重複判定に使うキーカラム（sum / ignore 以外）
    excluded = set(sum_cols) | set(ignore_cols)
    key_cols = [c for c in df.columns if c not in excluded]

    print(f"キーカラム（重複判定）  {len(key_cols):3d}件: {key_cols}")
    print(f"合計カラム（sum）       {len(sum_cols):3d}件: {sum_cols}")
    print(f"無視カラム（max）       {len(ignore_cols):3d}件: {ignore_cols}")

    if not key_cols:
        raise SystemExit("[エラー] キーカラムが0件です。")

    n_before = len(df)

    agg = {col: "sum" for col in sum_cols}
    agg.update({col: "max" for col in ignore_cols})

    if agg:
        result = (
            df.groupby(key_cols, dropna=False, sort=False)
            .agg(agg)
            .reset_index()
        )
        result = result[df.columns]  # 元のカラム順に戻す
    else:
        result = df.drop_duplicates(subset=key_cols).reset_index(drop=True)

    n_removed = n_before - len(result)
    print(f"\n削除件数: {n_removed:,}件  ({n_before:,} → {len(result):,}件)")

    # 合計カラムの合計値チェック
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
    parser.add_argument("--encoding", type=str, default="cp932",
                        help="入力CSVの文字コード (cp932=Shift-JIS, utf-8 など)")
    parser.add_argument("--sum-cols", nargs="*", default=[], metavar="カラム名",
                        help="重複時に合算するカラム名（数値）")
    parser.add_argument("--ignore-cols", nargs="*", default=[], metavar="カラム名",
                        help="重複判定に使わず max を取るカラム名（文字列想定）")
    args = parser.parse_args()

    if not args.in_dir.exists():
        raise SystemExit(f"[エラー] 入力フォルダ '{args.in_dir}' が存在しません。")

    df = load_csvs(args.in_dir, args.encoding)
    result = dedup(df, args.sum_cols, args.ignore_cols)

    args.out_dir.mkdir(exist_ok=True)
    out_path = args.out_dir / "dedup.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n→ {out_path} に保存しました。")


if __name__ == "__main__":
    main()
