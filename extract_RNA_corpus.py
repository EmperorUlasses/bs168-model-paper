# -*- coding: utf-8 -*-
"""
Extract sequences from Column B (Sequence) in gene_locustag.xlsx,
produce MLM continue‑pretraining text corpus (one sequence per line).

Output:
    RNA_train.txt
    RNA_val.txt
"""


import os
import random
import argparse

import pandas as pd


def clean_seq(s):
    s = str(s).strip().upper()
    s = "".join(c for c in s if c in "ATCGN")
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str,
                        default="data/gene_locustag.xlsx",
                        help="Excel 文件路径，默认 data/gene_locustag.xlsx")
    parser.add_argument("--sheet", type=str, default="Sheet1",
                        help="sheet 名，默认 Sheet1")
    parser.add_argument("--col", type=str, default="Sequence",
                        help="序列列名，默认 Sequence")
    parser.add_argument("--out_dir", type=str, default="data",
                        help="输出目录，默认 data")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="验证集比例，默认 0.05")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_len", type=int, default=1,
                        help="最短序列长度过滤，默认 1")
    parser.add_argument("--max_len", type=int, default=100000,
                        help="最长序列长度过滤，默认不限制")
    parser.add_argument("--dedup", action="store_true",
                        help="是否去重（默认不去重，保留每条基因序列）")
    args = parser.parse_args()

    if not os.path.exists(args.excel):
        print(f"[错误] 找不到文件：{args.excel}")
        return

    print(f"读取 Excel: {args.excel} (sheet={args.sheet})")
    df = pd.read_excel(args.excel, sheet_name=args.sheet)

    target = None
    for c in df.columns:
        if str(c).strip().lower() == args.col.strip().lower():
            target = c
            break

    if target is None:
        print(f"[warn] 没找到列 '{args.col}'，改用第二列：{df.columns[1]}")
        target = df.columns[1]

    print(f"使用列：'{target}'，共 {len(df)} 行")

    all_seqs = []
    for v in df[target].dropna():
        s = clean_seq(v)
        if s:
            all_seqs.append(s)

    print(f"提取到 {len(all_seqs)} 条序列")

    before = len(all_seqs)
    all_seqs = [s for s in all_seqs if args.min_len <= len(s) <= args.max_len]
    print(f"长度过滤: {before} -> {len(all_seqs)}")

    if args.dedup:
        all_seqs = list(dict.fromkeys(all_seqs))
        print(f"去重后: {len(all_seqs)} 条")

    lens = [len(s) for s in all_seqs]
    print(f"长度 min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.1f}")

    random.seed(args.seed)
    random.shuffle(all_seqs)

    n_val = max(1, int(args.val_ratio * len(all_seqs)))
    val_seqs = all_seqs[:n_val]
    train_seqs = all_seqs[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "RNA_train.txt")
    val_path = os.path.join(args.out_dir, "RNA_val.txt")

    with open(train_path, "w") as f:
        f.write("\n".join(train_seqs) + "\n")
    with open(val_path, "w") as f:
        f.write("\n".join(val_seqs) + "\n")

    print("=" * 60)
    print(f"train: {len(train_seqs)} 条 -> {train_path}")
    print(f"val:   {len(val_seqs)} 条 -> {val_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()