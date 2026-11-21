# -*- coding: utf-8 -*-
"""
Replace ISM‑tuned successful sequences back into the full quality‑control table checked_generated_sequences.csv.
Since the pandas index is not preserved in the CSV file, 
explicit row‑number matching is adopted.
Output the new consolidated table: 
merged_checked_generated.csv
"""

import pandas as pd
import os

BASE_DIR = "Conditional diffusion"
CHECKED_CSV = os.path.join(BASE_DIR, "checked_generated_sequences.csv")
TUNED_RES_CSV = os.path.join(BASE_DIR, "tuned_results.csv")
OUT_MERGED_CSV = os.path.join(BASE_DIR, "merged_checked_generated.csv")


def main():
    if not os.path.exists(CHECKED_CSV):
        raise FileNotFoundError(f"找不到质控文件 {CHECKED_CSV}")
    if not os.path.exists(TUNED_RES_CSV):
        raise FileNotFoundError(f"找不到微调结果 {TUNED_RES_CSV}")


    df_full = pd.read_csv(CHECKED_CSV)
    df_full["_original_row"] = df_full.index   

    df_tune = pd.read_csv(TUNED_RES_CSV)

    print(f"完整质控表总行数: {len(df_full)}")
    print(f"微调任务记录条数: {len(df_tune)}")


    tune_success = df_tune[df_tune["success"] == True].copy()
    print(f"微调成功数量: {len(tune_success)}")

    update_count = 0
    for _, row in tune_success.iterrows():
        seed_idx = int(row["seed_idx"])   
        new_seq = row["final_seq"]
        new_pred_log10 = row["pred_after"]

        mask = df_full["_original_row"] == seed_idx
        if mask.sum() == 0:
            print(f"警告：seed_idx={seed_idx}不在完整表中，跳过")
            continue

        df_full.loc[mask, "sequence"] = new_seq
        df_full.loc[mask, "predicted_log10"] = new_pred_log10
        df_full.loc[mask, "predicted_raw"] = 10.0 ** new_pred_log10
        update_count += 1

    print(f"实际完成替换行数：{update_count}")

    df_full.drop(columns=["_original_row"], inplace=True)

    df_full.to_csv(OUT_MERGED_CSV, index=False, encoding="utf-8-sig")
    print(f"已输出合并替换完成完整文件：{OUT_MERGED_CSV}")
    print("\n提示：接下来修改evaluate.py的INPUT_CSV为本输出文件，重新跑全套质控+DNABERT预测！")

if __name__ == "__main__":
    main()
