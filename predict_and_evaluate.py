# -*- coding: utf-8 -*-
"""
Load the 5-fold models trained by DNABERT-2_BPE+6-mer+CNN.py.
Use each fold to predict expression levels for new promoter sequences, then perform ensemble inference.
Save metrics for each fold, ensemble metrics, figures and tables.
"""


import os, time, glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoConfig, AutoTokenizer, BertConfig, AutoModelForSequenceClassification
import torch.nn as nn


MODEL_DIR = "results_new/results_RNA"   

EXCEL_INPUT = "data/95bp_数据集合并.xlsx"   
EXCEL_SHEET = "promoter_sampled_1000"       
RESULTS_DIR = "predict/predictions_RNA"  

SEQ_COL = "Sequence"

REAL_LABEL_COL = "Average GFP expression"  # e.g. "Average GFP expression"

LABEL_LOG_BASE = 10
LABEL_CLIP_MIN = 5e-4

MAX_LEN = 130  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clean_sequence(seq):
    return str(seq).upper().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

def load_fold_model(fold_dir):
    tokenizer = AutoTokenizer.from_pretrained(fold_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(fold_dir, trust_remote_code=True)
    config.num_labels = 1
    config.hidden_dropout_prob = 0.00001
    config.attention_probs_dropout_prob = 0.00001

    model = AutoModelForSequenceClassification.from_pretrained(
        fold_dir, config=config, trust_remote_code=True, ignore_mismatched_sizes=True
    )

    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def predict_hf(model, tokenizer, sequences, max_len=MAX_LEN, batch_size=32):
    model.eval()
    cleaned = [clean_sequence(s) for s in sequences]
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = out.logits.view(-1).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)


def evaluate(y_true, y_pred):
    pearson = pearsonr(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred)
    return {
        "pearson_r": pearson[0], "pearson_p": pearson[1],
        "spearman_r": spearman.correlation, "spearman_p": spearman.pvalue,
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def plot_results(y_true, y_pred, out_prefix, title_suffix=""):
    plt.figure(figsize=(7,6))
    plt.scatter(y_true, y_pred, alpha=0.3, edgecolors='none')  
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Expression')
    plt.ylabel('Predicted Expression')
    plt.title('Predicted vs Actual {title_suffix}'.strip())
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pred_vs_actual.png")
    plt.close()

    residuals = y_pred - y_true
    plt.figure(figsize=(7,6))
    sns.histplot(residuals, kde=True, alpha=0.3)  
    plt.xlabel('Residual (Predicted - Actual)')
    plt.title('Residuals Distribution {title_suffix}'.strip())
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_residuals.png")
    plt.close()


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_prefix = os.path.join(RESULTS_DIR, f"predict_{int(t0)}")

    fold_dirs = sorted([
        d for d in glob.glob(os.path.join(MODEL_DIR, "fold*"))
        if os.path.isdir(d)
    ])
    if not fold_dirs:
        raise RuntimeError(f"No fold* dirs found under {MODEL_DIR}")
    print(f"Found {len(fold_dirs)} fold dirs:")
    for d in fold_dirs:
        print("  ", d)

    df = pd.read_excel(EXCEL_INPUT, sheet_name=EXCEL_SHEET)
    if SEQ_COL not in df.columns:
        raise ValueError(f"Input Excel must have '{SEQ_COL}' column. Columns: {list(df.columns)}")
    df[SEQ_COL] = df[SEQ_COL].apply(clean_sequence)
    sequences = df[SEQ_COL].tolist()
    print(f"Loaded {len(sequences)} sequences from {EXCEL_INPUT}")

    if REAL_LABEL_COL is not None and REAL_LABEL_COL in df.columns:
        raw = df[REAL_LABEL_COL].astype(float).values
        y_true = np.log10(np.maximum(raw, LABEL_CLIP_MIN))
        has_truth = True
        print(f"Using real labels from '{REAL_LABEL_COL}' (log10(max(val, {LABEL_CLIP_MIN})))")
    else:
        y_true = None
        has_truth = False
        print("No real label column -> only predictions will be exported.")

    fold_preds = []          # list of (N,) arrays
    per_fold_rows = []
    for i, fold_dir in enumerate(fold_dirs):
        fold_name = os.path.basename(fold_dir)
        print(f"\nLoading {fold_name} ...")
        model, tokenizer = load_fold_model(fold_dir)
        preds = predict_hf(model, tokenizer, sequences, max_len=MAX_LEN)
        fold_preds.append(preds)

        if has_truth:
            m = evaluate(y_true, preds)
            per_fold_rows.append({"fold": fold_name, **m})
            print(f"  {fold_name}: R2={m['r2']:.4f}, MSE={m['mse']:.4f}, MAE={m['mae']:.4f}")

            fold_out_prefix = f"{out_prefix}_fold{i+1}"
            plot_results(y_true, preds, fold_out_prefix, title_suffix=f"(Fold {i+1})")

        del model
        torch.cuda.empty_cache()

    fold_preds = np.stack(fold_preds, axis=0)          # (n_folds, N)
    y_pred_mean = fold_preds.mean(axis=0)
    y_pred_std = fold_preds.std(axis=0)

    if per_fold_rows:
        per_fold_df = pd.DataFrame(per_fold_rows)
        per_fold_path = f"{out_prefix}_per_fold_metrics.xlsx"
        per_fold_df.to_excel(per_fold_path, index=False)
        print(f"\nPer-fold metrics -> {per_fold_path}")
        print(per_fold_df)

    if has_truth:
        ens_metrics = evaluate(y_true, y_pred_mean)
        print("\nEnsemble metrics:")
        for k, v in ens_metrics.items():
            print(f"  {k}: {v}")
        with open(f"{out_prefix}_ensemble_metrics.txt", "w") as f:
            for k, v in ens_metrics.items():
                f.write(f"{k}: {v}\n")

    out_df = df.copy()
    for i, p in enumerate(fold_preds):
        out_df[f"Pred_fold{i+1}"] = p
    out_df["Pred_mean"] = y_pred_mean
    out_df["Pred_std"] = y_pred_std
    if has_truth:
        out_df["Actual"] = y_true
        out_df["Residual_mean"] = y_pred_mean - y_true
    out_path = f"{out_prefix}_results.xlsx"
    out_df.to_excel(out_path, index=False)
    print(f"\nResults -> {out_path}")
    print(f"Done in {(time.time() - t0) / 60:.2f} min")

if __name__ == "__main__":
    main()
