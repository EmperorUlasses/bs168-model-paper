# -*- coding: utf-8 -*-
"""
Model framework from literature + custom data/labels/splitting/fine-tuning freezing/best-checkpoint saving

Data source: multiple sheets (promoter_part1~5), Average GFP expression
Label: log10(val)
Splitting: by sheet (part1-5) + optional deduplication on unique sequences
Freezing: freeze_layers to control the number of bottom layers frozen
Saving: weights from the epoch with the optimal R²
"""


import os
import gc
import sys
import math
import time
import random
import hashlib
import transformers
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


batch_size = 32 
lr=9e-6 
weight_decay = 0.0006   
epochs = 25 
max_len = 130   
seed = 42

freeze_layers = 0              
freeze_embeddings = False

all_sheets = ['promoter_part1', 'promoter_part2', 'promoter_part3',
              'promoter_part4', 'promoter_part5']

train_sheets = None
test_sheet = None

use_sheet_split = True
val_size = 0.1                 

label_log_base = 10            
label_clip_min = 5e-4          

# CNN comparison model settings
cnn_epochs = 25
cnn_lr = 1e-3
cnn_weight_decay = 1e-4
cnn_dropout = 0.2

model_name = "output/bs168_RNA"  
data_filepath = "data/95bp_数据集合并.xlsx"
results_root = "results_new"  

results_dir = os.path.join(results_root, f"results_RNA")

pd.set_option("display.max_colwidth", None)


def set_global_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_sequence(seq):
    return str(seq).upper().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name)

    config.num_labels = 1

    config.hidden_dropout_prob = 0.00001    
    config.attention_probs_dropout_prob = 0.00001   

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True
    )

    return model, tokenizer


class GeneExpressionDataset(Dataset):
    def __init__(self, sequences, expressions, tokenizer, max_len):
        self.sequences = list(sequences)
        self.expressions = list(expressions)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        sequence = clean_sequence(self.sequences[item])
        expression = float(self.expressions[item])

        encoding = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(expression, dtype=torch.float32)
        }


def read_one_sheet(filepath, sheet_name):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    seqs, labs = [], []
    for i in range(len(df)):
        seq = str(df['Sequence'].iloc[i]).strip().upper()
        if seq.startswith('PROMOTER'):
            continue
        if not seq or set(seq) - set('ATCGN'):
            continue
        val = df['Average GFP expression'].iloc[i]
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        if math.isnan(val):
            continue

        val_clipped = max(val, 5e-4)
        label = math.log10(val_clipped)
        seqs.append(clean_sequence(seq))
        labs.append(label)
    return seqs, labs


def prepare_data(filepath, tokenizer, batch_size, max_len=130):
    if use_sheet_split:
        train_seqs, train_labs = [], []
        for s in train_sheets:
            seqs, labs = read_one_sheet(filepath, s)
            train_seqs.extend(seqs)
            train_labs.extend(labs)

        val_seqs, val_labs = read_one_sheet(filepath, test_sheet)

        train_set = set(train_seqs)
        overlap = train_set & set(val_seqs)
        if overlap:
            print(f"[warn] {len(overlap)} 条序列同时出现在 train/val，已从 val 移除")
            keep = [i for i, s in enumerate(val_seqs) if s not in overlap]
            val_seqs = [val_seqs[i] for i in keep]
            val_labs = [val_labs[i] for i in keep]

        print(f"按 sheet 划分：训练 {len(train_seqs)} 条，测试 {len(val_seqs)} 条")
    else:
        all_seqs, all_labs = [], []
        for s in train_sheets + [test_sheet]:
            seqs, labs = read_one_sheet(filepath, s)
            all_seqs.extend(seqs)
            all_labs.extend(labs)

        df = pd.DataFrame({'Sequence': all_seqs, 'NORM': all_labs})
        unique_seqs = df['Sequence'].drop_duplicates()
        train_u, val_u = train_test_split(
            unique_seqs, test_size=val_size, random_state=seed, shuffle=True
        )

        train_df = df[df["Sequence"].isin(train_u)]
        val_df = df[df["Sequence"].isin(val_u)]

        overlap = set(train_df['Sequence']) & set(val_df['Sequence'])
        assert len(overlap) == 0, f"Sequence leakage: {len(overlap)}"

        train_seqs = train_df['Sequence'].tolist()
        train_labs = train_df['NORM'].tolist()
        val_seqs = val_df['Sequence'].tolist()
        val_labs = val_df['NORM'].tolist()

        print(f"按 unique sequence 划分：训练 {len(train_seqs)} 条，测试 {len(val_seqs)} 条")

    y_train = np.array(train_labs, dtype=float)
    y_val = np.array(val_labs, dtype=float)
    print("标签分布（log10）：")
    print(pd.Series(y_train).describe())

    train_dataset = GeneExpressionDataset(train_seqs, y_train, tokenizer, max_len=max_len)
    val_dataset = GeneExpressionDataset(val_seqs, y_val, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader,
            np.array(train_seqs), y_train,
            np.array(val_seqs), y_val)


def apply_freeze(model, freeze_layers, freeze_embeddings=False):
    if freeze_layers > 0:
        for i in range(freeze_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    if freeze_embeddings:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen first {freeze_layers} encoder layers. "
          f"Trainable params: {trainable:,} / {total:,}")
    return trainable, total


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, scheduler=None,
                    epoch=0, epochs=1):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.view(-1)
        labels = labels.view(-1)

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    return total_loss / len(train_loader)


def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.view(-1)

            all_preds.extend(np.atleast_1d(preds.cpu().numpy()))
            all_labels.extend(np.atleast_1d(labels.cpu().numpy()))

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return all_preds, all_labels, mse, mae, r2


def sequence_to_spaced_kmers(seq, k):

    seq = clean_sequence(seq)
    if len(seq) < k:
        return ""
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


def run_kmer_ridge_baseline(X_train, y_train, X_val, y_val, k=6):
    print("######################################")
    print(f"Running {k}-mer Ridge baseline...")

    X_train_kmers = [sequence_to_spaced_kmers(seq, k) for seq in X_train]
    X_val_kmers = [sequence_to_spaced_kmers(seq, k) for seq in X_val]

    model = Pipeline([
        ("vectorizer", CountVectorizer(token_pattern=r"(?u)\b\w+\b")),
        ("regressor", Ridge(alpha=1.0))
    ])

    model.fit(X_train_kmers, y_train)
    preds = model.predict(X_val_kmers)

    mse = mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print(f"{k}-mer Ridge: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    return {
        "Model": f"{k}-mer Ridge",
        "Validation_MSE": mse,
        "Validation_MAE": mae,
        "Validation_R2": r2
    }


class OneHotSequenceDataset(Dataset):
    def __init__(self, sequences, expressions, max_len):
        self.sequences = list(sequences)
        self.expressions = list(expressions)
        self.max_len = max_len
        self.base_to_idx = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = clean_sequence(self.sequences[idx])
        x = torch.zeros(4, self.max_len, dtype=torch.float32)

        for pos, base in enumerate(seq[:self.max_len]):
            if base in self.base_to_idx:
                x[self.base_to_idx[base], pos] = 1.0

        y = torch.tensor(float(self.expressions[idx]), dtype=torch.float32)
        return x, y


class SimplePromoterCNN(torch.nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(4, 64, kernel_size=8, padding=4),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),

            torch.nn.Conv1d(64, 128, kernel_size=8, padding=4),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),

            torch.nn.Conv1d(128, 128, kernel_size=8, padding=4),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1)
        )

        self.regressor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.regressor(self.conv(x)).view(-1)


def evaluate_cnn_model(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)

            all_preds.extend(np.atleast_1d(preds.cpu().numpy()))
            all_labels.extend(np.atleast_1d(y.cpu().numpy()))

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return all_preds, all_labels, mse, mae, r2


def run_cnn_baseline(X_train, y_train, X_val, y_val, device, output_dir, timestamp,
                     max_len=130, batch_size=28, epochs=25, lr=1e-3, weight_decay=1e-4,
                     dropout=0.2):
    print("######################################")
    print("Running simple one-hot CNN baseline...")

    train_ds = OneHotSequenceDataset(X_train, y_train, max_len=max_len)
    val_ds = OneHotSequenceDataset(X_val, y_val, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SimplePromoterCNN(dropout=dropout).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_losses, val_mses, val_maes, val_r2s = [], [], [], []

    best_r2 = -float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        _, _, mse, mae, r2 = evaluate_cnn_model(model, val_loader, device)

        val_mses.append(mse)
        val_maes.append(mae)
        val_r2s.append(r2)

        print(f"CNN epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}, "
              f"MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"CNN best Validation R2: {best_r2:.4f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    all_preds, all_labels, mse, mae, r2 = evaluate_cnn_model(model, val_loader, device)

    # Save CNN scores from each training round.
    cnn_metrics_df = pd.DataFrame({
        "Epoch": np.arange(1, len(train_losses) + 1),
        "Training_Loss": train_losses,
        "Validation_MSE": val_mses,
        "Validation_MAE": val_maes,
        "Validation_R2": val_r2s
    })

    cnn_metrics_path = os.path.join(output_dir, f"simple_cnn_training_metrics_{timestamp}.xlsx")
    cnn_metrics_df.to_excel(cnn_metrics_path, index=False)
    print(f"Simple CNN training metrics exported to {cnn_metrics_path}")

    pd.DataFrame({
        "Sequence": X_val, "Actual": all_labels, "Predicted": all_preds,
        "Residuals": np.array(all_preds) - np.array(all_labels),
    }).to_excel(os.path.join(output_dir, f"simple_cnn_evaluation_results_{timestamp}.xlsx"), index=False)

    torch.save(model.state_dict(), os.path.join(output_dir, f"simple_cnn_model_{timestamp}.pth"))

    return {
        "Model": "Simple one-hot CNN",
        "Validation_MSE": mse,
        "Validation_MAE": mae,
        "Validation_R2": r2,
    }


def plot_training_metrics(train_losses, val_mses, val_maes, val_r2s, model_name, timestamp, output_dir):
    safe = os.path.basename(model_name)
    ep = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(ep, train_losses, marker="o", label="Training Loss")
    plt.plot(ep, val_mses, marker="s", linestyle="--", label="Validation MSE")
    plt.plot(ep, val_maes, marker="^", linestyle=":", label="Validation MAE")
    plt.title("Training Loss and Validation Error over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Loss / Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe}_training_loss_val_metrics_{timestamp}.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(ep, val_r2s, marker="o", label="Validation R2")
    plt.title("Validation R2 over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("R2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe}_validation_r2_{timestamp}.png"))
    plt.close()


def plot_evaluation_results(all_preds, all_labels, model_name, output_dir=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = os.path.basename(model_name)
    all_preds = np.array(all_preds); all_labels = np.array(all_labels)
    r2 = r2_score(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_preds, alpha=0.7)
    lo = min(all_labels.min(), all_preds.min())
    hi = max(all_labels.max(), all_preds.max())
    plt.plot([lo, hi], [lo, hi], color="red", lw=2)
    plt.text(0.05, 0.95, f"$R^2$ = {r2:.4f}", transform=plt.gca().transAxes,
             fontsize=14, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.title(f"Predicted vs Actual\n{safe}")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.tight_layout()
    plt.grid(True)

    out = os.path.join(output_dir, f"{safe}_pred_vs_actual_{timestamp}.png") if output_dir \
        else f"{safe}_pred_vs_actual_{timestamp}.png"

    plt.savefig(out)
    plt.close()

    residuals = all_preds - all_labels
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f"Residuals Distribution\n{safe}")
    plt.xlabel("Residuals"); plt.ylabel("Frequency")
    plt.tight_layout()
    out = os.path.join(output_dir, f"{safe}_residuals_{timestamp}.png") if output_dir \
        else f"{safe}_residuals_{timestamp}.png"

    plt.savefig(out)
    plt.close()


def export_training_metrics(train_losses, val_mses, val_maes, val_r2s, model_name, timestamp, output_dir):
    safe = os.path.basename(model_name)

    df = pd.DataFrame({
        "Epoch": np.arange(1, len(train_losses) + 1),
        "Training_Loss": train_losses,
        "Validation_MSE": val_mses,
        "Validation_MAE": val_maes,
        "Validation_R2": val_r2s
    })

    path = os.path.join(output_dir, f"{safe}_training_metrics_{timestamp}.xlsx")
    df.to_excel(path, index=False)
    print(f"Training metrics exported to {path}")


def export_results_to_excel(all_preds, all_labels, model_name, final_val_mse=None, final_val_mae=None,
                            final_val_r2=None, sequences=None, names=None, output_dir=None):
    df = pd.DataFrame({
        "Actual": all_labels,
        "Predicted": all_preds,
        "Residuals": np.array(all_preds) - np.array(all_labels)
    })

    if names is not None:
        df.insert(0, "Name", names)

    if sequences is not None:
        df.insert(1 if names is not None else 0, "Sequence", sequences)

    if final_val_mse is not None:
        df["Final_Validation_MSE"] = [final_val_mse] + [None] * (len(df) - 1)

    if final_val_mae is not None:
        df["Final_Validation_MAE"] = [final_val_mae] + [None] * (len(df) - 1)

    if final_val_r2 is not None:
        df["Final_Validation_R2"] = [final_val_r2] + [None] * (len(df) - 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = os.path.basename(model_name)
    path = os.path.join(output_dir, f"{safe}_evaluation_results_{timestamp}.xlsx") if output_dir \
        else f"{safe}_evaluation_results_{timestamp}.xlsx"

    df.to_excel(path, index=False)
    print(f"Results exported to {path}")


def export_baseline_comparison(baseline_results, dnabert_metrics, output_dir, timestamp):
    rows = [{
        "Model": "DNABERT-2 fine-tuned",
        "Validation_MSE": dnabert_metrics["Validation_MSE"],
        "Validation_MAE": dnabert_metrics["Validation_MAE"],
        "Validation_R2": dnabert_metrics["Validation_R2"]
    }]

    rows.extend(baseline_results)
    df = pd.DataFrame(rows)

    path = os.path.join(output_dir, f"model_baseline_comparison_{timestamp}.xlsx")
    df.to_excel(path, index=False)
    print(f"Baseline comparison exported to {path}")

    return df


def export_outliers_and_all_results(val_names, val_sequences, all_labels, all_preds, model_name, timestamp, output_dir):
    residuals = np.array(all_preds) - np.array(all_labels)
    std_res = np.std(residuals)
    abs_res = np.abs(residuals)
    outlier_mask = abs_res > 2 * std_res

    lines = ["Index\tName\tSequence\tActual\tPredicted\tResidual"]

    for i, is_out in enumerate(outlier_mask):
        if is_out:
            lines.append(f"{i}\t{val_names[i]}\t{val_sequences[i]}\t"
                         f"{all_labels[i]}\t{all_preds[i]}\t{residuals[i]}")

    n_best = max(1, int(0.2 * len(all_labels)))
    best_idx = np.argsort(abs_res)[:n_best]

    lines.append("\n# 20% Best Predicted Sequences")
    lines.append("Index\tName\tSequence\tActual\tPredicted\tResidual")

    for i in best_idx:
        lines.append(f"{i}\t{val_names[i]}\t{val_sequences[i]}\t"
                     f"{all_labels[i]}\t{all_preds[i]}\t{residuals[i]}")

    safe = os.path.basename(model_name)
    with open(os.path.join(output_dir, f"{safe}_outliers_{timestamp}.txt"), "w") as f:
        f.write("\n".join(lines))

    all_lines = ["Index\tName\tSequence\tActual\tPredicted\tResidual"]

    for i in range(len(all_labels)):
        all_lines.append(f"{i}\t{val_names[i]}\t{val_sequences[i]}\t"
                         f"{all_labels[i]}\t{all_preds[i]}\t{residuals[i]}")

    with open(os.path.join(output_dir, f"{safe}_all_results_{timestamp}.txt"), "w") as f:
        f.write("\n".join(all_lines))
    print("Outlier and all-results files written.")


def export_runinfo(model, model_name, data_filepath, batch_size, lr, weight_decay,
                   epochs, device, all_labels, all_preds, mse, mae, r2,
                   timestamp, output_dir, max_len=None, script_name=None,
                   model_description=None, val_mses=None, val_maes=None, val_r2s=None,
                   baseline_results=None, best_epoch=None, freeze_layers=0,
                   freeze_embeddings=False, trainable=None, total=None,
                   split_mode=None, label_log_base=None, label_clip_min=None):
    safe = os.path.basename(model_name)
    path = os.path.join(output_dir, f"{safe}_runinfo_{timestamp}.txt")

    with open(path, "w") as f:
        f.write(f"Model: {safe}\n")
        f.write(f"Training data file (full path): {data_filepath}\n")
        f.write(f"Training data file (basename): {os.path.basename(data_filepath)}\n")
        f.write(f"Python script used: {script_name}\n")
        f.write(f"Max sequence length (max_len): {max_len}\n")
        f.write(f"Split mode: {split_mode}\n")
        f.write(f"Label transform: log{label_log_base}(max(val, {label_clip_min}))\n")
        f.write(f"Final Validation MSE: {mse:.4f}\n")
        f.write(f"Final Validation MAE: {mae:.4f}\n")
        f.write(f"Final Validation R2: {r2:.4f}\n")
        if val_mses:
            f.write(f"Best Validation MSE: {np.min(val_mses):.4f}\n")
        if val_maes:
            f.write(f"Best Validation MAE: {np.min(val_maes):.4f}\n")
        if val_r2s:
            f.write(f"Best Validation R2: {np.max(val_r2s):.4f}\n")
        if best_epoch is not None:
            f.write(f"Best epoch (by R2): {best_epoch}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Weight decay: {weight_decay}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Loss function: torch.nn.MSELoss\n")
        f.write(f"Optimizer: AdamW\n")
        f.write(f"Scheduler: linear warmup/decay\n")
        f.write(f"Device: {device}\n")
        f.write(f"Validation set size: {len(all_labels)}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"freeze_layers: {freeze_layers}\n")
        f.write(f"freeze_embeddings: {freeze_embeddings}\n")
        if trainable is not None and total is not None:
            f.write(f"Trainable params: {trainable:,} / {total:,}\n")
        f.write(f"CNN baseline epochs: {cnn_epochs}\n")
        f.write(f"CNN baseline lr: {cnn_lr}\n")
        f.write(f"CNN baseline weight decay: {cnn_weight_decay}\n")
        f.write(f"CNN baseline dropout: {cnn_dropout}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"Transformers version: {transformers.__version__}\n")
        f.write(f"Pandas version: {pd.__version__}\n")
        f.write(f"Numpy version: {np.__version__}\n")
        if hasattr(model, "config") and hasattr(model.config, "to_json_string"):
            h = hashlib.md5(model.config.to_json_string().encode("utf-8")).hexdigest()
            f.write(f"Model config hash: {h}\n")
        if baseline_results:
            f.write("\nBaseline Results:\n")
            for r in baseline_results:
                f.write(f"{r['Model']}: MSE={r['Validation_MSE']:.4f}, "
                        f"MAE={r['Validation_MAE']:.4f}, R2={r['Validation_R2']:.4f}\n")
        if model_description:
            f.write("\nModel Description:\n" + model_description + "\n")
    print(f"Run info written to {path}")


def predict(model, tokenizer, sequences, device, max_len=130, save_attention_path=None):
    model.eval()
    all_attentions = []

    if isinstance(sequences, str):
        sequences = [sequences]

    cleaned = [clean_sequence(s) for s in sequences]

    encoding = tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        predictions = outputs.logits.view(-1).cpu().numpy()

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            all_attentions = [a.cpu().numpy() for a in outputs.attentions]
            if save_attention_path is not None:
                np.save(save_attention_path, np.array(all_attentions, dtype=object), allow_pickle=True)
                print(f"Attention weights saved to {save_attention_path}")

    return predictions, all_attentions


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_mode_str():
    return f"sheet split (train={train_sheets}, test={test_sheet})" \
        if use_sheet_split else f"unique-sequence split (val_size={val_size})"


def run_one_fold(fold_idx, test_sheet_arg, train_sheets_arg):
    global train_sheets, test_sheet
    train_sheets = train_sheets_arg
    test_sheet = test_sheet_arg

    print("=" * 80)
    print(f"[Fold {fold_idx}] test={test_sheet}, train={train_sheets}")
    print("=" * 80)

    set_global_seed(seed)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    model, tokenizer = load_model_and_tokenizer(model_name)
    model.to(device)

    trainable, total = apply_freeze(model, freeze_layers, freeze_embeddings)

    print(f"Model loaded: {model_name}")
    print(f"Device: {device}, batch={batch_size}, lr={lr}")
    print("Total trainable parameters:", count_model_parameters(model))

    (train_loader, val_loader,
     X_train, y_train, X_val, y_val) = prepare_data(
        data_filepath, tokenizer, batch_size=batch_size, max_len=max_len
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    loss_fn = torch.nn.MSELoss()

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print("######################################")
    print("Training model...")

    start_time = time.time()
    train_losses, val_mses, val_maes, val_r2s = [], [], [], []
    best_r2 = -float("inf")
    best_epoch = -1
    best_mse = float("inf")
    best_mae = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scheduler=scheduler,
            epoch=epoch,
            epochs=epochs
        )

        train_losses.append(train_loss)

        all_preds, all_labels, mse, mae, r2 = evaluate_model(model, val_loader, device)
        val_mses.append(mse)
        val_maes.append(mae)
        val_r2s.append(r2)

        print(f"Epoch {epoch+1}/{epochs}, loss={train_loss:.4f}, "
              f"MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_epoch = epoch + 1
            best_mse = mse
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    training_time = (time.time() - start_time) / 60
    print(f"Training time: {training_time:.2f} minutes")
    print(f"Best Validation R2: {best_r2:.4f} at epoch {best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(results_dir, exist_ok=True)
    output_dir_run = os.path.join(
        results_dir,
        f"fold{fold_idx}_{test_sheet}_{os.path.basename(model_name)}_{timestamp}"
    )
    os.makedirs(output_dir_run, exist_ok=True)

    export_training_metrics(
        train_losses,
        val_mses,
        val_maes,
        val_r2s,
        model_name,
        timestamp,
        output_dir_run
    )

    plot_training_metrics(
        train_losses,
        val_mses,
        val_maes,
        val_r2s,
        model_name,
        timestamp,
        output_dir_run
    )

    model.save_pretrained(output_dir_run)
    tokenizer.save_pretrained(output_dir_run)
    torch.save(model.state_dict(), os.path.join(output_dir_run, "pytorch_model.bin"))
    print(f"Model and tokenizer saved to {output_dir_run}")

    print("######################################")
    print("Evaluating model...")

    all_preds, all_labels, mse, mae, r2 = evaluate_model(model, val_loader, device)
    print(f"Validation MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    dnabert_metrics = {
        "Validation_MSE": mse,
        "Validation_MAE": mae,
        "Validation_R2": r2
    }

    plot_evaluation_results(all_preds, all_labels, os.path.basename(model_name), output_dir=output_dir_run)

    export_results_to_excel(
        all_preds,
        all_labels,
        model_name,
        final_val_mse=mse,
        final_val_mae=mae,
        final_val_r2=r2,
        sequences=X_val,
        names=[str(i) for i in range(len(X_val))],
        output_dir=output_dir_run
    )

    # Baselines
    print("######################################")
    print("Running baselines...")

    baseline_results = []

    baseline_results.append(
        run_kmer_ridge_baseline(
            X_train,
            y_train,
            X_val,
            y_val,
            k=6
        )
    )

    baseline_results.append(
        run_cnn_baseline(
            X_train,
            y_train,
            X_val,
            y_val,
            device=device,
            output_dir=output_dir_run,
            timestamp=timestamp,
            max_len=max_len,
            batch_size=batch_size,
            epochs=cnn_epochs,
            lr=cnn_lr,
            weight_decay=cnn_weight_decay,
            dropout=cnn_dropout
        )
    )

    comparison_df = export_baseline_comparison(
        baseline_results,
        dnabert_metrics,
        output_dir_run,
        timestamp
    )

    print("Model comparison:")
    print(comparison_df)

    val_names = [str(i) for i in range(len(X_val))]

    export_outliers_and_all_results(
        val_names,
        X_val,
        all_labels,
        all_preds,
        model_name,
        timestamp,
        output_dir_run
    )

    # runinfo
    export_runinfo(
        model,
        os.path.basename(model_name),
        data_filepath,
        batch_size,
        lr,
        weight_decay,
        epochs,
        device,
        all_labels,
        all_preds,
        mse,
        mae,
        r2,
        timestamp,
        output_dir_run,
        max_len=max_len,
        script_name=os.path.basename(__file__),
        model_description=(
            f"DNABERT-2 fine-tuned. fold={fold_idx}, test_sheet={test_sheet}, "
            f"train_sheets={train_sheets}. "
            f"freeze_layers={freeze_layers}, "
            f"freeze_embeddings={freeze_embeddings}, "
            f"trainable={trainable:,}/{total:,}. "
            f"Split: {split_mode_str()}. "
            f"Label: log{label_log_base}(max(val, {label_clip_min})). "
            f"Baselines: 6-mer Ridge + one-hot CNN."
        ),
        val_mses=val_mses, val_maes=val_maes, val_r2s=val_r2s,
        baseline_results=baseline_results, best_epoch=best_epoch,
        freeze_layers=freeze_layers, freeze_embeddings=freeze_embeddings,
        trainable=trainable, total=total, split_mode=split_mode_str(),
        label_log_base=label_log_base, label_clip_min=label_clip_min,
    )

    return {
        "fold": fold_idx,
        "test_sheet": test_sheet,
        "train_sheets": ",".join(train_sheets),
        "best_r2": best_r2,
        "best_mse": best_mse,
        "best_mae": best_mae,
        "best_epoch": best_epoch,
        "final_r2": r2,
        "final_mse": mse,
        "final_mae": mae,
        "output_dir": output_dir_run,
        "training_time": training_time,
    }


def main():
    fold_summaries = []

    for fold_idx, test_sheet_i in enumerate(all_sheets, start=1):
        train_sheets_i = [s for s in all_sheets if s != test_sheet_i]

        summary = run_one_fold(
            fold_idx=fold_idx,
            test_sheet_arg=test_sheet_i,
            train_sheets_arg=train_sheets_i,
        )
        fold_summaries.append(summary)

    print("\n" + "=" * 80)
    print("5-fold cross-validation summary")
    print("=" * 80)
    summary_df = pd.DataFrame(fold_summaries)
    print(summary_df)

    summary_path = os.path.join(
        results_dir,
        f"5fold_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
    summary_df.to_excel(summary_path, index=False)
    print(f"5-fold summary exported to {summary_path}")

    print(f"\nMean best R2:  {summary_df['best_r2'].mean():.4f} ± {summary_df['best_r2'].std():.4f}")
    print(f"Mean best MSE: {summary_df['best_mse'].mean():.4f} ± {summary_df['best_mse'].std():.4f}")
    print(f"Mean best MAE: {summary_df['best_mae'].mean():.4f} ± {summary_df['best_mae'].std():.4f}")


if __name__ == "__main__":
    main()
