# -*- coding: utf-8 -*-
"""
Full combinatorial enumeration over Top‑15 sites: C(n,5)·3^5 (k=5, no pruning; test all 3 non‑reference bases for each position)
Record the global optimum sequence
"""

import os, glob, itertools, warnings, math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from tqdm import tqdm

BEST_XLSX = 'push 10 try/best_sequence.xlsx'
SHEET_SEQ = 'best_promoter_95bp'
SHEET_FRAMES = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5']
OUT_CSV = 'push 10 try/push10_enum_results.csv'
BEST_PER_K_CSV = 'push 10 try/push10_enum_best_per_k.csv'
EVAL_INPUT_CSV = 'push 10 try/push10_enum_for_eval.csv'

FRAMES_SD = {'F0':14,'F1':14,'F2':14,'F3':11,'F4':11,'F5':11}
EDGES = np.array([3.0689, 3.5149, 3.5950, 3.7189, 3.8408, 3.9377,
                  4.0110, 4.0878, 4.2171, 4.5228])
TIER10_THR = EDGES[9]

_model = None
_tokenizer = None

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    global _model, _tokenizer
    if _model is not None: return
    print("\n[模型加载] 仅加载 fold3 DNABERT‑2...")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_new", "results_RNA")
    cands = glob.glob(os.path.join(root, f"fold3_*"))
    d = sorted(cands)[-1]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
        cfg = BertConfig.from_pretrained(d); cfg.num_labels = 1
        m = AutoModelForSequenceClassification.from_pretrained(d, config=cfg, trust_remote_code=True)
    m.to(_device).eval()
    _model = m
    _tokenizer = tok
    print("[模型加载] fold3 加载完成")


def predict_log10(seqs, batch_size=512):
    if _model is None: load_models()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = [str(s).upper() for s in seqs[i:i+batch_size]]
            enc = _tokenizer(batch, padding=True, truncation=True, max_length=130, return_tensors="pt")
            p = _model(input_ids=enc["input_ids"].to(_device),
                       attention_mask=enc["attention_mask"].to(_device)).logits.view(-1).float().cpu().numpy()
            out.append(p)
    return np.concatenate(out)


def main():
    df_seq = pd.read_excel(BEST_XLSX, sheet_name=SHEET_SEQ)
    df_seq.columns = [c.strip() for c in df_seq.columns]

    pos_map = {}
    for sh in SHEET_FRAMES:
        dfp = pd.read_excel(BEST_XLSX, sheet_name=sh)
        dfp.columns = [c.strip() for c in dfp.columns]
        pos_map[sh] = [int(p) for p in dfp['abs'].tolist()]
        n = len(pos_map[sh])
        tc = math.comb(n, 5) * 3**5 if n >= 5 else 0
        print(f"{sh}: 位点数 {n}，abs {sorted(pos_map[sh])}，每条 {tc:,} 组合")

    results, best_per_k_rows, eval_rows = [], [], []
    finished_ids = set()
    if os.path.exists(OUT_CSV):
        try:
            df_prev = pd.read_csv(OUT_CSV, encoding='utf-8-sig')
            if 'sem_id' in df_prev.columns:
                finished_ids = set(df_prev['sem_id'].astype(str).tolist())
                results = df_prev.to_dict('records')
                print(f"[断点续传] 已完成 {len(finished_ids)} 条")
        except Exception as e:
            print(f"[断点续传] 失败: {e}")

    BATCH_GEN = 100_000

    for _, row in tqdm(df_seq.iterrows(), total=len(df_seq), desc="全量枚举"):
        sem_id = str(row['SEM ID'])
        if sem_id in finished_ids: continue
        orig_pred = float(row['GSR predect log10'])
        seq = str(row['Sequence']).upper().strip()
        frame = str(row['骨架']).strip().upper()
        if len(seq) != 95 or frame not in FRAMES_SD:
            print(f"  跳过 {sem_id}"); continue
        sd_start, sd_end = 95 - FRAMES_SD[frame], 94

        p_orig = float(predict_log10([seq])[0])

        frame_sites = pos_map.get(frame, [])
        if not frame_sites:
            print(f"  跳过 {sem_id}：骨架 {frame} 无位点表"); continue
        enum_abs = [p for p in frame_sites if not (sd_start <= p <= sd_end)]
        if not enum_abs:
            print(f"  跳过 {sem_id}：所有位点都在骨架区"); continue
        print(f"  {sem_id} frame={frame} 枚举点={len(enum_abs)}/{len(frame_sites)}")

        options_per_pos = {p: [b for b in 'ACGT' if b != seq[p]] for p in enum_abs}

        best_overall_score, best_overall_seq = p_orig, seq
        best_per_k = {5: (p_orig, seq)} if len(enum_abs) >= 5 else {}

        for k in ([5] if len(enum_abs) >= 5 else []):
            pos_combos = list(itertools.combinations(enum_abs, k))
            n_combo_k = len(pos_combos) * (3 ** k)
            best_k_score, best_k_seq = p_orig, seq
            buf = []
            for combo in pos_combos:
                for bp_combo in itertools.product(*[options_per_pos[p] for p in combo]):
                    t = list(seq)
                    for p, b in zip(combo, bp_combo):
                        t[p] = b
                    buf.append(''.join(t))
                    if len(buf) >= BATCH_GEN:
                        sc = predict_log10(buf)
                        j = int(np.argmax(sc)); s = float(sc[j])
                        if s > best_k_score:
                            best_k_score, best_k_seq = s, buf[j]
                        buf = []
            if buf:
                sc = predict_log10(buf)
                j = int(np.argmax(sc)); s = float(sc[j])
                if s > best_k_score:
                    best_k_score, best_k_seq = s, buf[j]

            best_per_k[k] = (best_k_score, best_k_seq)
            if best_k_score > best_overall_score:
                best_overall_score, best_overall_seq = best_k_score, best_k_seq

            best_per_k_rows.append({
                'sem_id': sem_id, 'frame': frame, 'k': k,
                'n_combo_k': n_combo_k,
                'pred_orig': p_orig, 'pred_best_k': best_k_score,
                'delta_k': best_k_score - p_orig,
                'best_seq_k': best_k_seq,
                'reach_tier10': best_k_score > TIER10_THR,
            })
            pd.DataFrame(best_per_k_rows).to_csv(BEST_PER_K_CSV, index=False, encoding='utf-8-sig')
            print(f"  {sem_id} k={k}: combos={n_combo_k:,}, best={best_k_score:.4f}, Δ={best_k_score-p_orig:+.4f}")

        results.append({
            'sem_id': sem_id, 'frame': frame,
            'n_enum_sites': len(enum_abs),
            'pred_orig_given': orig_pred,
            'pred_orig_eval': p_orig,
            'pred_best': best_overall_score,
            'delta_vs_eval': best_overall_score - p_orig,
            'delta_vs_given': best_overall_score - orig_pred,
            'best_k': max(best_per_k, key=lambda k: best_per_k[k][0]) if best_per_k else 0,
            'reach_tier10': best_overall_score > TIER10_THR,
            'orig_seq': seq, 'best_seq': best_overall_seq,
        })
        eval_rows.append({'sequence': best_overall_seq, 'orig_sequence': seq,
                          'sem_id': sem_id, 'pred_best': best_overall_score})
        pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
        pd.DataFrame(eval_rows).to_csv(EVAL_INPUT_CSV, index=False, encoding='utf-8-sig')

    pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(best_per_k_rows).to_csv(BEST_PER_K_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(eval_rows).to_csv(EVAL_INPUT_CSV, index=False, encoding='utf-8-sig')

    df_res = pd.DataFrame(results)
    if len(df_res):
        print("\n" + "="*60)
        print(f"共处理 {len(df_res)} 条")
        print(f"  原序列平均分 (重评): {df_res['pred_orig_eval'].mean():.4f}")
        print(f"  全局最优平均分: {df_res['pred_best'].mean():.4f}")
        print(f"  平均 Δ: {df_res['delta_vs_eval'].mean():+.4f}")
        print(f"  提升条数: {(df_res['delta_vs_eval'] > 0).sum()} / {len(df_res)}")
        print(f"  达到 10 档 (>4.5228): {df_res['reach_tier10'].sum()} / {len(df_res)}")
        print("="*60)
        df_bk = pd.DataFrame(best_per_k_rows)
        if len(df_bk):
            print("\n各 k 的平均/最大最优分:")
            print(df_bk.groupby('k')['pred_best_k'].agg(['mean','max']).round(4))

if __name__ == '__main__':
    main()