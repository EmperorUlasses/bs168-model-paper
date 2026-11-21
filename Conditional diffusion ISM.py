# -*- coding: utf-8 -*-
"""
Multi‑base cooperative directed fine‑tuning based on Top‑10 sensitive sites (SD‑removed)

The push‑10 script has completed pushing to 10 intensity classes. 
Pipeline of this script:
  1. Flatten large‑size classes (Low/Med/High) → update df_master
  2. Supplement class‑1 and class‑9 (Low/High) → update df_master

Output final results in a unified manner.
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
import glob
import json
import itertools
import pandas as pd
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from tqdm import tqdm


BATCH_GEN = 100_000   

INPUT_CSV = 'Conditional diffusion/checked_generated_sequences.csv'
OUT_CSV = 'Conditional diffusion/tuned_results.csv'
EVAL_INPUT_CSV = 'Conditional diffusion/tuned_for_eval.csv'

META_JSON = 'TOP10_SD/交付_TOP10_去SD/k10_meta_noSD.json'

EDGES = np.array([3.0689, 3.5149, 3.5950, 3.7189, 3.8408, 3.9377,
                  4.0110, 4.0878, 4.2171, 4.5228])

FRAMES = {
    'F0': {'35': 54, '10': 31, 'sd': 14}, 'F1': {'35': 64, '10': 41, 'sd': 14},
    'F2': {'35': 74, '10': 51, 'sd': 14}, 'F3': {'35': 51, '10': 28, 'sd': 11},
    'F4': {'35': 61, '10': 38, 'sd': 11}, 'F5': {'35': 71, '10': 48, 'sd': 11}
}


_model = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    global _model, _tokenizer
    if _model is not None:
        return
    print("\n[模型加载] 仅加载 fold3 DNABERT‑2...")
    results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_new", "results_RNA")
    cands = glob.glob(os.path.join(results_root, f"fold3_*"))
    if not cands:
        raise FileNotFoundError("找不到fold3模型文件夹，请检查路径 results_new/results_RNA")
    model_dir = sorted(cands)[-1]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        cfg = BertConfig.from_pretrained(model_dir)
        cfg.num_labels = 1
        m = AutoModelForSequenceClassification.from_pretrained(model_dir, config=cfg, trust_remote_code=True)
    m.to(_device).eval()
    _model = m
    _tokenizer = tok
    print("[模型加载] fold3 加载完成")

def predict_log10(seqs, batch_size=512):
    if _model is None:
        load_models()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = [str(s).upper() for s in seqs[i:i+batch_size]]
            enc = _tokenizer(batch, padding=True, truncation=True, max_length=130, return_tensors="pt")
            p = _model(input_ids=enc["input_ids"].to(_device),
                       attention_mask=enc["attention_mask"].to(_device)).logits.view(-1).float().cpu().numpy()
            out.append(p)
    return np.concatenate(out)


def load_sites_from_meta(tier_name):
    if not os.path.exists(META_JSON):
        raise FileNotFoundError(f"找不到预计算位点文件: {META_JSON}")
    with open(META_JSON, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return [int(x) for x in meta['sites'][tier_name]]

# ================= 组合枚举微调：k=1‑5，找到满足条件即早停 =================
def exhaustive_tune(seq, base_p, rel_positions, sd_idx, mode, target_thr=None, target_val=None):
    abs_positions = [sd_idx + p for p in rel_positions if 0 <= (sd_idx + p) < len(seq)]
    # -------- push10逻辑：过滤SD骨架区间内位点，SD区内位点不做突变 --------
    sd_start = sd_idx
    sd_end = 94
    enum_abs = [p for p in abs_positions if not (sd_start <= p <= sd_end)]
    if len(enum_abs) == 0:
        return False, seq, base_p, []

    seq_list = list(seq)
    options_per_pos = {p: [b for b in 'ACGT' if b != seq[p]] for p in enum_abs}

    max_k = min(len(enum_abs), 5)   # 只枚举 1~5位点突变
    k_iter = tqdm(range(1, max_k+1), desc=f"枚举 k (p0={base_p:.3f})", leave=False, ncols=100)

    for k in k_iter:
        k_iter.set_description(f"k={k} (p0={base_p:.3f})")
        pos_combos = list(itertools.combinations(enum_abs, k))

        buf = []
        buf_meta = []
        k_candidates = []  

        for combo in pos_combos:
            bp_opts = [options_per_pos[p] for p in combo]
            for bp_combo in itertools.product(*bp_opts):
                variant = seq_list[:]
                for pos, bp in zip(combo, bp_combo):
                    variant[pos] = bp
                var_seq = ''.join(variant)
                buf.append(var_seq)
                buf_meta.append((combo, bp_combo))

                if len(buf) >= BATCH_GEN:
                    preds = predict_log10(buf)
                    for sc, s_seq, meta in zip(preds, buf, buf_meta):
                        k_candidates.append((float(sc), s_seq, meta[0], meta[1]))
                    buf.clear()
                    buf_meta.clear()

        if len(buf) > 0:
            preds = predict_log10(buf)
            for sc, s_seq, meta in zip(preds, buf, buf_meta):
                k_candidates.append((float(sc), s_seq, meta[0], meta[1]))
            buf.clear()
            buf_meta.clear()

        hit_list = []
        for score, mutseq, combo, bpcombo in k_candidates:
            ok = False
            if mode == 'up':
                ok = (score > target_thr)
            elif mode == 'down':
                ok = (score < target_thr)
            elif mode == 'to':
                ok = True 
            if ok:
                hit_list.append({"score":score, "seq":mutseq, "combo":combo, "bpcombo":bpcombo})

        if len(hit_list) > 0:
            if mode == 'up':
                best_entry = max(hit_list, key=lambda x:x["score"])
            elif mode == 'down':
                best_entry = min(hit_list, key=lambda x:x["score"])
            elif mode == 'to':
                best_entry = min(hit_list, key=lambda x: abs(x["score"] - target_val))

            log = []
            for pos, bp in zip(best_entry["combo"], best_entry["bpcombo"]):
                log.append({
                    'step': k,
                    'pos_0based': pos,
                    'pos_95based': 95 - pos,
                    'from_to': seq[pos] + '>' + bp,
                    'pred': float(best_entry["score"])
                })
            k_iter.close()
            return True, best_entry["seq"], float(best_entry["score"]), log

    k_iter.close()
    return False, seq, base_p, []


def main():
    print("加载数据与预计算位点...")
    df_master = pd.read_csv(INPUT_CSV)
    df_master = df_master[df_master['pass_dedup'] == True].copy()
    
    high_sites = load_sites_from_meta('High')
    low_sites = load_sites_from_meta('Low')
    med_sites = load_sites_from_meta('Med')
    
    results = []
    all_logs = []
            
    log_path = 'Conditional diffusion/tune_step_log.csv'
    if os.path.exists(log_path):
        try:
            df_logs_prev = pd.read_csv(log_path, encoding='utf-8-sig')
            if 'task' in df_logs_prev.columns and 'to_tier10' in df_logs_prev['task'].unique():
                all_logs = df_logs_prev.to_dict('records')
        except:
            pass

    def process_seeds(seeds_list, phase_name):
        if len(seeds_list) > 0:
            current_task_name = seeds_list[0]['task']
            if any(r.get('task') == current_task_name for r in results):
                print(f"\n>> {phase_name} 已在上次运行中完成，自动跳过。")
                return
            
        print(f"\n>> 开始执行：{phase_name}（共 {len(seeds_list)} 条）")
        
        finished_idx = {r['seed_idx'] for r in results if r.get('task') == current_task_name}
        
        for s in tqdm(seeds_list, desc=f"{phase_name}"):
            if s['idx'] in finished_idx:
                continue  
                
            seq = s['seq']
            p0 = s['p0']
            coords = FRAMES.get(f"F{s['frame']}")
            if not coords:
                continue

            if len(seq) != 95:
                print(f"跳过异常长度序列 {s['idx']}，长度: {len(seq)}")
                continue

            sd_idx = 95 - coords['sd']
            
            success, tuned_seq, final_p, log = exhaustive_tune(
                seq, p0, s['sites'], sd_idx, s['mode'], target_thr=s['thr'], target_val=s.get('target')
            )
            
            if s['mode'] == 'to':
                orig_bin = s['orig_bin']
                in_bin = (final_p >= EDGES[orig_bin-1]) and (final_p <= EDGES[orig_bin])
                close_enough = abs(final_p - s['target']) < 0.05
                success = in_bin and close_enough
                
            final_seq = tuned_seq if success else seq
            n_edit = len(log) if success else 0
                
            results.append({
                'task': s['task'], 'seed_idx': s['idx'], 
                'pred_before': p0, 'pred_after': final_p, 
                'success': success, 'n_edit': n_edit,
                'seed_seq': seq, 'final_seq': final_seq,
                'tuned_seq': tuned_seq
            })
            
            if success:
                for entry in log:
                    entry['seed_idx'] = s['idx']
                    entry['task'] = s['task']
                    all_logs.append(entry)
                    
                df_master.at[s['idx'], 'sequence'] = final_seq
                df_master.at[s['idx'], 'predicted_log10'] = final_p

                if s['mode'] == 'up' and final_p > s['thr']:
                    new_bin = np.digitize(final_p, EDGES[1:-1]) + 1
                    df_master.at[s['idx'], 'predicted_component'] = new_bin
                elif s['mode'] == 'down' and final_p < s['thr']:
                    new_bin = np.digitize(final_p, EDGES[1:-1]) + 1
                    df_master.at[s['idx'], 'predicted_component'] = new_bin

            pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
            if all_logs:
                pd.DataFrame(all_logs).to_csv('Conditional diffusion/tune_step_log.csv', index=False, encoding='utf-8-sig')

    print("\n>> 正在扫描大数量档（> 300 条）...")
    seeds_A = []
    for b in range(1, 10):
        sub = df_master[df_master['predicted_component'] == b]
        if len(sub) < 300:
            continue

        if b <= 3:
            sites_for_b = low_sites
        elif b <= 6:
            sites_for_b = med_sites
        else:
            sites_for_b = high_sites

        lo, hi = EDGES[b-1], EDGES[b]
        edges3 = np.linspace(lo, hi, 4)
        sub_copied = sub.copy()
        sub_copied['_sub'] = np.clip(np.digitize(sub_copied['predicted_log10'], edges3[1:-1]), 0, 2)
        cnt_raw = sub_copied['_sub'].value_counts().to_dict()

        cnt = {s: cnt_raw.get(s, 0) for s in [0,1,2]}
        total = len(sub_copied)
        avg = total // 3

        overflow = dict()   
        gap = dict()       
        for s in [0,1,2]:
            diff = cnt[s] - avg
            if diff > 0:
                overflow[s] = diff
            elif diff < 0:
                gap[s] = -diff

        if len(overflow) == 0 or len(gap) == 0:
            continue

        print(f"  -> 档{b} total={total}, avg={avg}；cnt[0,1,2]={cnt[0]},{cnt[1]},{cnt[2]}")
        print(f"     overflow(sub:num): {overflow} ; gap(sub:num): {gap}")

        for src_sub, out_num in overflow.items():
            df_src = sub_copied[sub_copied['_sub'] == src_sub].copy()
            remain_src = df_src.copy()
            for dst_sub, need_num in gap.items():
                if out_num <= 0 or need_num <=0 or len(remain_src) <=0:
                    continue
                take = min(out_num, need_num, len(remain_src))

                pick = remain_src.sample(n=take, random_state=42)

                target_mid = (edges3[dst_sub] + edges3[dst_sub+1]) / 2

                print(f"        src_sub={src_sub} → dst_sub={dst_sub}，计划迁移 {take} 条，target_mid={target_mid:.4f}")

                for idx, row in pick.iterrows():
                    seeds_A.append({
                        'task': f'to_inner_{b}', 'idx': idx, 'seq': row['sequence'], 'p0': row['predicted_log10'],
                        'frame': row['frame'], 'mode': 'to', 'thr': None, 'sites': sites_for_b, 'target': target_mid,
                        'orig_bin': b
                    })

                remain_src = remain_src.drop(pick.index)
                out_num -= take

    process_seeds(seeds_A, "任务 A：大数量档摊平")


    tier2 = df_master[df_master['predicted_component'] == 2]
    tier2_lowest = tier2.nsmallest(10, 'predicted_log10')
    seeds_C1 = []
    for idx, row in tier2_lowest.iterrows():
        seeds_C1.append({
            'task': 'to_tier1', 'idx': idx, 'seq': row['sequence'], 'p0': row['predicted_log10'],
            'frame': row['frame'], 'mode': 'down', 'thr': EDGES[1], 'sites': low_sites, 'target': None
        })
    tier8 = df_master[df_master['predicted_component'] == 8]
    tier8_highest = tier8.nlargest(10, 'predicted_log10')
    seeds_C9 = []
    for idx, row in tier8_highest.iterrows():
        seeds_C9.append({
            'task': 'to_tier9', 'idx': idx, 'seq': row['sequence'], 'p0': row['predicted_log10'],
            'frame': row['frame'], 'mode': 'up', 'thr': EDGES[8], 'sites': high_sites, 'target': None
        })
    
    process_seeds(seeds_C1 + seeds_C9, "任务 C：补档 1 与 9")

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(all_logs).to_csv('Conditional diffusion/tune_step_log.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("全流程微调完成，任务统计：")
    for task in sorted(df_res['task'].unique()):
        sub = df_res[df_res['task'] == task]
        n_ok = sub['success'].sum()
        print(f"  -> {task}: 尝试 {len(sub)} 条，成功 {n_ok} 条")
    print("="*50)
    
    print("\n正在生成 evaluate.py 专用文件...")
    eval_rows = []
    for r in results:
        orig_frame = df_master.at[r['seed_idx'], 'frame']
        orig_class = df_master.at[r['seed_idx'], 'class']
        eval_rows.append({
            'sequence': r['final_seq'],
            'frame': int(orig_frame),
            'class': int(orig_class)
        })
    pd.DataFrame(eval_rows).to_csv(EVAL_INPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"已生成 {EVAL_INPUT_CSV}，共 {len(eval_rows)} 条，可以直接用 evaluate.py 过滤")

if __name__ == '__main__':
    main()