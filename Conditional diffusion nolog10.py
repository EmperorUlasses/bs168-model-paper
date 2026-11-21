"""
Preprocessing script (data label binning + backbone feature extraction) 
for dataset preparation of the subsequent conditional‑diffusion model.
"""
import pandas as pd
import numpy as np


file_path = 'data/predict_results.xlsx'
sheet_name = 'Sheet1'

df = pd.read_excel(file_path, sheet_name=sheet_name)


print(f"原始数据行数: {len(df)}")
print(f"列名: {df.columns.tolist()}")
print(f"Average GFP expression (log10) 范围: {df['Average GFP expression'].min():.4f} - {df['Average GFP expression'].max():.4f}")


log10_values = df['Average GFP expression'].dropna().values

print(f"log10值范围: {log10_values.min():.4f} - {log10_values.max():.4f}")


bin_edges = np.percentile(log10_values, np.linspace(0, 100, 10))

print(f"\n分箱边界 (log10值，输入已是 log10):")
for i, edge in enumerate(bin_edges):
    print(f"  边界 {i}: {edge:.4f}")


result_df = pd.DataFrame()

result_df[''] = df['SEM ID'].values

result_df['raw_sequence'] = df['Sequence'].values

labels = list(range(1, 10))
result_df['component'] = pd.cut(
    df['Average GFP expression'].values,   
    bins=bin_edges,
    labels=labels,
    include_lowest=True
)


if result_df['component'].isna().any():
    print(f"\n警告: 有 {result_df['component'].isna().sum()} 行数据无法分箱")


def get_tier(comp):
    if pd.isna(comp): return None
    comp = int(comp)
    if 1 <= comp <= 3: return 'Low'
    elif 4 <= comp <= 6: return 'Med'
    elif 7 <= comp <= 9: return 'High'
    return None

result_df['tier'] = result_df['component'].apply(get_tier)

P35 = 'TTGACA'
P10 = 'TATAAT'
SD  = 'AGGAGG'


def get_frame(seq):
    seq = str(seq).upper()

    if len(seq) != 95:
        return 'F_none'
        
    frames = {
        'F0': {'35': 54, '10': 31, 'sd': 14},
        'F1': {'35': 64, '10': 41, 'sd': 14},
        'F2': {'35': 74, '10': 51, 'sd': 14},
        'F3': {'35': 51, '10': 28, 'sd': 11},
        'F4': {'35': 61, '10': 38, 'sd': 11},
        'F5': {'35': 71, '10': 48, 'sd': 11}
    }

    for frame_name, coords in frames.items():
        idx_35 = 95 - coords['35']
        idx_10 = 95 - coords['10']
        idx_sd = 95 - coords['sd']
        
        sub_35 = seq[idx_35 : idx_35 + 6]
        sub_10 = seq[idx_10 : idx_10 + 6]
        sub_sd = seq[idx_sd : idx_sd + 6]
        
        if sub_35 == P35 and sub_10 == P10 and sub_sd == SD:
            return frame_name

    return 'F_none'

print("正在计算骨架(Frame)分类...")
result_df['frame'] = result_df['raw_sequence'].apply(get_frame)

print("\n各骨架(frame)的数量分布:")
print(result_df['frame'].value_counts())

print("\n各档(tier)的数量分布:")
print(result_df['tier'].value_counts())

range_mapping = []
for i in range(1, 10):
    mask = result_df['component'] == i
    if mask.any():

        vals = df.loc[mask, 'Average GFP expression']
        
        range_mapping.append({
            'component': i,
            'count': mask.sum(),
            'log10_min': vals.min(),
            'log10_max': vals.max(),
            'raw_min': 10 ** vals.min(),   
            'raw_max': 10 ** vals.max(),
        })
    else:
        range_mapping.append({
            'component': i,
            'count': 0,
            'log10_min': None,
            'log10_max': None,
            'raw_min': None,
            'raw_max': None,
        })

real_max_log10 = log10_values.max()

range_mapping.append({
    'component': 10,
    'count': 0,
    'log10_min': real_max_log10,          
    'log10_max': None,                    
    'raw_min': 10 ** real_max_log10,
    'raw_max': None,
})

range_df = pd.DataFrame(range_mapping)

print(f"\n各区间对应的log10值和原始值范围:")
print(range_df.to_string(index=False))

output_file_main = 'data/95bp_数据集合并_分箱结果.csv'
result_df.to_csv(output_file_main, index=False, encoding='utf-8-sig')

output_file_range = 'data/95bp_数据集合并_范围对照表.xlsx'
range_df.to_excel(output_file_range, index=False)

print(f"\n结果已保存至:")
print(f"  - 主数据文件: {output_file_main} (CSV格式)")
print(f"    (列: index, raw_sequence, component, tier, frame)")
print(f"  - 范围对照表文件: {output_file_range} (Excel格式)")
print(f"\n主数据行数: {len(result_df)}")