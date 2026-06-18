import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("project/results/probing_full/all_layers_results.csv")
df['Layer'] = df['Layer'].astype(int)

# 计算总体 Precision
df['Avg_Precision'] = (df['Pos_P'] + df['Neg_P']) / 2

# 英文任务名映射（顺序按论文层级）
task_mapping = {
    '词汇层_一词多义': 'Polysemy',
    '词汇层_极性修饰词': 'Intensity',
    '句法层_比较句式': 'Comparison',
    '句法层_否定句式': 'Negation',
    '语用层_句末语气词': 'UFPs',
    '语用层_中英夹杂': 'Code-Mixing',
    '语用层_粤式反讽': 'Irony'
}
task_order = list(task_mapping.values())  # 英文顺序

# 排除 Ollama（只有一层，不适合热力图）
models = [m for m in df['Model'].unique() if m != 'Ollama_Embed']

for model_name in models:
    sub = df[df['Model'] == model_name]
    pivot = sub.pivot(index='Task', columns='Layer', values='Avg_Precision')
    
    # 将中文索引替换为英文
    pivot = pivot.rename(index=task_mapping)
    # 按英文顺序重排行
    pivot = pivot.reindex(task_order)
    
    # 可选：删除全部为0的层（如果有多余空列）
    # pivot = pivot.loc[:, (pivot != 0).any(axis=0)]

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Avg Precision'},
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 6}
    )
    plt.title(f'{model_name} - Average Precision across Layers', fontsize=14)
    plt.xlabel('Layer', fontsize=12)
    plt.xticks(fontsize=8)            # 横轴刻度字体
    plt.yticks(fontsize=8)            # 纵轴刻度字体
    plt.ylabel('Linguistic Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'test.jsonl/project/results/probing_full/heatmap_{model_name}_avg_precision_en.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: heatmap_{model_name}_avg_precision_en.png")
