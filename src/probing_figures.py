import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("results/all_layers_results_m+can.csv")
df['Layer'] = df['Layer'].astype(int)


# 英文任务名映射（顺序按论文层级）
task_mapping = {
    '词汇层_一词多义': 'False Friends',
    '词汇层_极性修饰词': 'Intensity',
    '句法层_比较句式': 'Comparison',
    '句法层_否定句式': 'Negation',
    '语用层_句末语气词': 'UFPs',
    '语用层_中英夹杂': 'Code-Mixing',
    '语用层_粤式反讽': 'Irony'
}
task_order = list(task_mapping.values())

# 排除 Ollama（只有一层，不适合热力图）
models = [m for m in df['Model'].unique() if m != 'Ollama_Embed']

for model_name in models:
    sub = df[df['Model'] == model_name]
    # 直接用 Macro_P 列
    pivot = sub.pivot(index='Task', columns='Layer', values='Macro_P')
    
    # 将中文索引替换为英文
    pivot = pivot.rename(index=task_mapping)
    # 按英文顺序重排行
    pivot = pivot.reindex(task_order)

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
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel('Linguistic Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'test.jsonl/project/results/probing_full/heatmap_{model_name}_avg_precision_en.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: heatmap_{model_name}_avg_precision_en.png")
