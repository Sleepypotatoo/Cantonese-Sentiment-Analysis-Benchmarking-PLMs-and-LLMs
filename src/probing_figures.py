import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

df = pd.read_csv("./results/probing/probing_final_results.csv")

feature_order = [
    "词汇层_一词多义",
    "词汇层_极性修饰词",
    "句法层_否定句式",
    "句法层_比较句式",
    "语用层_句末语气词",
    "语用层_中英夹杂",
    "语用层_粤式反讽"
]

short_en = {
    "词汇层_一词多义": "Polysemy",
    "词汇层_极性修饰词": "Polarity Modifiers",
    "句法层_否定句式": "Negative Structure",
    "句法层_比较句式": "Comparative Structure",
    "语用层_句末语气词": "SFP",
    "语用层_中英夹杂": "Code-mix",
    "语用层_粤式反讽": "Irony"
}

models = ["2-Class_Model", "3-Class_Model", "5-Class_Model"]
model_output_names = ["2class", "3class", "5class"]

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

for model, out_name in zip(models, model_output_names):
    sub = df[df["Model"] == model]
    pivot = sub.pivot(index="Task", columns="Layer", values="Accuracy")
    pivot = pivot.reindex(feature_order)
    pivot = pivot.reindex(columns=range(13))
    pivot.index = [short_en[t] for t in pivot.index]

    # 略微放大热力图尺寸
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=150)
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f",
                cmap="YlGnBu", cbar=False,
                annot_kws={'size': 5},      # 缩小数字
                linewidths=0.5,             # 稍微加粗格子线，帮助区分
                vmin=0.40, vmax=0.85)

    ax.set_title("")
    ax.set_xlabel("Layer", fontsize=8)
    ax.set_ylabel("")
    ax.set_xticks(range(13))
    ax.set_xticklabels(range(13), rotation=0, fontsize=7)

    plt.tight_layout(pad=0.5)
    plt.savefig(f"./results/probing/heatmap_{out_name}.png", dpi=300, bbox_inches='tight')
    plt.close()