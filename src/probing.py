import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import csv

# ----------------------- 配置 -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_NAME = "manueltonneau/bert-base-cantonese"   # 基础粤语BERT（用于tokenizer）
MAX_LEN = 128
PROBING_BATCH_SIZE = 32
PROBING_EPOCHS = 5
PROBING_LR = 1e-3

# 训练好的分类模型路径（请根据实际情况修改）
FINETUNED_MODELS = {
    "2-Class_Model": "./saved_models/hktv_2emos_cantonese_model",   # 请替换为你的2分类模型目录
    "3-Class_Model": "./saved_models/hktv_3emos_cantonese_model",   # 3分类模型目录
    "5-Class_Model": "./saved_models/hktv_5emos_cantonese_model",   # 5分类模型目录
}

PROBING_TASKS = {
    "词汇层_一词多义": "./probing_data/词汇层_一词多义_train.csv",
    "词汇层_极性修饰词": "./probing_data/词汇层_极性修饰词_train.csv",
    "句法层_比较句式": "./probing_data/句法层_比较句式_train.csv",
    "句法层_否定句式": "./probing_data/句法层_否定句式_train.csv",
    "语用层_句末语气词": "./probing_data/语用层_句末语气词_train.csv",
    "语用层_中英夹杂": "./probing_data/语用层_中英夹杂_外来词_train.csv",
    "语用层_粤式反讽": "./probing_data/语用层_粤式反讽_train.csv",
}

OUTPUT_CSV = "./results/probing/probing_final_results.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


# ----------------------- 数据集 -----------------------
class ProbingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ----------------------- 线性探针 -----------------------
class LinearProbe(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.classifier(x)


def train_probe_for_layer(probe, train_loader, device):
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBING_LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(PROBING_EPOCHS):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = probe(input_ids, attention_mask)  # 实际需要接收 hidden states，下面会改
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


def evaluate_probe(probe, test_loader, device):
    probe.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = probe(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)


def get_hidden_states(encoder, input_ids, attention_mask, layer_idx):
    """返回 encoder 第 layer_idx 层的 [CLS] 向量 (batch, hidden_dim)"""
    outputs = encoder(
        input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
    hidden_states = outputs.hidden_states  # tuple, length = num_layers+1 (embedding + hidden)
    cls_emb = hidden_states[layer_idx][:, 0, :]  # (batch, 768)
    return cls_emb


def run_probing_for_model(encoder, tokenizer, task_csv, device, model_name, task_name):
    """对单个模型的一个探测任务，返回每层准确率列表"""
    # 读取数据
    df = read_probing_csv(task_csv)
    if df.empty:
        print(f"  跳过空数据集: {task_csv}")
        return None

    df = df.dropna(subset=["text"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([0, 1])]
    if len(df) == 0:
        print(f"  无有效二分类样本: {task_csv}")
        return None

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_dataset = ProbingDataset(
        train_df["text"].values, train_df["label"].values, tokenizer, MAX_LEN
    )
    test_dataset = ProbingDataset(
        test_df["text"].values, test_df["label"].values, tokenizer, MAX_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=PROBING_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=PROBING_BATCH_SIZE, shuffle=False)

    num_layers = encoder.config.num_hidden_layers + 1  # 0: embedding, 1..12: layers
    layer_accuracies = []

    for layer_idx in range(num_layers):
        print(f"    Layer {layer_idx} ...")

        # 准备当前层的所有训练/测试 hidden states
        train_hidden = []
        train_labels = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with torch.no_grad():
                cls_vec = get_hidden_states(encoder, input_ids, attn_mask, layer_idx)
            train_hidden.append(cls_vec.cpu())
            train_labels.append(labels.cpu())

        train_hidden = torch.cat(train_hidden, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        test_hidden = []
        test_labels = []
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with torch.no_grad():
                cls_vec = get_hidden_states(encoder, input_ids, attn_mask, layer_idx)
            test_hidden.append(cls_vec.cpu())
            test_labels.append(labels.cpu())
        test_hidden = torch.cat(test_hidden, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        # 用 PyTorch 的 DataLoader 包装这些固定特征
        class FeatureDataset(Dataset):
            def __init__(self, feats, labels):
                self.feats = feats
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                return self.feats[idx], self.labels[idx]

        train_feat_loader = DataLoader(FeatureDataset(train_hidden, train_labels), batch_size=32, shuffle=True)
        test_feat_loader = DataLoader(FeatureDataset(test_hidden, test_labels), batch_size=32)

        probe = LinearProbe(input_dim=768).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=PROBING_LR)
        criterion = nn.CrossEntropyLoss()

        # 训练线性层
        probe.train()
        for epoch in range(PROBING_EPOCHS):
            for feats, lbls in train_feat_loader:
                feats = feats.to(device)
                lbls = lbls.to(device)
                opt.zero_grad()
                logits = probe(feats)
                loss = criterion(logits, lbls)
                loss.backward()
                opt.step()

        # 评估
        probe.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feats, lbls in test_feat_loader:
                feats = feats.to(device)
                lbls = lbls.to(device)
                logits = probe(feats)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        layer_accuracies.append(acc)

    return layer_accuracies


def read_probing_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gbk")
    # 尝试找到 text 列和 label 列
    text_col = None
    label_col = None
    for col in df.columns:
        col_low = col.lower()
        if col_low in ("text", "review", "content", "sentence"):
            text_col = col
        if col_low in ("label", "class", "tag"):
            label_col = col
    if text_col is None:
        # 假设第一列是 text
        text_col = df.columns[0]
    if label_col is None:
        # 假设第二列是 label
        label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    df = df.rename(columns={text_col: "text", label_col: "label"})
    return df[["text", "label"]]


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    all_results = []

    for model_name, model_dir in FINETUNED_MODELS.items():
        if not os.path.isdir(model_dir):
            print(f"警告: 模型目录不存在: {model_dir}，跳过 {model_name}")
            continue

        print(f"\n>>> 开始探测模型: {model_name}")
        # 加载微调后的完整分类模型
        full_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
        encoder = full_model.bert  # 提取 BERT 编码器
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        for task_name, csv_path in PROBING_TASKS.items():
            if not os.path.exists(csv_path):
                print(f"  数据文件不存在: {csv_path}，跳过任务 {task_name}")
                continue
            print(f"  --- 任务: {task_name}")
            accuracies = run_probing_for_model(
                encoder, tokenizer, csv_path, DEVICE, model_name, task_name
            )
            if accuracies is None:
                continue
            for layer_idx, acc in enumerate(accuracies):
                all_results.append({
                    "Model": model_name,
                    "Task": task_name,
                    "Layer": layer_idx,
                    "Accuracy": acc,
                })

    # 保存结果 CSV
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n实验完成！结果保存至: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()