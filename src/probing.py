#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整探测脚本：支持逐层探测（所有层）和仅最后一层，并自动汇总：
- 每层的正负例 Precision/Recall/F1
- 最后一层的表现
- 跨层最佳 F1
支持 HuggingFace 模型（BERT / RoBERTa / Qwen 等）和 Ollama 模型（单层）。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel
import ollama
from tqdm import tqdm

# ======================== 配置 ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 8                     # 特征提取批次（显存有限可调小）
PROBE_BATCH_SIZE = 32
PROBE_EPOCHS = 15
PROBE_LR = 1e-3
EARLY_STOP_PATIENCE = 3

# 模型配置
MODELS = {
    "Qwen2.5-3B": {
        "path": "Qwen2.5-3B",          
        "use_ollama": False,
        "dtype": torch.float16              # 使用半精度省显存
    },
    "mBERT": {
        "path": "bert-base-multilingual-cased",
        "use_ollama": False,
        "dtype": torch.float32
    },
    "CantoneseBERT": {
        "path": "bert-base-cantonese",
        "use_ollama": False,
        "dtype": torch.float32
    },
    "Chinese_RoBERTa": {
        "path": "roberta",
        "use_ollama": False,
        "dtype": torch.float32
    },
    # Ollama 模型示例（单层）
    # "Ollama_Embed": {
    #     "path": "nomic-embed-text",
    #     "use_ollama": True
    # }
}

DATA_ROOT = "project/probing_data/"
CATEGORIES = {
    "词汇层_一词多义": "词汇层_hktv_false friend.csv",
    "词汇层_极性修饰词": "词汇层_hk_极性修饰.csv",
    "句法层_比较句式": "词汇层_hk_比较.csv",
    "句法层_否定句式": "句法层_hk_否定.csv",
    "语用层_句末语气词": "语用层_hk_句末语气.csv",
    "语用层_中英夹杂": "语用层_hktv_中英.csv",
    "语用层_粤式反讽": "语用层_hk_粤式反讽_train.csv",
}

OUTPUT_DIR = "test.jsonl/project/results/probing_full/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== 工具函数 ========================
def load_category_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    return df['text'].tolist(), df['label'].tolist()

# ======================== 特征提取（复用模型） ========================
def extract_hf_features_from_model(model, tokenizer, texts, return_all_layers=True, batch_size=BATCH_SIZE):
    """
    使用已加载的模型提取特征，不重复加载模型。
    """
    model.eval()
    all_layer_feats = []
    if return_all_layers:
        num_layers = model.config.num_hidden_layers + 1
        all_layer_feats = [[] for _ in range(num_layers)]

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        if return_all_layers:
            for layer_idx, h in enumerate(hidden_states):
                cls = h[:, 0, :].cpu().to(torch.float32)   # 统一转为 float32
                all_layer_feats[layer_idx].append(cls)
        else:
            cls = outputs.last_hidden_state[:, 0, :].cpu().to(torch.float32)
            all_layer_feats.append(cls)

    if return_all_layers:
        result = [torch.cat(lst, dim=0) for lst in all_layer_feats]
        return result, model.config.hidden_size
    else:
        return torch.cat(all_layer_feats, dim=0), model.config.hidden_size

def extract_ollama_features(model_name, texts, batch_size=8):
    """Ollama 单层特征提取"""
    all_features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Ollama extracting"):
        batch = texts[i:i+batch_size]
        response = ollama.embed(model=model_name, input=batch)
        batch_emb = [torch.tensor(emb, dtype=torch.float32) for emb in response['embeddings']]
        all_features.append(torch.stack(batch_emb))
    features = torch.cat(all_features, dim=0)
    return features, features.shape[1]

# ======================== 线性探针 ========================
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

def train_probe(train_feats, train_labels, val_feats, val_labels, input_dim):
    probe = LinearProbe(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(list(zip(train_feats, train_labels)),
                              batch_size=PROBE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(list(zip(val_feats, val_labels)),
                            batch_size=PROBE_BATCH_SIZE, shuffle=False)

    best_val_f1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(PROBE_EPOCHS):
        probe.train()
        for feats, lbls in train_loader:
            feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = probe(feats)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

        probe.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
                logits = probe(feats)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(lbls.cpu().numpy())
        val_f1 = precision_recall_fscore_support(all_true, all_preds, average='macro', zero_division=0)[2]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = probe.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                break

    probe.load_state_dict(best_state)
    return probe

def evaluate_probe(probe, test_feats, test_labels):
    probe.eval()
    with torch.no_grad():
        feats = test_feats.to(DEVICE)
        logits = probe(feats)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets = test_labels.numpy()
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, labels=[0,1], average=None, zero_division=0)
    return {
        'neg_prec': p[0], 'neg_rec': r[0], 'neg_f1': f1[0],
        'pos_prec': p[1], 'pos_rec': r[1], 'pos_f1': f1[1]
    }

# ======================== 主流程 ========================
def main():
    all_results = []

    for model_label, model_info in MODELS.items():
        model_path = model_info['path']
        use_ollama = model_info.get('use_ollama', False)
        dtype = model_info.get('dtype', torch.float32)

        print(f"\n{'='*60}")
        print(f"处理模型: {model_label} (Ollama={use_ollama})")
        print(f"{'='*60}")

        # --- 加载模型（仅一次） ---
        if use_ollama:
            print("使用 Ollama 模式（单层）")
            # 验证 Ollama 服务
            try:
                _ = ollama.embed(model=model_path, input=["test"])
                print("Ollama 服务连接正常")
            except Exception as e:
                print(f"Ollama 不可用: {e}")
                continue
            tokenizer = None
            model = None
        else:
            print(f"加载 HuggingFace 模型: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if DEVICE.type == 'cuda' else None
            ).to(DEVICE)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            print(f"模型加载完成，共 {model.config.num_hidden_layers} 层")

        # --- 遍历所有任务 ---
        for cat_name, filename in CATEGORIES.items():
            file_path = os.path.join(DATA_ROOT, filename)
            if not os.path.exists(file_path):
                print(f"  跳过不存在的文件: {file_path}")
                continue

            print(f"\n  >>> 任务: {cat_name}")
            texts, labels = load_category_data(file_path)

            # 划分数据集
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
            )

            # --- 提取特征 ---
            if use_ollama:
                train_feats, hidden_dim = extract_ollama_features(model_path, train_texts)
                val_feats, _ = extract_ollama_features(model_path, val_texts)
                test_feats, _ = extract_ollama_features(model_path, test_texts)
                # 训练一个探针（单层）
                probe = train_probe(train_feats, torch.tensor(train_labels),
                                    val_feats, torch.tensor(val_labels), hidden_dim)
                metrics = evaluate_probe(probe, test_feats, torch.tensor(test_labels))
                row = {
                    'Model': model_label, 'Task': cat_name, 'Layer': -1,
                    'Neg_P': metrics['neg_prec'], 'Neg_R': metrics['neg_rec'], 'Neg_F1': metrics['neg_f1'],
                    'Pos_P': metrics['pos_prec'], 'Pos_R': metrics['pos_rec'], 'Pos_F1': metrics['pos_f1']
                }
                all_results.append(row)
            else:
                # 使用已加载模型提取所有层
                train_layer_feats, hidden_dim = extract_hf_features_from_model(
                    model, tokenizer, train_texts, return_all_layers=True, batch_size=BATCH_SIZE
                )
                val_layer_feats, _ = extract_hf_features_from_model(
                    model, tokenizer, val_texts, return_all_layers=True, batch_size=BATCH_SIZE
                )
                test_layer_feats, _ = extract_hf_features_from_model(
                    model, tokenizer, test_texts, return_all_layers=True, batch_size=BATCH_SIZE
                )
                num_layers = len(train_layer_feats)

                for layer_idx in range(num_layers):
                    probe = train_probe(
                        train_layer_feats[layer_idx], torch.tensor(train_labels),
                        val_layer_feats[layer_idx], torch.tensor(val_labels),
                        hidden_dim
                    )
                    metrics = evaluate_probe(probe, test_layer_feats[layer_idx], torch.tensor(test_labels))
                    row = {
                        'Model': model_label, 'Task': cat_name, 'Layer': layer_idx,
                        'Neg_P': metrics['neg_prec'], 'Neg_R': metrics['neg_rec'], 'Neg_F1': metrics['neg_f1'],
                        'Pos_P': metrics['pos_prec'], 'Pos_R': metrics['pos_rec'], 'Pos_F1': metrics['pos_f1']
                    }
                    all_results.append(row)

                # 清理特征张量以释放内存
                del train_layer_feats, val_layer_feats, test_layer_feats
                torch.cuda.empty_cache()

        # 模型用完后清理
        if not use_ollama:
            del model, tokenizer
        torch.cuda.empty_cache()
        print(f"模型 {model_label} 处理完毕，显存已清理。\n")

    # 保存结果
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(OUTPUT_DIR, "all_layers_results.csv"), index=False)

    # 汇总表格
    df_last = df_all.loc[df_all.groupby(['Model', 'Task'])['Layer'].idxmax()]
    print("\n" + "="*80)
    print("【最后一层】各模型在各任务上的正负例 F1")
    print("="*80)
    print(df_last[['Model', 'Task', 'Neg_F1', 'Pos_F1']].to_string(index=False))

    df_best = df_all.groupby(['Model', 'Task']).agg({
        'Neg_F1': 'max', 'Pos_F1': 'max',
        'Neg_P': 'max', 'Pos_P': 'max',
        'Neg_R': 'max', 'Pos_R': 'max'
    }).reset_index()
    print("\n" + "="*80)
    print("【跨层最佳】各模型在各任务上正负例 F1 的最大值（分别从不同层选取）")
    print("="*80)
    print(df_best[['Model', 'Task', 'Neg_F1', 'Pos_F1']].to_string(index=False))

    print(f"\n所有结果已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
