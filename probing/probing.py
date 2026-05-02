import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split


# 1. 定义线性探针模型
class LinearProbe(nn.Module):
    def __init__(self, input_dim=768):
        super(LinearProbe, self).__init__()
        self.classifier = nn.Linear(input_dim, 2)  # 二分类：是否存在该特征

    def forward(self, x):
        return self.classifier(x)

# 2. 定义探测数据集加载器
class ProbingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            self.texts[item],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[item], dtype=torch.long)
        }

# 3. 探针训练与评估核心函数
def run_probing_per_layer(model, train_loader, test_loader, device):
    model.eval()
    layer_accuracies = []
    num_layers = model.config.num_hidden_layers + 1   # embedding (0) + 12层 = 13

    for layer_idx in range(num_layers):
        print(f"  Training Probe for Layer {layer_idx}...")
        probe = LinearProbe().to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 训练
        probe.train()
        for epoch in range(5):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hidden_state = outputs.hidden_states[layer_idx][:, 0, :]

                optimizer.zero_grad()
                logits = probe(hidden_state)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        # 评估
        probe.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_state = outputs.hidden_states[layer_idx][:, 0, :]
                logits = probe(hidden_state)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        layer_accuracies.append(acc)
    return layer_accuracies

def read_probing_csv(file_path):
    """
    鲁棒地读取探测任务 CSV 文件。
    假设文件第一行为表头：label,text（或 text,label），
    但实际读取时自动适配列顺序。
    返回 DataFrame 包含 'text' 和 'label' 两列。
    """
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)   # 跳过表头
        if header is None:
            return pd.DataFrame(columns=['text', 'label'])
        # 确定 text 和 label 的列索引
        # 期望表头包含 'text' 和 'label'（不区分大小写）
        col_text = None
        col_label = None
        for i, name in enumerate(header):
            name_low = name.strip().lower()
            if name_low in ('text', 'review', 'content'):
                col_text = i
            elif name_low in ('label', 'class', 'tag'):
                col_label = i
        # 如果没有找到明确的列名，则假设第一列是 label，第二列是 text（常见格式）
        if col_label is None and col_text is None:
            col_label, col_text = 0, 1
        elif col_label is None:
            col_label = 0 if col_text != 0 else 1
        elif col_text is None:
            col_text = 0 if col_label != 0 else 1
        
        for row in reader:
            if len(row) <= max(col_label, col_text):
                continue   # 跳过列数不足的行
            try:
                label_val = row[col_label].strip()
                text_val = row[col_text].strip()
                if text_val and label_val:
                    rows.append([text_val, label_val])
            except:
                continue
    df = pd.DataFrame(rows, columns=['text', 'label'])
    return df

# 4. 主实验循环
def main_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-cantonese')
    model_configs = {
        #使用粤语plm 微调后的模型 2/3/5 class
        "2-Class_Model": r"./can_bert_2cls_or/model.safetensors",
        "3-Class_Model": r"./can_bert_3cls_or/model.safetensors",
        "5-Class_Model": r"./can_bert_5cls_or/model.safetensors"
    }

    probing_tasks = { 
    "词汇层_一词多义": "./data/probing_train/词汇层_一词多义_train.csv",
    "词汇层_极性修饰词": "./data/probing_train/词汇层_极性修饰词_train.csv",
    "句法层_比较句式": "./data/probing_train/句法层_比较句式_train.csv",
    "句法层_否定句式": "./data/probing_train/句法层_否定句式_train.csv",
    "语用层_句末语气词": "./data/probing_train/语用层_句末语气词_train.csv",
    "语用层_中英夹杂": "./data/probing_train/语用层_中英夹杂_外来词_train.csv",
    "语用层_粤式反讽": "./data/probing_train/语用层_粤式反讽_train.csv",
 }  # 保持不变

    final_results = []
    for model_name, model_path in model_configs.items():
        print(f"\n>>> Probing Model: {model_name}")
        
        # 加载基础 BERT
        base_model = BertModel.from_pretrained(r'./bert-base-cantonese')
        
        # 加载 safetensors 权重
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        
        # 转换参数名：去掉 'bert.' 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('bert.'):
                new_state_dict[key[5:]] = value
        
        missing, unexpected = base_model.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        base_model.to(device)
        for param in base_model.parameters():
            param.requires_grad = False

        for task_name, csv_path in probing_tasks.items():
            print(f"--- Task: {task_name}")
            df = read_probing_csv(csv_path)
            
            # 数据清洗
            df = df.dropna(subset=['text'])
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)
            df = df[df['label'].isin([0, 1])]   # 探测任务通常是二分类
            print(f"  有效样本数: {len(df)}")
            if len(df) == 0:
                print("  警告：无有效样本，跳过")
                continue
            
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
            train_dataset = ProbingDataset(train_df['text'].values, train_df['label'].values, tokenizer)
            test_dataset = ProbingDataset(test_df['text'].values, test_df['label'].values, tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # 调用
            accuracies = run_probing_per_layer(base_model, train_loader, test_loader, device)
                        
            for layer, acc in enumerate(accuracies):
                final_results.append({
                    "Model": model_name,
                    "Task": task_name,
                    "Layer": layer,
                    "Accuracy": acc
                })

    results_df = pd.DataFrame(final_results)
    results_df.to_csv(r"./probing/probing_final_results.csv", index=False)
    print("\nExperiment Completed!")

if __name__ == "__main__":
    main_experiment()
