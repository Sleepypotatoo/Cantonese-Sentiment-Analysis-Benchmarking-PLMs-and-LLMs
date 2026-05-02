import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.utils.class_weight import compute_class_weight

# ==================== 配置 ====================
set_seed(42)
MODEL_PATH = r"bert-base-cantonese"
DATA_PATH = "./data/5cls_data/hktvmall.csv"
MAX_LEN = 128
OUTPUT_DIR = "./saved_models/hktv_5emos_cantonese_model"

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label', 'text'])
    df['label'] = df['label'].astype(int)
    if df['label'].min() == 1:
        df['label'] = df['label'] - 1   # 转为 0-4
    df = df[df['label'].isin([0,1,2,3,4])]
    print(f"数据总量: {len(df)}")
    print(df['label'].value_counts().sort_index())
    return df

class CantoneseDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
        self.labels = labels
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()} | {'labels': torch.tensor(self.labels[idx])}
    def __len__(self):
        return len(self.labels)

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1.0, emb_name='word_embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='word_embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs1 = model(**inputs)
        outputs2 = model(**inputs)
        logits1, logits2 = outputs1.logits, outputs2.logits

        if self.class_weights is not None:
            # 关键：将权重移动到 logits 所在的设备（GPU）
            weight = self.class_weights.to(logits1.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
        else:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)

        ce_loss = (loss_fct(logits1, labels) + loss_fct(logits2, labels)) / 2
    # ... 其余代码不变
        p = F.log_softmax(logits1, dim=-1)
        q = F.log_softmax(logits2, dim=-1)
        p_tec = F.softmax(logits1, dim=-1)
        q_tec = F.softmax(logits2, dim=-1)
        kl_loss = F.kl_div(p, q_tec, reduction='batchmean') + F.kl_div(q, p_tec, reduction='batchmean')
        loss = ce_loss + 4.0 * (kl_loss / 4)
        return (loss, outputs1) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)

        fgm = FGM(model)
        fgm.attack(epsilon=0.5)
        loss_adv = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss_adv = loss_adv.mean()
        self.accelerator.backward(loss_adv)
        fgm.restore()

        return loss.detach() / self.args.gradient_accumulation_steps

def train():
    df = load_and_preprocess(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.arange(5), y=train_df['label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print("类别权重:", class_weights)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=5, ignore_mismatched_sizes=True)
    train_dataset = CantoneseDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, MAX_LEN)
    test_dataset = CantoneseDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer, MAX_LEN)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=3,
        per_device_eval_batch_size=8,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=True,
        logging_steps=10,
        save_total_limit=2,
        report_to="none"
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        class_weights=class_weights.to(model.device),
        compute_metrics=lambda p: {
            'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=-1)),
            'f1_macro': f1_score(p.label_ids, np.argmax(p.predictions, axis=-1), average='macro'),
            'f1_micro': f1_score(p.label_ids, np.argmax(p.predictions, axis=-1), average='micro')
        },
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 评估
    pred = trainer.predict(test_dataset)
    y_pred = np.argmax(pred.predictions, axis=-1)
    y_true = pred.label_ids
    print(classification_report(y_true + 1, y_pred + 1, digits=4))

    # 保存结果（包含 text、真实评分、预测评分、真实ID、预测ID）
    results_df = pd.DataFrame({
        'text': test_df['text'].values,
        'true_rating': y_true + 1,
        'pred_rating': y_pred + 1,
        'true_id': y_true,
        'pred_id': y_pred
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False, encoding='utf-8')
    print(f"预测结果已保存至 {os.path.join(OUTPUT_DIR, 'predictions.csv')}")

    # 保存错误样本
    errors = results_df[results_df['true_id'] != results_df['pred_id']]
    if len(errors) > 0:
        errors.to_csv(os.path.join(OUTPUT_DIR, 'errors.csv'), index=False, encoding='utf-8')
        print(f"错误样本数: {len(errors)}，已保存至 {os.path.join(OUTPUT_DIR, 'errors.csv')}")
    else:
        print("没有错误样本。")

    print("训练完成！")

if __name__ == "__main__":
    train()