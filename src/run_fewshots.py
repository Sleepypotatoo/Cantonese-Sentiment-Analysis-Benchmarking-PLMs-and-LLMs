import os
import gc
import asyncio
import aiohttp
import random
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.asyncio import tqdm

# ================= 配置区 =================
# 推荐先只跑 Gemma2-2B（速度最快），其他模型按需取消注释
MODEL_MAP = {
    "Qwen2.5-3B-Instruct": "qwen2.5:3b-instruct-q4_K_M",
    "Llama3.2-3B-Instruct": "llama3.2:3b-instruct-fp16",
    "Gemma2-2B": "gemma2:2b",
}

DATASETS = ["openrice", "hktvmall"] #五分类
SHOTS = [10, 5]               # 每类样本数（5→总25条，10→总50条）
NUM_SEEDS = 3
SEEDS = [42, 2026, 3407]

DATA_DIR = "test.jsonl/project/data/5cls_data"
GOLDEN_TRAIN_PATH = "test.jsonl/5emotions_train/train.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"

# 速度控制（全量测试较慢，建议先用 QUICK_TEST_SIZE = 500 验证）
QUICK_TEST_SIZE = None            # 设为数字（如500）可大幅加速
MAX_CONCURRENT_REQUESTS = 5
MAX_RETRIES_PER_SEED = 1
# ==========================================

def load_and_clean_golden_train(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 未找到专属训练集: {file_path}")
    df = pd.read_csv(file_path, on_bad_lines='skip').dropna(subset=['label', 'text'])
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype(int)
    if df['label'].min() == 1:
        df['label'] = df['label'] - 1
    df = df[df['label'].isin([0,1,2,3,4])]
    df['text_clean'] = df['text'].astype(str).str.strip()
    return df, set(df['text_clean'].tolist())

def get_balanced_few_shots(golden_train_df, shots_per_class, seed):
    if shots_per_class == 0:
        return ""
    sampled_chunks = []
    for label_val in [0,1,2,3,4]:
        subset = golden_train_df[golden_train_df['label'] == label_val]
        if len(subset) > 0:
            n = min(shots_per_class, len(subset))
            sampled_chunks.append(subset.sample(n=n, random_state=seed))
    if not sampled_chunks:
        return ""
    sampled_df = pd.concat(sampled_chunks).sample(frac=1, random_state=seed).reset_index(drop=True)
    ctx = []
    for _, row in sampled_df.iterrows():
        short_text = str(row['text_clean'])[:100]
        score = int(row['label']) + 1
        ctx.append(f"文本：{short_text}\n评分：{score}")
    return "\n\n".join(ctx)   # 返回纯文本示例块

def build_prompt_gemma(text, few_shot_ctx):
    system = ("你是一个精通粤语的情感分析专家。请仔细阅读给定的粤语文本，评估其情感倾向并给出评分（只能输出数字 1、2、3、4、5 中的一个，"
              "1代表极度负面，5代表极度正面）。只允许输出孤零零的一个数字，绝对不要包含任何多余的字符！")
    prompt = f"{system}\n\n"
    if few_shot_ctx:
        prompt += f"参考示例：\n{few_shot_ctx}\n\n"
    prompt += f"现在请对以下文本评分：\n文本：{text}\n评分："
    return prompt

def build_prompt_llama(text, few_shot_ctx):
    system = ("You are an expert in Cantonese sentiment analysis. "
              "Rate the sentiment of the Cantonese text on a scale of 1 to 5 "
              "(1=extremely negative, 5=extremely positive). Output ONLY a single number.")
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\n"
    if few_shot_ctx:
        # 将 few_shot_ctx 拆分为多个 user/assistant 回合
        examples = few_shot_ctx.split("\n\n")
        for ex in examples:
            if not ex:
                continue
            lines = ex.split("\n")
            if len(lines) >= 2:
                text_part = lines[0].replace("文本：", "").strip()
                score_part = lines[1].replace("评分：", "").strip()
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\nText: {text_part}<|eot_id|>\n"
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{score_part}<|eot_id|>\n"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\nText: {text}<|eot_id|>\n"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def build_prompt_qwen(text, few_shot_ctx):
    system = ("你是一个精通粤语的情感分析专家。请仔细阅读给定的粤语文本，评估其情感倾向并给出评分（1到5分，1=极度负面，5=极度正面）。"
              "你必须只输出一个数字（1-5），不要有任何其他内容。")
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
    if few_shot_ctx:
        examples = few_shot_ctx.split("\n\n")
        for ex in examples:
            if not ex:
                continue
            lines = ex.split("\n")
            if len(lines) >= 2:
                text_part = lines[0].replace("文本：", "").strip()
                score_part = lines[1].replace("评分：", "").strip()
                prompt += f"<|im_start|>user\n文本：{text_part}<|im_end|>\n"
                prompt += f"<|im_start|>assistant\n{score_part}<|im_end|>\n"
    prompt += f"<|im_start|>user\n文本：{text}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    return prompt

async def call_ollama(session, semaphore, model_tag, prompt, use_json):
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 5}
    }
    if use_json:
        payload["format"] = "json"
    async with semaphore:
        for _ in range(2):
            try:
                async with session.post(OLLAMA_URL, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        return res.get("response", "").strip()
            except Exception:
                await asyncio.sleep(1.0)
        return ""

async def run_inferences(session, semaphore, model_tag, prompts, use_json):
    tasks = [call_ollama(session, semaphore, model_tag, p, use_json) for p in prompts]
    return await tqdm.gather(*tasks, desc="推理中")

def extract_label(raw):
    clean = str(raw).strip().lower()
    num = re.search(r'[1-5]', clean)
    if num:
        return int(num.group()) - 1
    if any(w in clean for w in ["差","烂","难食","失望","麻麻"]):
        return 0
    if any(w in clean for w in ["好","正","满意","赞","推荐"]):
        return 4
    return 2

def calc_metrics(y_true, y_pred):
    y_true = [int(t) for t in y_true if not pd.isna(t)]
    y_pred = [int(p) for p in y_pred if not pd.isna(p)]
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    if not y_true:
        return {k: 0.0 for k in ["Accuracy","Precision (Macro)","Recall (Macro)",
                                 "F1 (Macro)","F1 (Micro)","F1 (Weighted)",
                                 "MAE","±1_Acc","Severe_Err"]}
    yt = np.array(y_true)
    yp = np.array(y_pred)
    abs_err = np.abs(yt - yp)
    return {
        "Accuracy": accuracy_score(yt, yp),
        "Precision (Macro)": precision_score(yt, yp, average='macro', zero_division=0),
        "Recall (Macro)": recall_score(yt, yp, average='macro', zero_division=0),
        "F1 (Macro)": f1_score(yt, yp, average='macro', zero_division=0),
        "F1 (Micro)": f1_score(yt, yp, average='micro', zero_division=0),
        "F1 (Weighted)": f1_score(yt, yp, average='weighted', zero_division=0),
        "MAE": np.mean(abs_err),
        "±1_Acc": np.mean(abs_err <= 1),
        "Severe_Err": np.mean(abs_err >= 3)
    }

async def evaluate(session, semaphore, model_tag, test_df, shots, seed, use_json, golden_df):
    texts = test_df['text_clean'].tolist()
    y_true = test_df['label'].tolist()
    if shots == 0:
        if "Gemma" in model_tag:
            prompts = [build_prompt_gemma(t, "") for t in texts]
        elif "llama" in model_tag:
            prompts = [build_prompt_llama(t, "") for t in texts]
        else:
            prompts = [build_prompt_qwen(t, "") for t in texts]
    else:
        ctx = get_balanced_few_shots(golden_df, shots, seed)
        if "Gemma" in model_tag:
            prompts = [build_prompt_gemma(t, ctx) for t in texts]
        elif "llama" in model_tag:
            prompts = [build_prompt_llama(t, ctx) for t in texts]
        else:
            prompts = [build_prompt_qwen(t, ctx) for t in texts]
    raw = await run_inferences(session, semaphore, model_tag, prompts, use_json)
    y_pred = [extract_label(r) for r in raw]
    return calc_metrics(y_true, y_pred)

async def async_main():
    print("📖 加载专属训练集...")
    golden_df, golden_texts_set = load_and_clean_golden_train(GOLDEN_TRAIN_PATH)

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    conn = aiohttp.TCPConnector(limit=100)

    async with aiohttp.ClientSession(connector=conn) as session:
        for model_name, model_tag in MODEL_MAP.items():
            use_json = ("qwen" in model_name.lower())   # Qwen 可以用 JSON，但系统指令已要求纯数字，设为 False 更简单
            if "qwen" in model_name.lower():
                use_json = False   # 让模型直接输出数字，不强制 JSON
            for dataset in DATASETS:
                print(f"\n========== {model_name} | {dataset} ==========")
                fpath = f"{DATA_DIR}/{dataset}.csv"
                if not os.path.exists(fpath):
                    print(f"跳过 {fpath}")
                    continue
                df = pd.read_csv(fpath, on_bad_lines='skip').dropna(subset=['label','text'])
                df['label'] = pd.to_numeric(df['label'], errors='coerce').astype(int)
                if df['label'].min() == 1:
                    df['label'] -= 1
                df = df[df['label'].isin(range(5))]
                df['text_clean'] = df['text'].astype(str).str.strip()
                df = df[~df['text_clean'].isin(golden_texts_set)].reset_index(drop=True)
                # 划分测试集
                _, test_full = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
                if QUICK_TEST_SIZE and len(test_full) > QUICK_TEST_SIZE:
                    # 简单随机下采样
                    test_df = test_full.sample(n=QUICK_TEST_SIZE, random_state=42).reset_index(drop=True)
                else:
                    test_df = test_full.reset_index(drop=True)
                print(f"测试集大小: {len(test_df)} 条")

                local_res = []
                # 0-shot
                metrics = await evaluate(session, sem, model_tag, test_df, 0, None, use_json, golden_df)
                metrics.update({"Model": model_name, "Dataset": dataset, "Shots": 0, "Seed": "N/A"})
                local_res.append(metrics)
                print(f"0-shot -> F1(Macro): {metrics['F1 (Macro)']:.4f}")

                for shots in [5,10]:
                    for seed in SEEDS:
                        metrics = await evaluate(session, sem, model_tag, test_df, shots, seed, use_json, golden_df)
                        metrics.update({"Model": model_name, "Dataset": dataset, "Shots": shots, "Seed": seed})
                        local_res.append(metrics)
                        print(f"{shots}-shot (seed={seed}) -> F1(Macro): {metrics['F1 (Macro)']:.4f}")

                # 保存当前模型-数据集结果
                df_out = pd.DataFrame(local_res)
                cols = ["Model","Dataset","Shots","Seed","Accuracy","Precision (Macro)","Recall (Macro)",
                        "F1 (Macro)","F1 (Micro)","F1 (Weighted)","MAE","±1_Acc","Severe_Err"]
                df_out = df_out[cols]
                print(f"\n{'='*60}\n即时结果：{model_name} | {dataset}\n{'='*60}")
                print(df_out.to_string(index=False))
                print(f"{'='*60}\n")
                df_out.to_csv(f"partial_{model_name}_{dataset}.csv", index=False, encoding="utf-8-sig")
                gc.collect()

if __name__ == "__main__":
    asyncio.run(async_main())
