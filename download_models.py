import os
from transformers import AutoModel, AutoTokenizer

MODELS = {
    "bert-base-cantonese": "manueltonneau/bert-base-cantonese",
    "bert-base-chinese": "bert-base-chinese",
    "bert-base-multilingual-cased": "bert-base-multilingual-cased"
}

def download_model(model_id, local_dir="./pretrained_models"):
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    save_path = os.path.join(local_dir, model_id.replace('/', '_'))
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    for alias, hf_id in MODELS.items():
        download_model(hf_id)