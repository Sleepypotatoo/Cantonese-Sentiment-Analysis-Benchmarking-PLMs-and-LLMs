# Cantonese Sentiment Analysis – Decoding Lexical, Syntactic, and Pragmatic Features

This repository contains the official implementation of our paper:  
**“Decoding Cantonese Sentiment: An Empirical Analysis Across Lexical, Syntactic, and Pragmatic Levels”**.

## Project Structure

├── README.md
├── requirements.txt
├── download_models.py
├── data/ # 存放所有原始数据集 (见下方准备)
├── probing_data/ # 7 个语言学特征的探测数据集
├── src/ # 核心代码
│ ├── train_plm_5cls.py # 训练 BERT 系列模型 (粤语/中文/mBERT)、五分类
│ ├── train_pycantonese.py # 使用 pycantonese 分词的训练脚本、五分类
│ ├── probing.py # 线性探测实验
│ ├── probing_figures.py # 绘制热力图 (13 层 × 7 特征)
│ └── evaluate_features.py # 特征级错误分析
├── results/ # 自动生成，存放模型和探测结果
└── scripts/ # 辅助脚本

## Environment Setup

````bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/Mac
# 或 .\venv\Scripts\activate (Windows)

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) 预下载模型到本地，加速训练
python download_models.py

## Data Preparation
请将以下数据集按路径放置：
数据集	路径	格式	来源/说明
OpenRice 二分类	data/openrice_2cls/train.csv , test.csv	csv (text,label)	HuggingFace
OpenRice 三分类	data/openrice_3cls/train.tsv , test.tsv	tsv	GitHub
OpenRice 五分类	data/5cls_data/openrice.csv	csv	GitHub
HKTVmall 五分类	data/5cls_data/hktvmall.csv	csv	内部构建（请联系作者）
探测数据集	probing_data/*.csv	csv (text,label)	构造的 7 个二分类特征集（见论文）
注意：所有 CSV/TSV 文件必须包含 text 和 label 两列（标签为整数，1~5 或 0/1）。

## Running Experiments
1. 训练分类模型（PLMs）
```bash
# 训练粤语BERT (CantoneseBERT)
python src/train_plm_5cls.py --model cantonese --data_path data/5cls_data/hktvmall.csv

# 训练中文BERT
python src/train_plm_5cls.py --model chinese

# 训练多语言BERT (mBERT)
python src/train_plm_5cls.py --model mbert

# 使用国内镜像加速下载
python src/train_plm_5cls.py --model cantonese --use_mirror

训练后的模型保存在 results/models/hktv_5emos_<model>_model/

2. 使用 pycantonese 分词的训练（额外基线）
```bash
python src/train_pycantonese.py

3. 特征级错误分析
首先确保已训练出模型并得到 predictions.csv 和 errors.csv，然后运行，假设训练出的模型为hktv_5emos_cantonese_model：
```bash
python src/evaluate_features.py \
    --error_file results/models/hktv_5emos_cantonese_model/errors.csv \
    --pred_file results/models/hktv_5emos_cantonese_model/predictions.csv
结果保存在 feature_analysis_results/ 目录。

4. 线性探测实验
```bash
python src/probing.py

5. 绘制热力图
```bash
python src/probing_figures.py
生成的图片为 results/probing/heatmap_2class.png 等。

## Notes for Double-Blind Review
本项目已通过 Anonymous GitHub 完成匿名化。
所有路径均使用相对路径，无需人工修改。
训练时模型会自动从 HuggingFace Hub 下载（需要网络），若网速慢可使用 --use_mirror 或提前运行 download_models.py。
请勿上传训练好的模型文件（.bin, .safetensors）到代码仓库，审稿人可自行训练获得相同结果。

## Acknowledgements
We thank the contributors of OpenRice, HKTVmall, and the pycantonese toolkit.
This work is supported by xxx (removed for anonymity).

## contact
For questions about the code, please open an issue in the anonymous repository.
````
