```markdown
# Decoding Cantonese Sentiment: Lexical, Syntactic, and Pragmatic Probing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

This repository contains the official code and data for the paper **"Decoding Cantonese Sentiment: An Empirical Analysis Across Lexical, Syntactic, and Pragmatic Levels"** (anonymous submission).  
We propose a hierarchical taxonomy of Cantonese linguistic features and systematically evaluate how they shape sentiment predictions across binary, 3‑class, and 5‑class granularities. Through layer‑wise probing, we reveal that task complexity dictates the depth at which models encode and retain dialect‑specific signals.

**Key findings**:
- Dialect‑specific PLMs (e.g., CantoneseBERT) outperform general LLMs in fine‑grained (5‑class) sentiment analysis.
- Cross‑dialectal *false friends* and *code‑mixing* trigger negative transfer from Mandarin priors, while *Utterance‑final Particles* serve as robust affective cues.
- Binary tasks exhibit deep‑layer *feature forgetting*; fine‑grained tasks force models to retain linguistic cues through the final layer.

> 📎 **Anonymous repository**: [https://anonymous.4open.science/r/Cantonese-Sentiment-Analysis-Benchmarking-PLMs-and-LLMs-854B/](https://anonymous.4open.science/r/Cantonese-Sentiment-Analysis-Benchmarking-PLMs-and-LLMs-854B/)  
> (URL will be updated upon acceptance.)

## Project Structure
```

.
├── README.md
├── requirements.txt
├── download_models.py            # Script to download required pre-trained models
├── data/                         # Raw datasets (see "Data Preparation" below)
├── probing_data/                 # 7 linguistic feature probing subsets (250 instances each)
├── src/                          # Core source code
│   ├── train_plm_5cls.py        # Fine-tune BERT-family models (CantoneseBERT, ChineseBERT, mBERT) for 5‑class sentiment
│   ├── train_pycantonese.py     # Train a PyCantonese lexicon‑based baseline for 5‑class sentiment
│   ├── probing.py               # Run linear probing experiments across 13 transformer layers
│   ├── probing_figures.py       # Generate layer‑wise heatmaps (13 layers × 7 features)
│   └── evaluate_features.py     # Feature‑level error analysis and delta‑F1 computation
└── results/                      # Auto‑generated: trained models, probing scores, figures

````text
## Features

- **Multi‑granularity Cantonese sentiment benchmarks**  
  OpenRice (Binary, 3‑class, 5‑class) + self‑collected HKTVmall 5‑class dataset, all with balanced or natural distributions.

- **Model zoo**  
  PyCantonese (lexicon), CantoneseBERT, ChineseBERT, mBERT, and Qwen3‑LoRA. Full‑parameter fine‑tuning plus parameter‑efficient adaptation.

- **Linguistic feature taxonomy**  
  Six feature categories spanning Lexical, Syntactic, and Pragmatic levels: *False Friends*, *Intensity Modifiers*, *Negation*, *Comparative Constructions*, *Code‑mixing*, and *Utterance‑final Particles*.

- **Layer‑wise probing suite**  
  Train linear probes on frozen CantoneseBERT representations to pinpoint where each feature is encoded. Outputs heatmaps for 2‑, 3‑, and 5‑class tasks.

- **Error analysis toolbox**  
  Qualitative case studies and quantitative delta‑F1 metrics to isolate the impact of dialectal phenomena.

## Installation

```bash
git clone https://anonymous.4open.science/r/Cantonese-Sentiment-Analysis-Benchmarking-PLMs-and-LLMs-854B/
cd Cantonese-Sentiment-Analysis-Benchmarking-PLMs-and-LLMs-854B
pip install -r requirements.txt
````

Download all required pre‑trained models by running:

```bash
python download_models.py
```

This will fetch CantoneseBERT, ChineseBERT, mBERT, and Qwen3‑4B weights from HuggingFace (models may be large; ensure stable network connection).

## Data Preparation

### Sentiment Datasets

1. **OpenRice‑Binary**: [HuggingFace dataset](https://huggingface.co/datasets/sepidmnorozy/Cantonese_sentiment)
   Download and place under `data/OpenRice/binary/`.
2. **OpenRice‑3C**: [GitHub repository](https://github.com/toastynews/openrice-senti)
   Place the TSV file under `data/OpenRice/3class/`.
3. **OpenRice‑5C**: [GitHub repository](https://github.com/Christainx/Dataset_Cantonese_Openrice.git)
   Place under `data/OpenRice/5class/`.
4. **HKTVmall‑5C** (self‑collected, anonymized)
   Provided in the repository at `data/HKTVmall/` (already included).

### Probing Subsets

Manually curated subsets for 6 linguistic features are located in `probing_data/`. Each subfolder contains positive (`pos.txt`) and negative (`neg.txt`) samples used for the binary probing classification.

## Usage

### Training Sentiment Classifiers

Train a 5‑class PLM model (CantoneseBERT, ChineseBERT, or mBERT):

```bash
python src/train_plm_5cls.py --model cantonese-bert --dataset openrice_5c --epochs 10 --batch_size 16
```

Run the PyCantonese baseline:

```bash
python src/train_pycantonese.py --dataset openrice_5c
```

Results (macro‑F1, checkpoints) are saved to `results/models/`.

### Probing Experiments

Train probes for a specific feature (e.g., `false_friends`) across all layers:

```bash
python src/probing.py --feature false_friends --granularity 5class --output_dir results/probes/
```

This will generate per‑layer accuracy files. To produce the heatmap figure (Figure 1 in the paper):

```bash
python src/probing_figures.py --probe_dir results/probes/ --output results/heatmap.png
```

### Feature‑level Error Analysis

Compute macro‑F1 on each linguistic subset and the relative Δ values:

```bash
python src/evaluate_features.py --model results/models/cantonese-bert_5c.pt --task 5class
```

The script produces a LaTeX‑ready table similar to Table 3 in the paper.

## Reproducing Main Results

The core experimental results are summarized below (Macro‑F1). All metrics can be reproduced using the provided scripts and datasets.

| Task | CantoneseBERT | ChineseBERT | mBERT | Qwen3‑LoRA | PyCantonese |
| ---  | --- | --- | --- | --- | --- |
| OpenRice‑Binary | 0.9392 | 0.9452 | 0.9270 | 0.9509 | 0.6213 |
| OpenRice‑3C | 0.7982 | 0.8043 | 0.7623 | 0.7989 | 0.6981 |
| OpenRice‑5C | 0.5735 | 0.4854 | 0.4759 | 0.4747 | 0.1582 |
| HKTVmall‑5C | 0.6262 | 0.6006 | 0.5655 | 0.5848 | 0.4356 |

Probing heatmaps illustrating the layer‑wise encoding of lexical vs. pragmatic features are shown in the paper's Figure 1.

## Citation

If you use our code or data, please cite the anonymous paper:

```bibtex
@inproceedings{anonymous2025decoding,
  title     = {Decoding Cantonese Sentiment: An Empirical Analysis Across Lexical, Syntactic, and Pragmatic Levels},
  author    = {Anonymous},
  year      = {2025},
  note      = {Under review}
}
```

(Full citation will be updated upon acceptance.)

## License

This project is licensed under the MIT License – see the [LICENSE](https://LICENSE) file for details.

```text

```
