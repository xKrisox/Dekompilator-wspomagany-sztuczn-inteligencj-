# AI-Assisted Decompiler

This repository contains code developed as part of a Master’s thesis project on building an AI-assisted decompiler using Transformer architectures. It includes two main components:

* **Custom Transformer** (`transformer_part/`): A hand‑implemented Transformer model, along with data preprocessing, training, and evaluation scripts.
* **T5 Experiments** (`T5_part/`): Fine‑tuning and inference pipelines leveraging Hugging Face’s pre‑trained T5 model for decompilation tasks.

---

## Repository Structure

```
├── transformer_part/       # Custom Transformer implementation
│   ├── preprocess.py       # Data preprocessing scripts
│   ├── train.py            # Training loop for the custom Transformer
│   ├── evaluate.py         # Evaluation and metrics calculation
│   └── requirements.txt    # Dependencies for transformer_part

├── T5_part/                # T5-based decompiler experiments
│   ├── fine_tune_t5.py     # Script to fine‑tune T5 on decompilation data
│   ├── inference_t5.py     # Inference pipeline using the fine‑tuned T5 model
│   └── requirements.txt    # Dependencies for T5_part

└── README.md               # Project overview and usage instructions
```

## Prerequisites

* Python 3.7 or higher
* Git
* CUDA‑enabled GPU (recommended for training)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/xKrisox/Dekompilator-wspomagany-sztuczn-inteligencj-.git
   cd Dekompilator-wspomagany-sztuczn-inteligencj-
   ```
2. Install dependencies for both components:

   ```bash
   pip install -r transformer_part/requirements.txt
   pip install -r T5_part/requirements.txt
   ```

## Usage

### 1. Custom Transformer

Preprocess data, train the model, and evaluate performance:

```bash
# Data preprocessing
python transformer_part/preprocess.py \
  --input_dir data/raw \
  --output_dir data/processed

# Training
python transformer_part/train.py \
  --config transformer_part/config.yaml \
  --data_dir data/processed \
  --output_dir checkpoints/transformer

# Evaluation
python transformer_part/evaluate.py \
  --checkpoint checkpoints/transformer/best_model.pt \
  --test_data data/processed/test.csv
```

### 2. T5 Fine‑Tuning & Inference

Fine‑tune a pre‑trained T5 model and run inference:

```bash
# Fine‑tuning
python T5_part/fine_tune_t5.py \
  --model_name t5-small \
  --train_data data/processed/train.csv \
  --output_dir checkpoints/t5

# Inference
python T5_part/inference_t5.py \
  --model_dir checkpoints/t5 \
  --input_file data/processed/test.bin \
  --output_file results/decompiled_code.txt
```

## Results & Analysis

Detailed evaluation metrics, error analysis, and loss curves are presented in the accompanying thesis document. You can find saved logs and plots in:

* `transformer_part/logs/`
* `T5_part/logs/`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your proposed changes.

## License

This project is released under the MIT License. See `LICENSE` for details.
