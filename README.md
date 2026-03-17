# Metric Learning for Artist Mark Retrieval

DINOv2 / ResNet-50 + triplet loss for trademark retrieval.

## Quick Start

```bash
pip install -r requirements.txt
python train.py       # Train model
python evaluate.py    # Evaluate
```

> For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above.

## Configuration

Edit `config.py` before training:

```python
MODEL_CONFIG = {'backbone': 'dinov2_vits14', 'embedding_dim': 256, 'dropout': 0}
TRAIN_CONFIG = {'learning_rate': 1e-3, 'backbone_lr': 1e-5, 'batch_size': 16}
LOSS_CONFIG  = {'margin': 0.6, 'triplet_type': 'hard'}
```

Set `backbone` to `'resnet50'` to use ResNet-50 instead of DINOv2.

For DINOv2, two learning rates are used: `backbone_lr` for the pretrained ViT backbone and `learning_rate` for the projection head. On first run, DINOv2 weights are downloaded automatically via `torch.hub`.

## Usage

### Training

```bash
python train.py
```

- Trains until validation loss stops improving (early stopping, patience=25)
- Best model saved to `checkpoints/<backbone>_metric_best.pth`
- LR scheduler reduces LR on plateau automatically

### Evaluation

```bash
python evaluate.py
```

Loads the best checkpoint and reports:
- **mAP** — mean Average Precision over the full ranking
- **Accuracy@1 / @10** — whether a correct match appears in the top-1 / top-10 results

### Hyperparameter Search

```bash
python random_search.py
```

Runs multiple trials with randomly sampled hyperparameters. **Note: random search was only used with ResNet-50 and is not configured for DINOv2.** Results are saved to `random_search_results/all_results.json`, ranked by mAP. Each trial's best checkpoint is saved to `random_search_results/`.

Search space (edit in `random_search.py`):

| Parameter           | Range / Options       |
| ------------------- | --------------------- |
| `learning_rate`     | 1e-5 → 1e-3 (log)    |
| `embedding_dim`     | 128, 256, 512         |
| `batch_size`        | 16, 32, 64            |
| `margin`            | 0.2 → 1.0             |
| `dropout`           | 0.0 → 0.7             |
| `m_per_class`       | 2, 4, 8               |

## Key Scripts

| Script              | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `train.py`          | Train model (saves to `checkpoints/`)      |
| `evaluate.py`       | Evaluate with mAP and Accuracy@K           |
| `random_search.py`  | Hyperparameter search (ResNet-50)          |
| `config.py`         | All hyperparameters                        |

## Project Structure

```
metric-learning/
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── random_search.py     # Hyperparameter search
├── config.py            # Configuration
├── model.py             # DINOv2 / ResNet-50 embedder
├── dataset.py           # Dataset class
├── dataloader.py        # DataLoader setup
├── labels/              # CSV files (train/val/test splits)
├── Other_Marks/         # Images (not in git — add your own)
└── checkpoints/         # Saved models (not in git)
```

## Data Setup

Place your images in `Other_Marks/`. Label CSVs are already included in `labels/` and expect `filename` and `artist` columns.
