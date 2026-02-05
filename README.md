# Metric Learning for Artist Mark Retrieval

ResNet-50 + triplet loss for trademark retrieval.

## Quick Start

```bash
pip install -r requirements.txt
python train.py                    # Train model
python evaluate_test_to_all.py     # Evaluate
```

## Configuration

Edit `config.py`:

```python
MODEL_CONFIG = {'embedding_dim': 512, 'dropout': 0.5}
TRAIN_CONFIG = {'learning_rate': 5e-5, 'batch_size': 32, 'm_per_class': 4}
LOSS_CONFIG = {'margin': 0.7}
```

## Key Scripts

| Script                    | Purpose                               |
| ------------------------- | ------------------------------------- |
| `train.py`                | Train model (saves to `checkpoints/`) |
| `evaluate_test_to_all.py` | Evaluate test→all retrieval           |
| `config.py`               | Hyperparameters                       |

## Project Structure

```
metric-learning/
├── train.py                 # Training script
├── evaluate_test_to_all.py  # Evaluation script
├── config.py                # Configuration
├── model.py                 # ResNet-50 architecture
├── dataset.py & dataloader.py
├── labels/                  # CSV files (train/val/test splits)
├── Other_Marks/             # Images (not in git)
└── checkpoints/             # Saved models (not in git)
```

## Data Setup

Place your images in `Other_Marks/` directory (not included in git).
CSV labels should be in `labels/` directory.
