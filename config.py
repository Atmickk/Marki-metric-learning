"""
Configuration file for metric learning training
"""
import os

# Get base directory (current directory is the root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_CONFIG = {
    'train_csv': os.path.join(BASE_DIR, "labels", "final_labels2_train.csv"),
    'val_csv': os.path.join(BASE_DIR, "labels", "final_labels2_val.csv"),
    'image_dir': os.path.join(BASE_DIR, "Other_Marks"),
    'checkpoint_dir': os.path.join(BASE_DIR, "checkpoints"),
}

# Model configuration
MODEL_CONFIG = {
    'embedding_dim': 512,
    'pretrained': True,
    'dropout': 0.5,
}

# Training configuration
TRAIN_CONFIG = {
    'num_epochs': 100,
    'batch_size': 32,  # 4 × 16 classes per batch
    'learning_rate': 5e-5,  # Lower LR for more stable training
    'weight_decay': 1e-4,  # Increased regularization
    'patience': 25,  # More patience for early stopping
    'lr_scheduler_patience': 8,  # More patience before reducing LR
    'lr_scheduler_factor': 0.7,  # Gentler LR reduction (0.7 instead of 0.5)
}

# Loss configuration
LOSS_CONFIG = {
    'margin': 0.7,  # Increased margin for better separation
    'triplet_type': 'hard',  # 'hard', 'semihard', or 'all'
}

# DataLoader configuration
DATALOADER_CONFIG = {
    'm_per_class': 4,  # Images per class per batch (3 allows ~10-11 artists per batch)
    'num_workers': 0 if os.name == 'nt' else 4,  # Windows compatibility
    'pin_memory': True,
}

# Logging
LOG_CONFIG = {
    'log_interval': 10,  # Log every N batches
    'save_best_only': True,
}
