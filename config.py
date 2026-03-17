"""
Configuration file for metric learning training
"""
import os

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
    'backbone': 'dinov2_vits14',   # 'resnet50' or 'dinov2_vits14'
    'embedding_dim': 256,
    'pretrained': True,
    'dropout': 0,
    'freeze_backbone': False,
}

# Training configuration
TRAIN_CONFIG = {
    'num_epochs': 100,
    'batch_size': 16,
    'learning_rate': 1e-3,         # LR for projection head (randomly initialized)
    'backbone_lr': 1e-5,           # LR for DINOv2 backbone (pretrained)
    'weight_decay': 1e-4,
    'patience': 25,
    'lr_scheduler_patience': 10,
    'lr_scheduler_factor': 0.5,
}

# Loss configuration
LOSS_CONFIG = {
    'margin': 0.6,
    'triplet_type': 'hard',  # Options: 'hard', 'semihard', 'all'
}

# DataLoader configuration
DATALOADER_CONFIG = {
    'm_per_class': 4,
    'num_workers': 0 if os.name == 'nt' else 4,
    'pin_memory': True,
}

# Logging
LOG_CONFIG = {
    'log_interval': 10,
    'save_best_only': True,
}
