import os
import torch

# Force CUDA initialization before any other imports
if torch.cuda.is_available():
    # Force CUDA initialization by accessing device properties
    _ = torch.cuda.device_count()
    print(f"[INIT] CUDA initialized: {torch.cuda.get_device_name(0)}")
else:
    print("[INIT] CUDA not available")

import torch.optim as optim
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner
from pytorch_metric_learning.distances import CosineSimilarity
from tqdm import tqdm
from dataloader import train_loader, val_loader
from model import ResNet50_Embedder, DINOv2_Embedder
from config import MODEL_CONFIG, TRAIN_CONFIG, LOSS_CONFIG

# Configuration from config.py
backbone = MODEL_CONFIG.get('backbone', 'resnet50')
CONFIG = {
    'backbone': backbone,
    'embedding_dim': MODEL_CONFIG['embedding_dim'],
    'learning_rate': TRAIN_CONFIG['learning_rate'],
    'num_epochs': TRAIN_CONFIG['num_epochs'],
    'margin': LOSS_CONFIG['margin'],
    'patience': TRAIN_CONFIG['patience'],
    'save_dir': 'checkpoints',
    'model_name': f'{backbone}_metric_best.pth'
}

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create checkpoint directory
os.makedirs(CONFIG['save_dir'], exist_ok=True)

# Model
if backbone == 'resnet50':
    model = ResNet50_Embedder(
        embedding_dim=CONFIG['embedding_dim'],
        dropout=MODEL_CONFIG.get('dropout', 0.3)
    ).to(device)
else:
    model = DINOv2_Embedder(
        embedding_dim=CONFIG['embedding_dim'],
        model_name=backbone,
        dropout=MODEL_CONFIG.get('dropout', 0.2),
        freeze_backbone=MODEL_CONFIG.get('freeze_backbone', False)
    ).to(device)

# Loss and Miner (use cosine similarity for L2-normalized embeddings)
loss_func = TripletMarginLoss(margin=CONFIG['margin'], distance=CosineSimilarity())
miner = BatchHardMiner(distance=CosineSimilarity())

# Optimizer with differential learning rates
if backbone != 'resnet50':
    # DINOv2: lower LR for pretrained backbone, higher LR for projection head
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.embedding.parameters())
    if hasattr(model, 'dropout') and hasattr(model.dropout, 'parameters'):
        head_params += list(model.dropout.parameters())
    backbone_lr = TRAIN_CONFIG.get('backbone_lr', 1e-5)
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': CONFIG['learning_rate']},
    ], weight_decay=TRAIN_CONFIG['weight_decay'])
    print(f"Differential LR: backbone={backbone_lr:.1e}, head={CONFIG['learning_rate']:.1e}")
else:
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=TRAIN_CONFIG['lr_scheduler_factor'],
    patience=TRAIN_CONFIG['lr_scheduler_patience']
)
last_lr = CONFIG['learning_rate']

# Training and validation functions
def train_epoch(model, loader, loss_func, miner, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training')

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(images)
        hard_triplets = miner(embeddings, labels)
        loss = loss_func(embeddings, labels, hard_triplets)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)

def validate(model, loader, loss_func, miner, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            hard_triplets = miner(embeddings, labels)
            loss = loss_func(embeddings, labels, hard_triplets)
            total_loss += loss.item()

    return total_loss / len(loader)

# Training loop with early stopping
best_val_loss = float('inf')
patience_counter = 0

print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
print(f"Model: {CONFIG['backbone']}, Embedding dim: {CONFIG['embedding_dim']}, LR: {CONFIG['learning_rate']}\n")

for epoch in range(CONFIG['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")

    # Train
    train_loss = train_epoch(model, train_loader, loss_func, miner, optimizer, device)

    # Validate
    val_loss = validate(model, val_loader, loss_func, miner, device)

    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != last_lr:
        print(f"Learning rate reduced: {last_lr:.2e} -> {current_lr:.2e}")
        last_lr = current_lr

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_path = os.path.join(CONFIG['save_dir'], CONFIG['model_name'])
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': CONFIG
        }, save_path)
        print(f"✓ Best model saved (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{CONFIG['patience']})")

    # Early stopping
    if patience_counter >= CONFIG['patience']:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break

print(f"\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: {os.path.join(CONFIG['save_dir'], CONFIG['model_name'])}")
