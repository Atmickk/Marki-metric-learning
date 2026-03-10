"""
Random Search for Hyperparameter Tuning
Trains multiple models with randomly sampled hyperparameters
and evaluates each on the validation set.
"""
import os
import json
import random
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import ResNet50_Embedder
from dataset import ArtistImageDataset
from evaluate import extract_embeddings, compute_similarity_matrix, compute_map, compute_accuracy_at_k

# ──────────────────────────────────────────────────────────────
# Search Space – adjust ranges to your needs
# ──────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    'learning_rate':    (1e-5, 1e-3, 'log'),       # log-uniform
    'weight_decay':     (1e-5, 1e-2, 'log'),       # log-uniform
    'dropout':          (0.0, 0.7, 'uniform'),     # uniform
    'margin':           (0.2, 1.0, 'uniform'),     # uniform
    'embedding_dim':    [128, 256, 512],            # categorical
    'batch_size':       [16, 32, 64],               # categorical
    'm_per_class':      [2, 4, 8],                  # categorical
    'lr_scheduler_factor':   (0.3, 0.8, 'uniform'),
    'lr_scheduler_patience': [3, 5, 8, 10],        # categorical
}

# ──────────────────────────────────────────────────────────────
# General settings
# ──────────────────────────────────────────────────────────────
NUM_TRIALS = 20              # how many random configs to try
MAX_EPOCHS = 60              # max epochs per trial (early stopping will cut short)
PATIENCE = 15                # early stopping patience per trial
K_VALUES = [1, 10]           # Accuracy@K values to report
RESULTS_DIR = 'random_search_results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "labels", "final_labels2_train.csv")
VAL_CSV = os.path.join(BASE_DIR, "labels", "final_labels2_val.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "Manually_cropped_marks")
IMAGE_SIZE = 224


def sample_hyperparams():
    """Sample one random configuration from the search space."""
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if isinstance(spec, list):
            # Categorical
            params[name] = random.choice(spec)
        else:
            lo, hi, scale = spec
            if scale == 'log':
                params[name] = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
            else:
                params[name] = float(np.random.uniform(lo, hi))
    return params


def build_dataloaders(params):
    """Build train/val dataloaders with the given batch_size and m_per_class."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ArtistImageDataset(TRAIN_CSV, IMAGE_DIR, transform=train_transform)
    val_dataset = ArtistImageDataset(VAL_CSV, IMAGE_DIR, transform=val_transform)

    m = params['m_per_class']
    bs = params['batch_size']
    nw = 0 if os.name == 'nt' else 4

    train_sampler = MPerClassSampler(train_dataset.labels, m=m,
                                     length_before_new_iter=len(train_dataset))
    val_sampler = MPerClassSampler(val_dataset.labels, m=m,
                                   length_before_new_iter=len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=bs, sampler=train_sampler,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, sampler=val_sampler,
                            num_workers=nw, pin_memory=True, drop_last=True)
    return train_loader, val_loader


def run_trial(trial_id, params):
    """Train + evaluate one configuration. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"Trial {trial_id} — Hyperparameters:")
    for k, v in params.items():
        print(f"  {k:25s}: {v}")
    print(f"{'='*60}")

    # Build data
    train_loader, val_loader = build_dataloaders(params)

    # Model
    model = ResNet50_Embedder(
        embedding_dim=int(params['embedding_dim']),
        pretrained=True,
        dropout=params['dropout']
    ).to(DEVICE)

    # Loss & miner
    loss_func = TripletMarginLoss(margin=params['margin'],
                                  distance=CosineSimilarity())
    miner = BatchHardMiner(distance=CosineSimilarity())

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(),
                           lr=params['learning_rate'],
                           weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=params['lr_scheduler_factor'],
        patience=int(params['lr_scheduler_patience'])
    )

    # ── Training loop ──
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f'  T{trial_id} E{epoch} train', leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            embeddings = model(images)
            hard_triplets = miner(embeddings, labels)
            loss = loss_func(embeddings, labels, hard_triplets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'  T{trial_id} E{epoch} val  ', leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                embeddings = model(images)
                hard_triplets = miner(embeddings, labels)
                loss = loss_func(embeddings, labels, hard_triplets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            # Save best checkpoint for this trial
            ckpt_path = os.path.join(RESULTS_DIR, f"trial_{trial_id}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'params': params,
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  best={best_val_loss:.4f}")

    # ── Evaluate best model on val set ──
    ckpt_path = os.path.join(RESULTS_DIR, f"trial_{trial_id}_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, val_loader_eval = build_dataloaders(params)
    embeddings, labels = extract_embeddings(model, val_loader_eval, DEVICE)
    sim_matrix = compute_similarity_matrix(embeddings)
    map_score, _ = compute_map(sim_matrix, labels)
    accuracies, _ = compute_accuracy_at_k(sim_matrix, labels, K_VALUES)

    result = {
        'trial_id': trial_id,
        'params': {k: (v if not isinstance(v, float) else round(v, 8))
                   for k, v in params.items()},
        'best_val_loss': round(best_val_loss, 6),
        'best_epoch': best_epoch,
        'mAP': round(map_score, 6),
    }
    for k, v in accuracies.items():
        result[f'accuracy@{k}'] = round(v, 6)

    print(f"\n  ► Trial {trial_id} done — "
          f"val_loss={best_val_loss:.4f}  mAP={map_score:.4f}  "
          f"Acc@1={accuracies.get(1, 0):.4f}  Acc@10={accuracies.get(10, 0):.4f}")

    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Running {NUM_TRIALS} random trials, max {MAX_EPOCHS} epochs each\n")

    all_results = []

    for trial_id in range(1, NUM_TRIALS + 1):
        params = sample_hyperparams()
        result = run_trial(trial_id, params)
        all_results.append(result)

        # Save results after every trial (crash-safe)
        results_file = os.path.join(RESULTS_DIR, 'all_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("RANDOM SEARCH COMPLETE — Top 5 Trials by mAP")
    print(f"{'='*60}")
    ranked = sorted(all_results, key=lambda r: r['mAP'], reverse=True)
    for i, r in enumerate(ranked[:5], 1):
        print(f"\n  #{i}  Trial {r['trial_id']}  "
              f"mAP={r['mAP']:.4f}  val_loss={r['best_val_loss']:.4f}")
        for k, v in r['params'].items():
            print(f"      {k:25s}: {v}")

    # Save best config separately for easy reuse
    best = ranked[0]
    best_file = os.path.join(RESULTS_DIR, 'best_config.json')
    with open(best_file, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"\nBest config saved to {best_file}")


if __name__ == '__main__':
    main()
