"""
Evaluate test set performance against ALL images (train+val+test) using embeddings
Calculates mAP using the same method as mAP.py in root folder
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from model import ResNet50_Embedder
from torch.utils.data import DataLoader
from dataset import ArtistImageDataset
from torchvision import transforms

# Configuration
CHECKPOINT_PATH = os.path.join('checkpoints', '12.5_resnet50_metric_best.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "labels", "final_labels2_train.csv")
VAL_CSV = os.path.join(BASE_DIR, "labels", "final_labels2_val.csv")
TEST_CSV = os.path.join(BASE_DIR, "labels", "final_labels2_test.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "Other_Marks")


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = ResNet50_Embedder(embedding_dim=checkpoint['config']['embedding_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def load_all_data():
    """Load all images (train+val+test) and their labels"""
    import pandas as pd
    
    # Evaluation transform (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load all CSVs and create GLOBAL label mapping
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Combine all artists to create consistent label mapping
    all_artists = pd.concat([train_df['artist'], val_df['artist'], test_df['artist']]).unique()
    artist_to_label = {artist: idx for idx, artist in enumerate(sorted(all_artists))}
    
    # Load all datasets
    train_dataset = ArtistImageDataset(TRAIN_CSV, IMAGE_DIR, transform=eval_transform)
    val_dataset = ArtistImageDataset(VAL_CSV, IMAGE_DIR, transform=eval_transform)
    test_dataset = ArtistImageDataset(TEST_CSV, IMAGE_DIR, transform=eval_transform)
    
    # Create combined dataloader
    from torch.utils.data import ConcatDataset
    all_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    
    all_loader = DataLoader(
        all_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    # Track which images are test images
    test_filenames = set(test_df['filename'].str.strip().tolist())
    
    # Get all filenames
    all_filenames = (
        train_dataset.filenames + 
        val_dataset.filenames + 
        test_dataset.filenames
    )
    
    # Create CONSISTENT labels using global mapping
    all_artists_list = (
        train_df['artist'].tolist() + 
        val_df['artist'].tolist() + 
        test_df['artist'].tolist()
    )
    all_labels = np.array([artist_to_label[artist] for artist in all_artists_list])
    
    # Mark test indices
    test_indices = [i for i, fname in enumerate(all_filenames) if fname in test_filenames]
    
    return all_loader, all_labels, test_indices, all_filenames


def extract_embeddings(model, dataloader, device):
    """Extract embeddings for all images"""
    embeddings = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc='Extracting embeddings', ncols=80):
            images = images.to(device)
            batch_embeddings = model(images)
            embeddings.append(batch_embeddings.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def compute_map(similarity_matrix, labels, query_indices, top_k=100):
    """
    Compute mean Average Precision for test queries against top-k results
    Matches the calculation method from mAP.py in root folder
    
    Args:
        similarity_matrix: (N, N) similarity scores
        labels: (N,) ground truth labels
        query_indices: List of indices for test queries
        top_k: Number of top results to consider (default: 100)
    """
    average_precisions = []
    
    for i in query_indices:
        query_label = labels[i]
        
        # Create mask excluding the query itself
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[i] = False
        
        # Get similarities and labels excluding query
        similarities = similarity_matrix[i][mask].cpu().numpy()
        relevant_mask = (labels[mask] == query_label).cpu().numpy()
        n_relevant = relevant_mask.sum()
        
        if n_relevant == 0:
            continue
        
        # Sort by similarity (descending) - this gives us ranked retrieval
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_relevant = relevant_mask[sorted_indices][:top_k]  # Limit to top-k results
        
        # Calculate AP using same method as mAP.py:
        # For each relevant item found in top-k, add precision at that rank
        num_relevant_seen = 0
        precision_sum = 0.0
        
        for rank, is_relevant in enumerate(sorted_relevant, start=1):
            if is_relevant:
                num_relevant_seen += 1
                precision_at_k = num_relevant_seen / rank
                precision_sum += precision_at_k
        
        # Average over all relevant items (matches mAP.py line 162)
        ap = precision_sum / n_relevant
        average_precisions.append(ap)
    
    mean_ap = np.mean(average_precisions) if average_precisions else 0.0
    return mean_ap


def main():
    """Main evaluation function - simplified to show only mAP"""
    # Load model
    model = load_model(CHECKPOINT_PATH, DEVICE)
    
    # Load all data
    all_loader, all_labels, test_indices, all_filenames = load_all_data()
    
    # Extract embeddings for ALL images
    embeddings = extract_embeddings(model, all_loader, DEVICE)
    
    # Convert labels to tensor
    all_labels = torch.tensor(all_labels)
    
    # Compute similarity matrix (cosine similarity via dot product of L2-normalized embeddings)
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    
    # Compute mAP for different top-k values
    map_50 = compute_map(similarity_matrix, all_labels, test_indices, top_k=50)
    map_100 = compute_map(similarity_matrix, all_labels, test_indices, top_k=100)
    map_150 = compute_map(similarity_matrix, all_labels, test_indices, top_k=150)
    map_full = compute_map(similarity_matrix, all_labels, test_indices, top_k=len(embeddings))
    
    # Print results
    print(f"mAP@50:  {map_50:.4f}")
    print(f"mAP@100: {map_100:.4f}")
    print(f"mAP@150: {map_150:.4f}")
    print(f"mAP (full): {map_full:.4f}")
    
    return map_100


if __name__ == '__main__':
    main()
