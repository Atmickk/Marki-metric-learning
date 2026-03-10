"""
Evaluation script for metric learning model
Computes Accuracy@K and mean Average Precision (mAP) metrics
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from model import ResNet50_Embedder
from dataloader import test_loader, val_loader

# Configuration
CHECKPOINT_PATH = os.path.join('random_search_results', 'resnet50_metric_best.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_VALUES = [1, 10]  # For Accuracy@K


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Support both regular checkpoints ('config') and random search checkpoints ('params')
    cfg = checkpoint.get('config') or checkpoint.get('params', {})
    embedding_dim = cfg.get('embedding_dim', 512)

    # Initialize model
    model = ResNet50_Embedder(embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}\n")

    return model


def extract_embeddings(model, dataloader, device):
    """Extract embeddings for all images in dataloader"""
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in tqdm(dataloader, desc='Extracting embeddings'):
            images = images.to(device)
            batch_embeddings = model(images)

            embeddings.append(batch_embeddings.cpu())
            labels.append(batch_labels)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    return embeddings, labels


def compute_similarity_matrix(embeddings):
    """Compute cosine similarity matrix for L2-normalized embeddings"""
    # Embeddings are already L2-normalized, so dot product = cosine similarity
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    return similarity_matrix


def compute_accuracy_at_k(similarity_matrix, labels, k_values):
    """
    Compute Accuracy@K for each query
    Accuracy@K = 1 if at least one relevant item is in top-K, else 0

    Args:
        similarity_matrix: (N, N) similarity scores
        labels: (N,) ground truth labels
        k_values: list of K values to compute accuracy for

    Returns:
        Dictionary with accuracy@k for each k value
    """
    n_queries = similarity_matrix.size(0)
    accuracies = {k: [] for k in k_values}

    for i in range(n_queries):
        # Get similarities for this query (excluding itself)
        query_sims = similarity_matrix[i].clone()
        query_sims[i] = -float('inf')  # Exclude self

        # Get ground truth: which images have the same label
        query_label = labels[i]
        relevant_mask = (labels == query_label)
        relevant_mask[i] = False  # Exclude self
        n_relevant = relevant_mask.sum().item()

        if n_relevant == 0:
            continue  # Skip if no relevant items

        # Get top-k predictions
        for k in k_values:
            _, top_k_indices = torch.topk(query_sims, min(k, len(query_sims)))

            # Check if ANY relevant item is in top-k
            has_relevant = relevant_mask[top_k_indices].any().item()
            
            # Accuracy@K = 1 if at least one relevant, else 0
            accuracies[k].append(1.0 if has_relevant else 0.0)

    # Average across all queries
    mean_accuracies = {k: np.mean(v) if v else 0.0 for k, v in accuracies.items()}
    n_queries_used = len(accuracies[k_values[0]]) if k_values and accuracies[k_values[0]] else 0
    return mean_accuracies, n_queries_used


def compute_map(similarity_matrix, labels):
    """
    Compute traditional mean Average Precision (mAP)
    
    This computes FULL mAP over the entire ranking (not truncated at any K).
    For each query:
      1. Rank ALL database items by similarity
      2. Compute precision at each position where a relevant item appears
      3. Average these precisions over all relevant items
    Then average AP across all queries.

    Args:
        similarity_matrix: (N, N) similarity scores
        labels: (N,) ground truth labels

    Returns:
        Mean average precision score (full ranking, no cutoff)
    """
    n_queries = similarity_matrix.size(0)
    average_precisions = []

    for i in range(n_queries):
        # Get ground truth: which images have the same label
        query_label = labels[i]

        # Create mask excluding the query itself
        mask = torch.ones(n_queries, dtype=torch.bool)
        mask[i] = False

        # Get similarities and labels excluding query
        similarities = similarity_matrix[i][mask].cpu().numpy()
        relevant_mask = (labels[mask] == query_label).cpu().numpy()
        n_relevant = relevant_mask.sum()

        if n_relevant == 0:
            continue  # Skip if no relevant items

        # Sort ALL results by similarity (no truncation - full ranking)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_relevant = relevant_mask[sorted_indices]

        # Calculate AP over the entire ranking
        num_relevant_seen = 0
        precision_sum = 0.0

        for rank, is_relevant in enumerate(sorted_relevant, start=1):
            if is_relevant:
                num_relevant_seen += 1
                precision_at_k = num_relevant_seen / rank
                precision_sum += precision_at_k

        # Average over all relevant items
        ap = precision_sum / n_relevant
        average_precisions.append(ap)

    # Mean over all queries
    mean_ap = np.mean(average_precisions) if average_precisions else 0.0
    n_queries_used = len(average_precisions)
    return mean_ap, n_queries_used


def evaluate(model, dataloader, device, dataset_name='Test'):
    """
    Evaluate model on a dataset

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        dataset_name: Name of dataset for logging
    """
    print(f"\n{'='*50}")
    print(f"Evaluating on {dataset_name} Set")
    print(f"{'='*50}\n")

    # Extract embeddings
    embeddings, labels = extract_embeddings(model, dataloader, device)
    n_embeddings = embeddings.shape[0]
    n_classes = len(torch.unique(labels))
    print(f"Extracted {n_embeddings} embeddings from {n_classes} classes\n")

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Compute mAP
    print("Computing mAP...")
    map_score, map_queries = compute_map(similarity_matrix, labels)
    print(f"  mAP: {map_score:.4f} ({map_score*100:.2f}%) over {map_queries} queries")

    # Compute Accuracy@K
    print("\nComputing Accuracy@K...")
    accuracies, acc_queries = compute_accuracy_at_k(similarity_matrix, labels, K_VALUES)

    print(f"Accuracy@K Results (over {acc_queries} queries):")
    for k, accuracy in accuracies.items(): 
        print(f"  Accuracy@{k:2d}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return {
        'embeddings': embeddings,
        'labels': labels,
        'accuracies': accuracies,
        'map': map_score
    }


def main():
    """Main evaluation function"""
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please train the model first using train.py")
        return

    # Load model
    print(f"Loading model from {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH, DEVICE)

    # Evaluate on test set if available, otherwise use validation set
    if test_loader is not None:
        results = evaluate(model, test_loader, DEVICE, dataset_name='Test')
    elif val_loader is not None:
        print("Test set not found. Evaluating on validation set...")
        results = evaluate(model, val_loader, DEVICE, dataset_name='Validation')
    else:
        print("Error: No test or validation data available!")
        return

    print(f"\n{'='*50}")
    print("Evaluation Complete!")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
