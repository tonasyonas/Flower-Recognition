"""
Combined cross-entropy + batch-hard triplet loss for embedding learning.
Used optionally during VPT-Deep training to encourage tighter per-class embedding clusters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_distances(embeddings, squared=False, eps=1e-12):
    """Compute pairwise Euclidean distances between rows of `embeddings`."""
    # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2
    dot_product = embeddings @ embeddings.t()
    square_norm = torch.diagonal(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = distances.clamp(min=0.0)
    if not squared:
        # Add epsilon under the sqrt to keep gradients finite at zero distance.
        distances = torch.sqrt(distances + eps)
    return distances


def batch_hard_triplet_loss(embeddings, labels, margin=0.5, squared=False, normalize=True):
    """
    Batch-hard triplet loss. For each anchor, use the hardest positive (same class, max distance)
    and hardest negative (different class, min distance) within the batch.

    L2-normalizes embeddings by default for numerical stability under mixed precision.
    Returns a scalar loss. Anchors without valid positives are skipped.
    """
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    distances = _pairwise_distances(embeddings, squared=squared)
    labels = labels.view(-1)

    # Positive and negative masks (shape [B, B])
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    identity = torch.eye(labels.size(0), dtype=torch.bool, device=embeddings.device)
    positive_mask = label_equal & ~identity
    negative_mask = ~label_equal

    # Anchors that have at least one positive partner in the batch.
    valid_anchors = positive_mask.any(dim=1)
    if valid_anchors.sum() == 0:
        return torch.zeros((), device=embeddings.device)

    # Hardest positive: maximum distance within same-class pairs.
    masked_positives = distances.masked_fill(~positive_mask, -float('inf'))
    hardest_positive_dist, _ = masked_positives.max(dim=1)

    # Hardest negative: minimum distance among different-class pairs.
    masked_negatives = distances.masked_fill(~negative_mask, float('inf'))
    hardest_negative_dist, _ = masked_negatives.min(dim=1)

    triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
    # Only average over anchors that had valid positives.
    return triplet_loss[valid_anchors].mean()


class CombinedLoss(nn.Module):
    """
    Cross-entropy + alpha * triplet loss. Triplet loss is computed on the embeddings
    returned as the second output of the model's forward pass.
    """
    def __init__(self, alpha=0.2, margin=0.5, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, embeddings, targets):
        ce_loss = self.ce(logits, targets)
        if self.alpha == 0 or embeddings is None:
            return ce_loss
        triplet = batch_hard_triplet_loss(embeddings, targets, margin=self.margin)
        return ce_loss + self.alpha * triplet
