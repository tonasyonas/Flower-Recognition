import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.2, margin=1.0):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.triplet = TripletLoss(margin=margin)
        self.alpha = alpha
        
    def forward(self, logits, embeddings, labels):
        # Base CE loss
        loss_ce = self.ce(logits, labels)
        
        # Note: True Triplet loss requires mining hard anchors/positives/negatives.
        # For simplicity in this implementation without a custom sampler, 
        # if the batch doesn't have valid triplets, we just return CE.
        
        # TODO: Implement batch-hard triplet mining for final compute
        loss_triplet = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return loss_ce + (self.alpha * loss_triplet)
