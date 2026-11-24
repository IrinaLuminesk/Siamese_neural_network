import torch

import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, pair_type, margin) -> None:
        super(LossFunction, self).__init__()
        self.pair_type = pair_type
        self.margin = margin

    def forward(self, embs1, embs2, embs3=None, label=None):
        match self.pair_type:
            case 1:
                return self.Contrastive_Loss(embs1=embs1, embs2=embs2, labels=label)
            case 2:
                return self.Triplet_Loss(embs1=embs1, embs2=embs2, embs3=embs3)
            case _:
                raise ValueError(
                    f"Invalid pair_type: {self.pair_type}. Expected 1 or 2."
                )
    def Contrastive_Loss(self, embs1, embs2, labels):
        """
        Computes Contrastive Loss
        """
        dist = F.pairwise_distance(embs1, embs2)

        loss = (1 - labels) * torch.pow(dist, 2) \
            + (labels) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.mean(loss)

        return loss
    
    def Triplet_Loss(self, embs1, embs2, embs3):
        """
        Computes Triplet Loss
        """
        pos_dist = F.pairwise_distance(embs1, embs2)
        neg_dist = F.pairwise_distance(embs1, embs3)

        loss = torch.clamp(
            pos_dist - neg_dist + self.margin,
            min=0.0
        )
        loss = loss.mean()


        return loss
