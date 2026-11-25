# Accuracy
# Precision
# Recall
# F1-score
# ROC-AUC
# EER
# FAR / FRR
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, roc_curve
import numpy as np

class MetricCal():
    def __init__(self, device, pair_type) -> None:
        self.device = device
        self.pair_type = pair_type
        self.reset()
    def reset(self):
        self.y_true = []
        self.y_preds = []
        self.y_similarity = []

        self.eer = None
        self.eer_threshold = None
        
        self.loss = torch.zeros(1, device=self.device)
        self.total_batch_size = torch.zeros(1, device=self.device)
    def get_threshold(self):
        dissimilarity = 1 - np.array(self.y_similarity)

        fpr, tpr, thresholds = roc_curve(
            self.y_true,
            dissimilarity,
            pos_label=1
        )

        fnr = 1 - tpr

        eer_index = np.nanargmin(np.abs(fpr - fnr))

        self.eer = (fpr[eer_index] + fnr[eer_index]) / 2

        self.eer_threshold = thresholds[eer_index]

        # Generate predictions using EER threshold
        predictions = (
            dissimilarity >= self.eer_threshold
        ).astype(int)

        self.y_preds = predictions.tolist()

        return self.eer, self.eer_threshold
    @torch.no_grad()
    def update(self, loss, embs1, embs2, embs3=None, labels=None):
        batch_size = embs1.size(0)
        self.loss += loss.detach() * batch_size
        self.total_batch_size += batch_size
        match self.pair_type:
            case 1: #Constraive Loss
                similarity = F.cosine_similarity(
                    embs1, embs2, dim=1
                )
                self.y_true.extend(labels.detach().cpu().tolist())

                self.y_similarity.extend(
                    similarity.detach().cpu().tolist()
                )
            case _:
                raise ValueError(
                    f"Invalid pair_type: {self.pair_type}. Expected 1 or 2."
                )
    @property
    def avg_loss(self):
        """Average loss over all accumulated batches."""
        return (self.loss / self.total_batch_size).item() if self.total_batch_size > 0 else 0.0
    
    @property
    def ROC_AUC(self):
        """
        AUC ≈ 0.5 → useless
        AUC > 0.95 → strong
        AUC > 0.99 → excellent
        """
        return roc_auc_score(
            self.y_true,
            1 - np.array(self.y_similarity),
            average="macro",
        )
    @property
    def Accuracy_score(self):
        """Accuracy score over all accumulated batches."""
        return accuracy_score(y_pred=self.y_preds, y_true=self.y_true)
    
    @property
    def Precision_score(self):
        """Accuracy score over all accumulated batches."""
        return precision_score(y_pred=self.y_preds, y_true=self.y_true, average="macro")
    
    @property
    def Recall_score(self):
        """Accuracy score over all accumulated batches."""
        return recall_score(y_pred=self.y_preds, y_true=self.y_true, average="macro")
    
    @property
    def F1_score(self):
        """Accuracy score over all accumulated batches."""
        return f1_score(y_pred=self.y_preds, y_true=self.y_true, average="macro")
    
    
