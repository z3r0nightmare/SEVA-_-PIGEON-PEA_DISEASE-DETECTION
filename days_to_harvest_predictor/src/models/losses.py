"""
Loss Functions for Crop Phenology and Harvest Date Prediction.
Includes Masked MSE Loss, Masked Huber/SmoothL1 Loss, and Weibull Negative Log-Likelihood (WTTE).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error computed only on valid (non-padded) sequence steps.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        mask = (targets != -1.0).float()
        sq_err = (preds - targets) ** 2
        
        if weights is not None:
            sq_err = sq_err * weights
            
        loss = (sq_err * mask).sum() / (mask.sum() + self.eps)
        return loss


class MaskedMAELoss(nn.Module):
    """
    Weighted Mean Absolute Error computed only on valid (non-padded) sequence steps.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        mask = (targets != -1.0).float()
        abs_err = torch.abs(preds - targets)
        
        if weights is not None:
            abs_err = abs_err * weights
            
        loss = (abs_err * mask).sum() / (mask.sum() + self.eps)
        return loss


class MaskedHuberLoss(nn.Module):
    """
    Smooth L1 / Huber Loss robust to noisy pseudo-labels and outlier cloudy observations.
    """
    def __init__(self, beta: float = 2.0, eps: float = 1e-5):
        super().__init__()
        self.beta = beta
        self.eps = eps
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        mask = (targets != -1.0).float()
        huber = F.smooth_l1_loss(preds, targets, beta=self.beta, reduction='none')
        
        if weights is not None:
            huber = huber * weights
            
        loss = (huber * mask).sum() / (mask.sum() + self.eps)
        return loss


def wtte_nll_loss(y_true: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, 
                  mask: torch.Tensor, weights: torch.Tensor = None, eps: float = 1e-5) -> torch.Tensor:
    """
    Negative Log-Likelihood of the Continuous Weibull Time-To-Event Distribution.
    
    Probability Density Function:
        f(t) = (beta / alpha) * (t / alpha)^(beta - 1) * exp(-(t / alpha)^beta)
    
    Log-Likelihood:
        log f(t) = log(beta) - log(alpha) + (beta - 1)*(log(t) - log(alpha)) - (t / alpha)^beta
    """
    y_safe = torch.clamp(y_true, min=1e-4)
    
    term1 = torch.log(beta + eps) - torch.log(alpha + eps)
    term2 = (beta - 1.0) * (torch.log(y_safe) - torch.log(alpha + eps))
    term3 = -torch.pow(y_safe / (alpha + eps), beta)
    
    log_lik = term1 + term2 + term3
    
    if weights is not None:
        weighted_nll = -log_lik * mask * weights
    else:
        weighted_nll = -log_lik * mask
        
    return weighted_nll.sum() / (mask.sum() + eps)
