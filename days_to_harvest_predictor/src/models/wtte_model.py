"""
Complementary Model: Weibull Time-To-Event RNN (WTTE-RNN).
Probabilistic survival modeling of harvest event times with confidence intervals.
Outputs Weibull parameters Alpha (Scale) and Beta (Shape).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .lstm_model import BaseEncoder


class WTTERNN(nn.Module):
    """
    Weibull Time-To-Event Recurrent Neural Network.
    """
    def __init__(self, num_dim: int, cat_dims: List[int], hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = BaseEncoder(num_dim, cat_dims, hidden_dim=hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.out_alpha = nn.Linear(hidden_dim, 1)
        self.out_beta = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_emb = self.encoder(x_num, x_cat)
        out, _ = self.lstm(x_emb)
        
        alpha = F.softplus(self.out_alpha(out)).squeeze(-1) + 1e-5
        beta  = F.softplus(self.out_beta(out)).squeeze(-1) + 1e-5
        
        return alpha, beta

    def predict_expected_and_quantiles(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        alpha, beta = self.forward(x_num, x_cat)
        median = alpha * (torch.log(torch.tensor(2.0, device=alpha.device)) ** (1.0 / beta))
        q10    = alpha * ((-torch.log(torch.tensor(0.9, device=alpha.device))) ** (1.0 / beta))
        q90    = alpha * ((-torch.log(torch.tensor(0.1, device=alpha.device))) ** (1.0 / beta))
        return median, q10, q90
