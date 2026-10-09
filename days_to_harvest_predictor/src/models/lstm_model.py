"""
Primary Model: Causal LSTM Phenology Regressor.
Strictly causal sequence-to-sequence neural network for predicting Days to Harvest (DTH).
Zero temporal leakage: uses unidirectional recurrence and causal padding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BaseEncoder(nn.Module):
    """
    Encodes dense numeric features and discrete categorical features (district, zone, variety, phase)
    into a joint latent representation via learnable entity embeddings.
    """
    def __init__(self, num_dim: int, cat_dims: List[int], embed_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim) for num_classes in cat_dims
        ])
        
        in_dim = num_dim + len(cat_dims) * embed_dim
        self.proj = nn.Linear(in_dim, hidden_dim)
        
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_num shape: (Batch, Seq_Len, Num_Features)
        # x_cat shape: (Batch, Seq_Len, Num_Categoricals)
        embeds = []
        for i, emb in enumerate(self.embeddings):
            cat_idx = torch.clamp(x_cat[:, :, i], min=0, max=emb.num_embeddings - 1)
            embeds.append(emb(cat_idx))
            
        if embeds:
            x_cat_emb = torch.cat(embeds, dim=-1)
            x_in = torch.cat([x_num, x_cat_emb], dim=-1)
        else:
            x_in = x_num
            
        return F.relu(self.proj(x_in))


class LSTMRegressor(nn.Module):
    """
    Primary SOTA Architecture: Deep Causal LSTM Phenology Regressor.
    
    Architectural Highlights:
    1. Base Joint Embedding: Projects spectral vegetation indices (NDVI, NDWI, LSWI, REP, BSI),
       SAR radar backscatter (VH, VV, VH/VV ratio), agrometeorology (GDD, VPD, precip),
       and spatial categoricals into a 64-dimensional latent embedding.
    2. 2-Layer Unidirectional LSTM: Recurrently processes the temporal trajectory up to
       current timestep t without ever attending to future observations.
    3. Regression Head: Fully connected linear projection to continuous scalar 'Days to Harvest'.
    """
    def __init__(self, num_dim: int, cat_dims: List[int], hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = BaseEncoder(num_dim, cat_dims, hidden_dim=hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass over the sequence.
        Returns:
            preds: (Batch, Seq_Len) continuous days-to-harvest predictions.
        """
        x_emb = self.encoder(x_num, x_cat)
        out, _ = self.lstm(x_emb)
        preds = self.out(out).squeeze(-1)
        return preds

    def predict_step(self, x_num_step: torch.Tensor, x_cat_step: torch.Tensor, hidden_state=None):
        """
        Online step-by-step real-time inference.
        """
        if x_num_step.ndim == 2:
            x_num_step = x_num_step.unsqueeze(1)
            x_cat_step = x_cat_step.unsqueeze(1)
            
        x_emb = self.encoder(x_num_step, x_cat_step)
        out, next_hidden = self.lstm(x_emb, hidden_state)
        pred = self.out(out).squeeze(-1).squeeze(-1)
        return pred, next_hidden
