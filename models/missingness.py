import torch
import torch.nn as nn

class MissingnessEncoder(nn.Module):
    """Extension 1 (PROJECT_PLAN §4.1, M4-2): dual-axis embedding of the conditioning mask.

    One token per (hour, feature): the mask bit, projected to ``d_model``, plus a learned
    feature embedding (which feature the token is) and a learned hour embedding (which hour,
    so temporal attention can tell contiguous hours from scattered ones). Each layer pair
    attends over time within every feature, then over features within every hour, so a
    cross-feature structure like block missingness is visible to it, which a temporal-only
    encoder could not see.

    The output is scaled by a learned ``gate`` that starts at 0 (M4-2 review), so Ext 1 starts
    out as the base model and the encoder's output enters only as far as training finds it
    useful. Without it, the untrained encoder's output (RMS 1.0 per value) would outweigh the
    denoiser input it is added to (RMS 0.62) from the first step. Because the gate starts at 0,
    the gate itself trains first and the layers before it follow once it has moved.

    Takes the conditioning mask ``(B, T, D)`` and returns ``mask_emb`` ``(B, T, D, d_model)``,
    which the model adds to the denoiser input (README, input-hook contract). It must only
    ever be given the conditioning mask, never ``observed_mask``: that one still marks the
    held-out targets (PROJECT_PLAN §3.1).
    """

    def __init__(self, n_features, n_timesteps, d_model, n_heads=2, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)                 # one token per (hour, feature)
        self.feature_emb = nn.Embedding(n_features, d_model)    # which feature a token is
        self.time_emb = nn.Embedding(n_timesteps, d_model)      # which hour it is
        self.temporal = nn.ModuleList([         # dim_feedforward=64, as in the denoiser's layers
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=64, batch_first=True)
            for _ in range(n_layers)])
        self.feature = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=64, batch_first=True)
            for _ in range(n_layers)])
        self.gate = nn.Parameter(torch.zeros(()))   # starts closed: Ext 1 starts as base

    def forward(self, mask):                    # mask: (B, T, D) float
        B, T, D = mask.shape
        x = (self.input_proj(mask.unsqueeze(-1))        # (B, T, D, d_model)
             + self.feature_emb.weight                  # (D, d_model)
             + self.time_emb.weight[:, None])           # (T, 1, d_model)
        for t_layer, f_layer in zip(self.temporal, self.feature):
            x = t_layer(x.transpose(1, 2).reshape(B * D, T, -1))           # attend over time
            x = x.reshape(B, D, T, -1).transpose(1, 2)
            x = f_layer(x.reshape(B * T, D, -1)).reshape(B, T, D, -1)      # attend over features
        return self.gate * x                    # (B, T, D, d_model), added to the denoiser input
