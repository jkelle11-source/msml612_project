import torch
import torch.nn as nn

def tslo(cond_mask):
    """cond_mask: (B, T, D) float tensor, 1 = visible to the model.
    Returns δ: (B, T, D), same device and dtype as cond_mask."""
    counters = []
    counter = torch.zeros_like(cond_mask[:, 0, :])
    for h in range(cond_mask.shape[1]):
        counter = torch.where(cond_mask[:, h, :] == 1, 0, counter + 1)
        counters.append(counter)
    return torch.stack(counters, dim = 1)

class ContinuousTimeEncoding(nn.Module):
    """Extension 2 (PROJECT_PLAN §4.2, M5-2): true Time2Vec of TSLO δ at each (hour, feature).

    δ, the hours since a feature was last visible to the model, becomes ``d_k`` features: one
    linear feature, ``w0(δ)``, which keeps growing and so says how long, and ``d_k - 1`` waves,
    ``sin(w(δ))``, each with its own learned frequency and phase, which can tell kinds of
    staleness apart. They are concatenated, not summed (a sum of linear terms is still one
    linear term), then projected to ``d_model`` by one ``proj`` shared by every (hour, feature),
    so each value gets its own vector; the denoiser's feature embedding tells features apart.

    The output is scaled by a learned ``gate`` that starts at 0, as Extension 1's does (settled
    in M5-2), so Ext 2 starts out as the base model. δ goes in as raw hours (0 to 48), which
    keeps the waves informative from the start (on δ / 48 they barely vary), but makes the
    untrained output large: on 64 training patients with ``TrainMasker`` masks, its RMS was
    1.26 at ``d_k`` 16, from 0.42 to 3.18 across init seeds, against 0.67 for the denoiser input
    it is added to. Because the gate starts at 0, the gate trains first and the layers before it
    follow once it has moved.

    Takes δ ``(B, T, D)`` in hours and returns ``time_enc`` ``(B, T, D, d_model)``, which the
    model adds to the denoiser input (README, input-hook contract). δ must come from the
    conditioning mask, ``tslo(cond_mask)``, never from ``observed_mask`` or the dataset's
    ``delta_obs``: those still count the held-out targets as observed (PROJECT_PLAN §3.1).
    """

    def __init__(self, d_k, d_model):
        super().__init__()
        self.w0 = nn.Linear(1, 1)
        self.w = nn.Linear(1, d_k - 1)
        self.proj = nn.Linear(d_k, d_model)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, delta):
        d = delta.unsqueeze(-1)     # (B, T, D, 1)
        features = torch.cat([self.w0(d), torch.sin(self.w(d))], 
                             dim = -1)  # (B, T, D, d_k)
        return self.gate * self.proj(features)  # (B, T, D, d_model)
