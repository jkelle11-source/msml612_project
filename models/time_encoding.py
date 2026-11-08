import torch

def tslo(cond_mask):
    """cond_mask: (B, T, D) float tensor, 1 = visible to the model.
    Returns δ: (B, T, D), same device and dtype as cond_mask."""
    counters = []
    counter = torch.zeros_like(cond_mask[:, 0, :])
    for h in range(cond_mask.shape[1]):
        counter = torch.where(cond_mask[:, h, :] == 1, 0, counter + 1)
        counters.append(counter)
    return torch.stack(counters, dim = 1)