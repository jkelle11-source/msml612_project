import numpy as np

def _hidden(pred, truth, target_mask):
    """Validate inputs; return them as arrays plus the boolean mask of scored boxes."""
    pred, truth, target_mask = np.asarray(pred), np.asarray(truth), np.asarray(target_mask)
    if not (pred.shape == truth.shape == target_mask.shape):
        raise ValueError(f"shape mismatch: pred {pred.shape}, truth {truth.shape}, mask {target_mask.shape}")
    hidden = target_mask == 1
    if not hidden.any():
        raise ValueError("target_mask has no hidden boxes to score")
    return pred, truth, hidden

def masked_errors(pred, truth, target_mask):
    """Signed errors ``pred - truth`` at the hidden boxes only, as a flat float64 array.

    Boxes are selected before subtracting, so anything outside the targets (even NaN) is ignored.
    """
    pred, truth, hidden = _hidden(pred, truth, target_mask)
    return pred[hidden].astype(np.float64) - truth[hidden].astype(np.float64)

def masked_mae(pred, truth, target_mask):
    """MAE pooled over every hidden box of every patient (CSDI's convention, PROJECT_PLAN §3.2)."""
    return float(np.abs(masked_errors(pred, truth, target_mask)).mean())

def masked_rmse(pred, truth, target_mask):
    """RMSE pooled over every hidden box of every patient."""
    return float(np.sqrt((masked_errors(pred, truth, target_mask) ** 2).mean()))

def per_variable_mae(pred, truth, target_mask, std=None):
    """MAE per feature (last axis), shape ``(D,)``. NaN where a feature has no hidden boxes.

    Pass the training-split ``std`` to get raw clinical units: undoing z-scoring is
    ``z * std + mean`` and the mean cancels in ``pred - truth``, so raw error = error * std.
    """
    pred, truth, hidden = _hidden(pred, truth, target_mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        abs_err = np.where(hidden, np.abs(pred.astype(np.float64) - truth.astype(np.float64)), 0.0)
        axes = tuple(range(pred.ndim - 1))
        mae = abs_err.sum(axis=axes) / hidden.sum(axis=axes)
    return mae if std is None else mae * np.asarray(std, dtype=np.float64)
