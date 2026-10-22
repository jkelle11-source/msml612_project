"""BRITS, the learned baseline (M3-3, owner: Sankha).

Cao et al. (2018), "BRITS: Bidirectional Recurrent Imputation for Time Series", NeurIPS, in plain
PyTorch, matching the original code (github.com/caow13/BRITS, which CSDI's published BRITS numbers
come from) where it differs from the paper: no sigmoid on the mixing weight β, the imputation loss
summed over hours and weighted 0.3, and its initialization of the decay layers. Imputation only:
the original also trains a mortality classifier alongside (``label_weight``), which needs outcome
labels our pipeline doesn't load, so that loss is left out. Each direction (RITS)
reads a patient's 48 hours with an LSTM and, at every hour, guesses every variable *before*
reading it, from the decayed memory (history regression) and from the other variables at the
same hour (feature regression, which never sees the variable itself), mixed by learned weights.
The two directions' imputations are averaged, and a consistency loss pulls them together.

    uv run python -m models.brits --seeds 42 43 44 45 46

trains one model per seed on the training patients, keeps the weights with the best val MAE
(random 10% holdout; the test split never chooses anything), scores them with
``eval.baselines.baseline_results`` on both frozen holdouts, and writes ``LEARNED_PATH``: each
seed's scores and their mean ± SE, in ``results/baselines.json``'s layout. Each seed is also a
wandb run (``brits_seed<seed>``, job type ``baseline``).

**What the model sees.** It is only ever given ``values`` and a mask: the full mask for training
patients (the loss is on visible values, each guessed before it is read, so copying is
impossible) and ``EvalMasker``'s / ``BlockEvalMasker``'s ``cond_mask`` for val and test. The
time gaps it decays by are computed here, from that same mask. The dataset's ``delta_obs`` is
never used: it comes from the full mask, so for a val or test patient it would tell the model
when the held-out values were measured.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "physionet_set_a.npz"
LEARNED_PATH = ROOT / "results" / "learned_baseline.json"
MODEL_NAME = "BRITS (Cao et al. 2018), our implementation"

# Settings from the original BRITS code (github.com/caow13/BRITS): hidden size 64, Adam at 1e-3,
# the imputation loss summed over the 48 hours and weighted 0.3 (its README's recommended
# impute_weight), consistency weight 0.1, and 1000 epochs as the cap. Batch 64 is its README's
# example; its code defaults to 32. Our stopping rule is added on top: best val MAE, stop after
# PATIENCE epochs without improvement, which stops every seed on Set A well before the cap.
HIDDEN = 64
LR = 1.0e-3
BATCH_SIZE = 64
IMPUTE_WEIGHT = 0.3
CONSISTENCY = 0.1
MAX_EPOCHS = 1000
PATIENCE = 20

def time_gaps(mask):
    """BRITS's δ on the hourly grid (Cao et al. 2018, eq. 2): hours since each variable was last
    visible, counted up to the hour before. ``δ_0 = 0``; ``δ_t = 1`` if it was visible at ``t-1``,
    else ``1 + δ_{t-1}``. ``mask`` is ``(B, T, D)``; so is the result."""
    gaps = [torch.zeros_like(mask[:, 0])]
    for t in range(1, mask.shape[1]):
        gaps.append(1 + gaps[-1] * (1 - mask[:, t - 1]))
    return torch.stack(gaps, dim=1)

def masked_l1(pred, truth, mask):
    return ((pred - truth).abs() * mask).sum() / (mask.sum() + 1e-5)

class TemporalDecay(nn.Module):
    """``γ = exp(-relu(W δ + b))``: 1 for a fresh value, toward 0 for a stale one (eq. 3).
    ``diagonal`` lets each variable's decay depend only on its own gap (the input decay γ_x).
    Initialized as the original code does: uniform in ±1/√n_out, weights and bias."""

    def __init__(self, n_in, n_out, diagonal):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        bound = 1 / n_out ** 0.5
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)
        self.register_buffer("keep", torch.eye(n_in) if diagonal else torch.ones(n_out, n_in))

    def forward(self, gaps):
        return torch.exp(-F.relu(F.linear(gaps, self.linear.weight * self.keep, self.linear.bias)))

class FeatureRegression(nn.Module):
    """``ẑ = W x + b`` with W's diagonal zeroed, so a variable is never guessed from itself (eq. 7)."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, n_features)
        self.register_buffer("off_diagonal", 1 - torch.eye(n_features))

    def forward(self, x):
        return F.linear(x, self.linear.weight * self.off_diagonal, self.linear.bias)

class RITS(nn.Module):
    """One direction of BRITS. Returns the imputed grid (visible values kept) and its imputation
    loss: the three guesses' masked L1, summed over the hours, as in the original code."""

    def __init__(self, n_features, hidden):
        super().__init__()
        self.hidden = hidden
        self.decay_h = TemporalDecay(n_features, hidden, diagonal=False)
        self.decay_x = TemporalDecay(n_features, n_features, diagonal=True)
        self.history = nn.Linear(hidden, n_features)
        self.feature = FeatureRegression(n_features)
        self.combine = nn.Linear(2 * n_features, n_features)
        self.cell = nn.LSTMCell(2 * n_features, hidden)

    def forward(self, values, mask):
        gaps = time_gaps(mask)
        h = c = values.new_zeros(len(values), self.hidden)
        loss, filled = 0.0, []
        for t in range(values.shape[1]):
            x, m, d = values[:, t], mask[:, t], gaps[:, t]
            h = h * self.decay_h(d)
            x_hist = self.history(h)                                 # from memory (eq. 4)
            z = self.feature(m * x + (1 - m) * x_hist)               # from the other variables
            # The mix (eq. 8). The paper puts a sigmoid on β; the original code doesn't, and CSDI's
            # published BRITS numbers come from that code, so neither do we.
            beta = self.combine(torch.cat([self.decay_x(d), m], dim=1))
            x_comb = beta * z + (1 - beta) * x_hist
            loss = loss + masked_l1(x_hist, x, m) + masked_l1(z, x, m) + masked_l1(x_comb, x, m)
            x_filled = m * x + (1 - m) * x_comb
            filled.append(x_filled)
            h, c = self.cell(torch.cat([x_filled, m], dim=1), (h, c))
        return torch.stack(filled, dim=1), loss

class BRITS(nn.Module):
    """Both directions. ``forward(values, mask)`` returns ``(imputation, loss)``; boxes where
    ``mask == 0`` are zeroed first, so whatever they hold (a held-out value, NaN) is never read."""

    def __init__(self, n_features=35, hidden=HIDDEN, impute_weight=IMPUTE_WEIGHT, consistency=CONSISTENCY):
        super().__init__()
        self.forwards, self.backwards = RITS(n_features, hidden), RITS(n_features, hidden)
        self.impute_weight, self.consistency = impute_weight, consistency

    def forward(self, values, mask):
        values = torch.where(mask == 1, values, torch.zeros_like(values))
        ahead, loss_ahead = self.forwards(values, mask)
        behind, loss_behind = self.backwards(values.flip(1), mask.flip(1))
        behind = behind.flip(1)
        loss = (self.impute_weight * (loss_ahead + loss_behind)
                + self.consistency * (ahead - behind).abs().mean())
        return (ahead + behind) / 2, loss

def impute(model, values, cond_mask, device, batch_size=256):
    """BRITS's guess for every box, from ``values`` where ``cond_mask == 1`` only: ``(N, 48, D)`` float32."""
    model.eval()
    values = torch.as_tensor(np.asarray(values, dtype=np.float32))
    cond = torch.as_tensor(np.asarray(cond_mask, dtype=np.float32))
    out = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            chunk = slice(start, start + batch_size)
            out.append(model(values[chunk].to(device), cond[chunk].to(device))[0].cpu())
    return torch.cat(out).numpy()

def train(dataset, seed, device, max_epochs=MAX_EPOCHS, patience=PATIENCE, on_epoch_end=None):
    """Train BRITS on the training patients; returns ``(model, curve, best_epoch)``.

    After every epoch the val split's random 10% ``EvalMasker`` targets are scored, and the
    weights with the lowest val MAE are the ones returned. Training stops after ``patience``
    epochs without improvement. The seed fixes the starting weights and the batch order, and
    reruns are bit-identical (deterministic kernels, as in ``diffusion.train``).
    """
    from diffusion.train import use_deterministic_kernels
    from eval import EvalMasker, masked_mae

    use_deterministic_kernels()
    torch.manual_seed(seed)
    model = BRITS().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    order = torch.Generator().manual_seed(seed)

    idx = dataset["train_idx"]
    values = torch.as_tensor(dataset["values"][idx]).to(device)
    mask = torch.as_tensor(dataset["mask"][idx]).to(device)
    val = dataset["val_idx"]
    val_values = dataset["values"][val]
    val_cond, val_target = EvalMasker.from_config()(dataset["mask"][val], dataset["record_ids"][val])

    curve, best, best_state, best_epoch = [], float("inf"), None, -1
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for batch in torch.randperm(len(idx), generator=order).split(BATCH_SIZE):
            optimizer.zero_grad()
            _, loss = model(values[batch], mask[batch])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_mae = masked_mae(impute(model, val_values, val_cond, device), val_values, val_target)
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        curve.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mae": val_mae})
        if on_epoch_end is not None:
            on_epoch_end(curve[-1])
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, curve, best_epoch

def mean_se(per_seed):
    """Mean ± SE over seeds of every ``brits`` score, in ``results/baselines.json``'s layout.

    SE is the sample standard deviation over seeds divided by √n (``None`` for one seed).
    """
    out = json.loads(json.dumps(per_seed[0]))
    for holdout in ("random", "block"):
        for fraction, rows in out[holdout]["fractions"].items():
            for split, row in rows.items():
                for metric in row["brits"]:
                    xs = np.array([s[holdout]["fractions"][fraction][split]["brits"][metric] for s in per_seed])
                    se = float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else None
                    row["brits"][metric] = {"mean": round(float(xs.mean()), 6),
                                            "se": None if se is None else round(se, 6)}
    return out

def run_seeds(dataset, seeds, device, max_epochs=MAX_EPOCHS, wandb_mode=None, log=print):
    """Train and score one BRITS per seed (each a wandb run); returns the ``LEARNED_PATH`` record."""
    import wandb

    from diffusion.run import TRACKING as tracking
    from eval.baselines import baseline_results, wandb_summary

    settings = {"hidden": HIDDEN, "lr": LR, "batch_size": BATCH_SIZE, "impute_weight": IMPUTE_WEIGHT,
                "consistency": CONSISTENCY,
                "max_epochs": max_epochs, "patience": PATIENCE,
                "selection": "best val MAE on EvalMasker's random 10% val targets; test never used"}
    per_seed = {}
    for seed in seeds:
        with wandb.init(entity=tracking["wandb_entity"], project=tracking["wandb_project"], mode=wandb_mode,
                        name=f"brits_seed{seed}", job_type="baseline", tags=["baseline", "brits", f"seed{seed}"],
                        config={**settings, "model": MODEL_NAME, "seed": seed, "device": device}) as run:
            # Per-epoch keys stay apart from the summary's val/mae: wandb keeps the last logged
            # value of a history key as its summary, but the scored weights are the best epoch's.
            def on_epoch_end(row):
                run.log({"epoch/loss_mean": row["loss"], "epoch/val_mae": row["val_mae"]}, step=row["epoch"])
                log(f"brits seed {seed}  epoch {row['epoch'] + 1}  loss {row['loss']:.4f}  val MAE {row['val_mae']:.4f}")

            model, curve, best_epoch = train(dataset, seed, device, max_epochs, on_epoch_end=on_epoch_end)
            scores = baseline_results(dataset, {"brits": lambda v, c: impute(model, v, c, device)})
            per_seed[str(seed)] = {"best_epoch": best_epoch, "epochs_run": len(curve), "scores": scores}
            run.summary.update(wandb_summary(scores, "brits") | {"best_epoch": best_epoch, "epochs_run": len(curve)})
    return {"model": MODEL_NAME, "fallback": False, "settings": settings, "seeds": list(seeds),
            "per_seed": per_seed, "mean_se": mean_se([s["scores"] for s in per_seed.values()])}

def main(argv=None):
    from data.physionet import load_dataset
    from diffusion.run import pick_device

    parser = argparse.ArgumentParser(description="Train and score BRITS, the learned baseline (M3-3).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", default="cpu", help="cpu (default), mps, cuda, or auto")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--out", type=Path, default=LEARNED_PATH)
    parser.add_argument("--wandb", choices=("online", "offline", "disabled"),
                        help="wandb mode (default: WANDB_MODE if set, else online)")
    args = parser.parse_args(argv)
    record = run_seeds(load_dataset(args.dataset), args.seeds, pick_device(args.device), args.max_epochs,
                       args.wandb, log=lambda line: print(line, flush=True))
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["mean_se"]["random"]["fractions"], indent=1))

if __name__ == "__main__":
    main()
