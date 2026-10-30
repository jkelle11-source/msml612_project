"""Extension 1's overfit sanity check (M4-2): can the model with the MissingnessEncoder fit a tiny subset?

    uv run python -m diffusion.overfit

Takes the first ``N_PATIENTS`` patients of the train split, splits each with ``BlockEvalMasker``
(the frozen block holdout, M2-4), and trains the Ext 1 model for ``EPOCHS`` epochs on exactly
that split: the conditioning mask is the block holdout's, so the loss is taken on the very
targets the check scores. That is the point of an overfit check, and it is why the number says
nothing about how Ext 1 imputes unseen values. It then scores the block targets with the frozen
sampler (``EvalSampler.from_config()``: 2021, 100 draws, median) and passes at MAE below
``MAE_BAR``, standardized. It also records the encoder's gradient norm on the last step, which
must be nonzero for the encoder to train at all.

Departure from the issue's wording, recorded in the output: 10 patients are a single batch, so
an epoch of the training loop would be one optimizer step, and 50 such steps leave a diffusion
model far from fitting anything. Each epoch here takes ``steps_per_epoch`` steps on the same
batch. Everything else is the training loop's: Adam with the config's lr and weight decay, its
learning-rate drops at the same fractions of the epochs, the run's random streams.

Writes ``results/ext1_overfit.json``.
"""

import argparse
import copy
import json
import time

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from data.physionet import load_dataset
from diffusion.data import to_batch
from diffusion.run import CONFIG_PATH, DATASET_PATH, ROOT, pick_device
from diffusion.streams import RunStreams
from diffusion.train import use_deterministic_kernels
from eval import BlockEvalMasker, EvalSampler, masked_mae, masked_rmse, median_imputation
from models.build import build_model

OVERFIT_PATH = ROOT / "results" / "ext1_overfit.json"
N_PATIENTS = 10
EPOCHS = 50
STEPS_PER_EPOCH = 20
MAE_BAR = 0.1

def grad_norm(module):
    return float(torch.norm(torch.stack([p.grad.norm() for p in module.parameters() if p.grad is not None])))

def overfit(config, dataset, seed=42, device="cpu", epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
            n_patients=N_PATIENTS, sampler=None, log=print):
    """Train Ext 1 on ``n_patients`` train patients' block split and score its targets; returns the record."""
    config = copy.deepcopy(config)
    config["model"]["use_mask_embedding"] = True
    idx = dataset["train_idx"][:n_patients]
    values, mask = dataset["values"][idx], dataset["mask"][idx]
    cond, target = BlockEvalMasker.from_config()(mask, dataset["record_ids"][idx])
    batch = to_batch(*(torch.from_numpy(a) for a in (values, mask, cond)))

    use_deterministic_kernels()
    streams = RunStreams(seed)
    streams.seed_model_init()
    model = build_model(config, device)
    train_config = config["train"]
    optimizer = Adam(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])
    scheduler = MultiStepLR(optimizer, milestones=[max(1, int(f * epochs)) for f in train_config["lr_milestones"]],
                            gamma=train_config["lr_gamma"])

    started, losses = time.time(), []
    for epoch in range(epochs):
        streams.start_epoch(epoch)
        model.train()
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            loss = model(batch, generator=streams.diffusion)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            log(f"epoch {epoch + 1}/{epochs}  loss {np.mean(losses[-steps_per_epoch:]):.4f}  [{time.time() - started:.0f}s]")
    encoder_grad_norm = grad_norm(model.mask_encoder)       # the last step's, still on the weights

    sampler = sampler or EvalSampler.from_config()
    guess = median_imputation(sampler(model, values, mask, cond))
    mae = masked_mae(guess, values, target)
    return {
        "check": "M4-2 overfit sanity check, Extension 1 (MissingnessEncoder)",
        "status": "PASS" if mae < MAE_BAR else "FAIL",
        "mae": round(mae, 6),
        "rmse": round(masked_rmse(guess, values, target), 6),
        "mae_below": MAE_BAR,
        "zero_guess_mae": round(float(np.abs(values[target == 1]).mean()), 6),     # for scale
        "patients": [int(r) for r in dataset["record_ids"][idx]],
        "n_targets": int(target.sum()),
        "holdout": "BlockEvalMasker.from_config(), trained and scored on the same split",
        "sampler": {"seed": sampler.seed, "n_samples": sampler.n_samples, "batch_size": sampler.batch_size},
        "seed": seed,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "departure": "steps_per_epoch steps on the one batch per epoch; the training loop takes 1",
        "loss_first_epoch": round(float(np.mean(losses[:steps_per_epoch])), 6),
        "loss_last_epoch": round(float(np.mean(losses[-steps_per_epoch:])), 6),
        "encoder_grad_norm_last_step": encoder_grad_norm,
        "device": device,
        "seconds": round(time.time() - started, 1),
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="Extension 1's overfit sanity check (M4-2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--out", default=OVERFIT_PATH)
    args = parser.parse_args(argv)

    config = yaml.safe_load(open(args.config))
    record = overfit(config, load_dataset(args.dataset), args.seed, pick_device(args.device),
                     steps_per_epoch=args.steps_per_epoch)
    with open(args.out, "w") as f:
        f.write(json.dumps(record, indent=1) + "\n")
    print(f"block MAE {record['mae']:.4f} on {record['n_targets']} targets (bar < {MAE_BAR}, "
          f"zero guess {record['zero_guess_mae']:.4f}); encoder grad norm "
          f"{record['encoder_grad_norm_last_step']:.3g}: {record['status']}  -> {args.out}")

if __name__ == "__main__":
    main()
