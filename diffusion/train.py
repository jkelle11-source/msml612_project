import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_
from diffusion.streams import RunStreams
from diffusion.data import train_loader, to_batch
from models.build import build_model

# The training regimes (PROJECT_PLAN §3.1, M6-2), by ``train.regime``: TrainMasker's strategy, and
# the holdout validation tracks (diffusion.score's names). Test scoring is the same in both.
REGIMES = {"random": {"strategy": "random", "holdout": "random"},
           "block-aware": {"strategy": "mix", "holdout": "block"}}

def regime_name(config):
    """``random`` or ``block-aware``, from ``train.regime``; a config saved before M6-2 has none: random."""
    regime = config["train"].get("regime", "random")
    if regime not in REGIMES:
        raise ValueError(f"train.regime must be one of {list(REGIMES)}, not {regime!r}")
    return regime

def use_deterministic_kernels():
    """Make reruns bit-identical on CUDA too (M3-1).

    Some CUDA kernels add floating-point numbers in whatever order their threads finish, so two
    identical runs drift apart. This switches PyTorch to deterministic kernels, and makes it raise
    on any op that has none. cuBLAS reads CUBLAS_WORKSPACE_CONFIG when CUDA first runs a matrix
    multiply, so this must come before any CUDA work in the process; a value already set is kept.
    The switch is process-wide and stays on, so validation and scoring passes after it are covered.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)

def use_tf32(on):
    """Let CUDA compute float32 matrix multiplies and cuDNN convolutions in TF32, or not (M6-2).

    ``protocol.tf32`` decides it, the same for every run (PROJECT_PLAN §3.3). Off, both are full
    float32 ("ieee"), overriding PyTorch's default of TF32 convolutions on an A100, L4 or H100. A T4,
    MPS and the CPU have no TF32, so there it changes nothing. Process-wide, like the kernels switch.
    """
    precision = "tf32" if on else "ieee"
    torch.backends.cuda.matmul.fp32_precision = precision
    torch.backends.cudnn.conv.fp32_precision = precision

def train(config, dataset, run_seed, device, on_epoch_end = None, on_step_end = None, on_batch = None,
          on_start = None, on_checkpoint = None, resume = None):
    """Train one model; returns ``(model, history)``, the history of the epochs trained in this call.

    ``on_checkpoint(state)`` is called at the end of every epoch, after ``on_epoch_end``, with
    ``{"epoch", "model", "optimizer", "scheduler"}`` (the epoch just finished and the three state
    dicts): all a run needs to resume at that epoch boundary (M6-2). ``resume`` is such a state: the
    model is still built from the run seed and handed to ``on_start`` as built, then given the
    state, and training goes on from the next epoch. Every stream is a function of the run seed and
    the epoch (``RunStreams``), so a resumed run ends exactly as an uninterrupted one.
    """

    use_deterministic_kernels()
    use_tf32(config["protocol"].get("tf32", False))     # a config from before M6-2 has none: float32

    streams = RunStreams(run_seed, REGIMES[regime_name(config)]["strategy"],
                         config["protocol"]["block_window_hours"])
    streams.seed_model_init()

    model = build_model(config, device)

    # Pairing checks (M6-2): the starting weights, as built from the run seed, once. A copy, so the
    # hook cannot change them; a run resumed from a checkpoint loads it after this, never before.
    if on_start is not None:
        on_start({key: value.detach().clone() for key, value in model.state_dict().items()})

    train_config = config["train"]
    
    optimizer = Adam(model.parameters(), 
                     lr = train_config["lr"],
                     weight_decay = train_config["weight_decay"])
    
    # A drop is never before epoch 1 (ours, M3-2 review): upstream's int(0.75 * epochs) is 0 for a
    # 1-epoch run, and a milestone of 0 drops the lr before the first step. From 2 epochs on, and
    # at 150 and 180 of 200, the milestones are upstream's.
    scheduler = MultiStepLR(optimizer = optimizer, 
                            milestones = [
                                max(1, int(f * train_config["epochs"]))
                                for f in train_config["lr_milestones"]
                                ],gamma = train_config["lr_gamma"])
    
    loader = train_loader(dataset = dataset,
                          batch_size = train_config["batch_size"],
                          generator = streams.data_order)
    
    max_norm = float("inf") if train_config["clip_grad_norm"] is None else train_config["clip_grad_norm"]

    first_epoch = 0
    if resume is not None:              # after on_start, so the pairing hash is of the weights as built
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        first_epoch = resume["epoch"] + 1

    train_history = []

    for epoch in range(first_epoch, train_config["epochs"]):
        
        streams.start_epoch(epoch)
        
        model.train()
        
        for values, mask in loader:
        
            cond_mask, _ = streams.train_masker(mask)

            if on_batch is not None:
                drawn = (values.clone(), mask.clone(), cond_mask.clone())

            optimizer.zero_grad()

            loss = model(to_batch(values, mask, cond_mask),
                         generator = streams.diffusion)

            # Pairing checks (M6-2): what this step drew from the data-order and train-mask streams,
            # as copies taken before the model saw them, and the diffusion generator's state once the
            # step's timesteps and noise are drawn. Called after the forward pass, so nothing the hook
            # does to what it is given can reach training.
            if on_batch is not None:
                on_batch(epoch, *drawn, streams.diffusion.get_state())

            loss.backward()
            
            grad_norm = clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            train_history.append({"epoch": epoch,
                                  "loss": loss.item(),
                                  "grad_norm": grad_norm.item(),
                                  "lr": scheduler.get_last_lr()[0]})

            # Live view of the history (M3-1): the run log and wandb (M3-2) read each record here.
            if on_step_end is not None:
                on_step_end(dict(train_history[-1]))

        scheduler.step()
        
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)

        if on_checkpoint is not None:
            on_checkpoint({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict()})

    return model, train_history
        
        
            
            
        
        
    
    