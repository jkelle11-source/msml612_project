import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_
from diffusion.streams import RunStreams
from diffusion.data import train_loader, to_batch
from models.build import build_model

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

def train(config, dataset, run_seed, device, on_epoch_end = None, on_step_end = None):

    use_deterministic_kernels()

    streams = RunStreams(run_seed)
    streams.seed_model_init()
    
    model = build_model(config, device)
    
    train_config = config["train"]
    
    optimizer = Adam(model.parameters(), 
                     lr = train_config["lr"],
                     weight_decay = train_config["weight_decay"])
    
    scheduler = MultiStepLR(optimizer = optimizer, 
                            milestones = [
                                int(f * train_config["epochs"]) 
                                for f in train_config["lr_milestones"]
                                ],gamma = train_config["lr_gamma"])
    
    loader = train_loader(dataset = dataset,
                          batch_size = train_config["batch_size"],
                          generator = streams.data_order)
    
    max_norm = float("inf") if train_config["clip_grad_norm"] is None else train_config["clip_grad_norm"]
    
    train_history = []
    
    for epoch in range(train_config["epochs"]):
        
        streams.start_epoch(epoch)
        
        model.train()
        
        for values, mask in loader:
        
            cond_mask, _ = streams.train_masker(mask)
            
            optimizer.zero_grad()
            
            loss = model(to_batch(values, mask, cond_mask), 
                         generator = streams.diffusion)
            
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
    
    return model, train_history
        
        
            
            
        
        
    
    