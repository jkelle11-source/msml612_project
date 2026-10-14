import numpy as np
import torch

from eval.masking import TrainMasker

STREAMS = ("model_init", "data_order", "train_mask", "diffusion", "global_rng")

def derive_seed(run_seed, stream, epoch=0):
    """The seed of one stream in one epoch: its own for every ``(run_seed, stream, epoch)``.

    Seeding every stream with the bare run seed would hand them all the same underlying
    sequence, so the shuffle, the masks and the noise would be copies of each other.
    """
    entropy = [run_seed, STREAMS.index(stream), epoch]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])

class RunStreams:
    """The random streams of one training run, paired across variants (PROJECT_PLAN §3.3).

    Upstream draws everything from the global RNG, where a single extra draw (an extension's
    layer init, one more dropout layer) shifts every draw after it and unpairs the run. Here
    each source of training randomness has a generator of its own:

    - ``data_order``: shuffling. Pass it to the DataLoader.
    - ``train_masker``: the conditioning/target split of each batch (``eval.TrainMasker``).
    - ``diffusion``: timestep and noise. Pass it as ``model(batch, generator=...)``.

    ``seed_model_init()`` goes right before the model is built, and ``start_epoch(epoch)`` at
    the top of every epoch, after the model is built. Both seed the global RNG; ``start_epoch``
    also re-seeds the three generators in place, so objects built once (a DataLoader holding
    ``data_order``) keep working. Every stream is then a function of ``(run_seed, epoch)``
    alone, which is what lets a run resumed at an epoch boundary continue exactly without any
    RNG state in its checkpoint.

    Dropout is the one stream this cannot pair: it draws from the global RNG, and a variant
    with extra dropout layers consumes more of it. The per-epoch seed still starts every
    variant's epoch from the same state.
    """

    def __init__(self, run_seed):
        self.run_seed = run_seed
        self.data_order = torch.Generator().manual_seed(derive_seed(run_seed, "data_order"))
        self.train_masker = TrainMasker(derive_seed(run_seed, "train_mask"))
        self.diffusion = torch.Generator().manual_seed(derive_seed(run_seed, "diffusion"))

    def seed_model_init(self):
        """Seed the global RNG for building the model. Build the parts every variant shares
        first, so their initial weights match across variants."""
        torch.manual_seed(derive_seed(self.run_seed, "model_init"))

    def start_epoch(self, epoch):
        """Put every stream at the start of ``epoch`` (0-based)."""
        self.data_order.manual_seed(derive_seed(self.run_seed, "data_order", epoch))
        self.train_masker.generator.manual_seed(derive_seed(self.run_seed, "train_mask", epoch))
        self.diffusion.manual_seed(derive_seed(self.run_seed, "diffusion", epoch))
        torch.manual_seed(derive_seed(self.run_seed, "global_rng", epoch))
