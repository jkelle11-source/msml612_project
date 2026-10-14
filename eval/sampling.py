from pathlib import Path

import torch
import yaml

from .masking import CONFIG_PATH

class EvalSampler:
    """One evaluation pass of the reused sampler, with frozen noise (PROJECT_PLAN §3.3).

    Draws ``n_samples`` imputations for every patient. All of the sampler's noise comes from a
    private generator that is re-seeded with the frozen ``seed`` at the start of every pass, so
    a checkpoint's samples depend on its weights alone -- never on the run seed, the global
    RNG, or how many passes came before. Variants, seeds and epochs are all scored with the
    same noise, and a validation pass in the middle of training cannot shift the training
    streams. Patients go through in the order given, ``batch_size`` at a time; the noise a
    patient receives depends on its position in the pass, so the batch size is frozen too.
    """

    def __init__(self, seed, n_samples, batch_size):
        self.seed = seed
        self.n_samples = n_samples
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, path=CONFIG_PATH):
        protocol = yaml.safe_load(Path(path).read_text())["protocol"]
        return cls(protocol["eval_sampling_seed"], protocol["eval_n_samples"], protocol["eval_batch_size"])

    def __call__(self, model, values, mask, cond_mask):
        """Sample imputations for ``values`` (N, T, D); returns a ``(K, N, T, D)`` float32 array.

        ``cond_mask`` is all the model may see (``EvalMasker``'s) and goes in as upstream's
        ``gt_mask``. The model is put in eval mode for the pass and handed back as it was.
        """
        generator = torch.Generator().manual_seed(self.seed)
        was_training = model.training
        model.eval()
        try:
            batches = []
            for start in range(0, len(values), self.batch_size):
                rows = slice(start, start + self.batch_size)
                observed_data = torch.as_tensor(values[rows])
                batch = {
                    "observed_data": observed_data,
                    "observed_mask": torch.as_tensor(mask[rows]),
                    "gt_mask": torch.as_tensor(cond_mask[rows]),
                    "timepoints": torch.arange(observed_data.shape[1]).float().expand(len(observed_data), -1),
                }
                samples = model.evaluate(batch, self.n_samples, generator=generator)[0]     # (B, K, D, T)
                batches.append(samples.permute(1, 0, 3, 2).cpu())                           # (K, B, T, D)
        finally:
            model.train(was_training)
        return torch.cat(batches, dim=1).numpy()
