"""Training loop around the reused CSDI diffusion process.

The diffusion math is not reimplemented here. The noise schedule, forward noising,
masked denoising objective and reverse sampler are upstream's, vendored in
``models/csdi/main_model.py`` (``CSDI_base``) and kept as published so the reproduction
gate is measured against the configuration behind the paper's result (PROJECT_PLAN §0,
decision 1). This package holds what we build around it (PROJECT_PLAN §5, Week 2): the
training loop, which takes its conditioning masks from ``eval.TrainMasker`` in place of
upstream's internal masking.
"""
