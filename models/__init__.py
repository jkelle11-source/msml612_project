"""Denoiser and model variants (M1-2, owner: Josh).

Home for the *reused* CSDI denoiser (forked and cited, PROJECT_PLAN §0 decision 1) and
the two architectural extensions built on top of it:

- Extension 1 — ``MissingnessEncoder`` (dual-axis embedding of the mask, §4.1)
- Extension 2 — ``ContinuousTimeEncoding`` (true Time2Vec of per-feature TSLO, §4.2)

The denoiser forward pass follows the input-hook contract documented in the project
README: it accepts ``mask_emb`` and ``time_enc``, both defaulting to ``None``, and adds
them to the denoiser input (not to the attention logits).
"""
