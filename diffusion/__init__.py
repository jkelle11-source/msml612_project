"""Diffusion process: noising schedule, forward ``q_sample``, and reverse sampler.

Implements the conditional score-based diffusion machinery CSDI is built on
(PROJECT_PLAN §5, Week 2): a cosine noising schedule, ``q_sample`` for the forward
process, the masked denoising objective, and the DDPM reverse sampler used to draw
imputation posteriors at inference.
"""
