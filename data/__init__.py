"""PhysioNet Set A data pipeline (M1-1, owner: Jake).

Parses PhysioNet/CinC Challenge 2012 Set A into a dense ``(N, 48, D)`` grid with a
binary observation mask, computes and stores the per-feature time-since-last-observation
(TSLO) ``delta`` that Extension 2 consumes, and z-scores per variable using
**train-split statistics only** (see PROJECT_PLAN §3.2, §4.2).

Raw downloads live in ``data/raw/`` and processed tensors/scalers in ``data/processed/``;
both are git-ignored (see ``.gitignore``) so artifacts are not committed.
"""
