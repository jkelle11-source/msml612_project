"""Reused CSDI implementation, vendored from the official repository (M1-2).

Upstream: https://github.com/ermongroup/CSDI
Commit:   7f24a436f08d98853a6b43d4f7f04e5a65ecdf27 (2024-03-15)
Paper:    Tashiro, Song, Song, Ermon (2021), "CSDI: Conditional Score-based Diffusion
          Models for Probabilistic Time Series Imputation", NeurIPS. arXiv:2107.03502
License:  MIT, Copyright (c) 2021 Yusuke Tashiro (see ``LICENSE`` in this directory)

``diff_models.py`` and ``main_model.py`` are copies of the upstream files at that commit.
Every line we changed is marked with an ``ADAPTED`` comment naming the issue that changed
it (``M1-2``, ``M2-1``); the project README lists what is reused and what is adapted.
"""

from .diff_models import diff_CSDI
from .main_model import CSDI_base, CSDI_Physio

__all__ = ["diff_CSDI", "CSDI_base", "CSDI_Physio"]
