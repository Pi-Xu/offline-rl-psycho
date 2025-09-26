from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seeds(seed: Optional[int] = None) -> int:
    """Set global seeds for Python, NumPy, and (optionally) PyTorch.

    Returns the seed used.
    """
    if seed is None:
        seed = int(os.getenv("SEED", 42))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        # For CUDA, enabling deterministic algorithms without setting
        # CUBLAS_WORKSPACE_CONFIG can raise at runtime. Guard it.
        is_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        if is_cuda:
            # Prefer deterministic cuDNN kernels; avoid non-deterministic autotune
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass
            if os.getenv("CUBLAS_WORKSPACE_CONFIG"):
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
            # else: skip strict deterministic mode to avoid CuBLAS runtime error
        else:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        # Torch may not be available or deterministic not supported in env
        pass
    return seed
