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
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Torch may not be available or deterministic not supported in env
        pass
    return seed

