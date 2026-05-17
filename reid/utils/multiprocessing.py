"""PyTorch multiprocessing runtime helpers."""

from __future__ import annotations

import torch.multiprocessing as mp


def configure_torch_multiprocessing_sharing(preferred: str = "file_system") -> str | None:
    """Prefer a sharing strategy that is stable for large DataLoader eval loops."""

    try:
        available = mp.get_all_sharing_strategies()
        if preferred in available and mp.get_sharing_strategy() != preferred:
            mp.set_sharing_strategy(preferred)
        return mp.get_sharing_strategy()
    except (AttributeError, RuntimeError, ValueError):
        return None
