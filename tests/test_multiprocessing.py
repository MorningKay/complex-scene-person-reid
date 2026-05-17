import torch.multiprocessing as mp

from reid.utils import configure_torch_multiprocessing_sharing


def test_configure_torch_multiprocessing_sharing_prefers_file_system() -> None:
    original = mp.get_sharing_strategy()
    try:
        strategy = configure_torch_multiprocessing_sharing()
        available = mp.get_all_sharing_strategies()
        if "file_system" in available:
            assert strategy == "file_system"
        else:
            assert strategy == original
    finally:
        mp.set_sharing_strategy(original)


def test_configure_torch_multiprocessing_sharing_keeps_current_for_unknown_strategy() -> None:
    original = mp.get_sharing_strategy()
    try:
        strategy = configure_torch_multiprocessing_sharing("not_a_strategy")
        assert strategy == original
    finally:
        mp.set_sharing_strategy(original)
