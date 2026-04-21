from .kodak import KodakDataset
from .div2k import DIV2KDataset
from .stanford_3d import Stanford3DDataset


def get_dataset(name, data_root, **kwargs):
    _datasets = {
        'kodak':       KodakDataset,
        'div2k':       DIV2KDataset,
        'stanford_3d': Stanford3DDataset,
    }
    if name not in _datasets:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(_datasets.keys())}")
    return _datasets[name](data_root, **kwargs)
