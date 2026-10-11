"""M0 environment sanity checks."""

import sys

import numpy as np

from kuairand_rec import paths
from kuairand_rec.utils import set_seed


def test_python_version():
    assert sys.version_info[:2] == (3, 12)


def test_core_imports():
    import implicit  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    import torch

    assert torch.tensor([1.0, 2.0]).sum().item() == 3.0


def test_implicit_als_fits_tiny_matrix():
    import scipy.sparse as sp
    from implicit.cpu.als import AlternatingLeastSquares

    m = sp.random(20, 15, density=0.3, format="csr", random_state=0, dtype=np.float32)
    model = AlternatingLeastSquares(factors=4, iterations=2, random_state=0)
    model.fit(m, show_progress=False)
    assert model.user_factors.shape == (20, 4) and model.item_factors.shape == (15, 4)


def test_seed_reproducible():
    import torch

    set_seed(123)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(123)
    b = (np.random.rand(3), torch.rand(3))
    assert np.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_forbidden_files_not_in_pipeline_inputs():
    allowed = set(paths.STANDARD_LOGS) | {paths.USER_FEATURES, paths.VIDEO_FEATURES_BASIC}
    assert not allowed & set(paths.FORBIDDEN_FEATURE_FILES)
    assert not allowed & set(paths.EXCLUDED_LOGS)
