import random
from types import SimpleNamespace

import numpy as np
import torch

from utils import restore_rng, rng_state, set_seed


def test_rng_state_survives_safe_checkpoint_load(tmp_path):
    set_seed(7)
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(7))
    path = tmp_path / "rng.pt"
    torch.save(rng_state(loader), path)
    expected = (random.random(), np.random.random(), torch.rand(1),
                torch.rand(1, generator=loader.generator))

    random.random()
    np.random.random()
    torch.rand(1)
    torch.rand(1, generator=loader.generator)
    restore_rng(torch.load(path, weights_only=True), loader)
    actual = (random.random(), np.random.random(), torch.rand(1),
              torch.rand(1, generator=loader.generator))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])
