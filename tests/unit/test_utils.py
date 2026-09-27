"""Unit tests for utility modules: seed, git, logging."""

import random

import numpy as np
import pytest

from taorg.utils.seed import set_seed
from taorg.utils.git import get_git_commit


class TestSetSeed:
    """Tests for set_seed."""

    def test_deterministic_python(self):
        set_seed(42)
        a = [random.random() for _ in range(5)]
        set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_deterministic_numpy(self):
        set_seed(123)
        a = np.random.rand(5).tolist()
        set_seed(123)
        b = np.random.rand(5).tolist()
        assert a == b

    def test_negative_seed_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            set_seed(-1)

    def test_different_seeds_differ(self):
        set_seed(1)
        a = random.random()
        set_seed(2)
        b = random.random()
        assert a != b


class TestGetGitCommit:
    """Tests for get_git_commit."""

    def test_returns_string(self):
        # May return a commit hash or "unknown" depending on environment
        result = get_git_commit()
        assert isinstance(result, str)
        assert len(result) > 0
