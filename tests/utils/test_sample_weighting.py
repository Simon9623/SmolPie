#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the sample weighting infrastructure."""

from unittest.mock import Mock

import pytest
import torch

from lerobot.utils.sample_weighting import (
    SampleWeighter,
    SampleWeightingConfig,
    UniformWeighter,
    make_sample_weighter,
)

# =============================================================================
# SampleWeightingConfig Tests
# =============================================================================


def test_config_default_values():
    """Test default configuration values."""
    config = SampleWeightingConfig()
    assert config.type == "uniform"
    assert config.extra_params == {}


def test_config_custom_values():
    """Test configuration with custom values."""
    config = SampleWeightingConfig(
        type="uniform",
        extra_params={"fallback_weight": 0.5},
    )
    assert config.type == "uniform"
    assert config.extra_params == {"fallback_weight": 0.5}


def test_config_uniform_type():
    """Test configuration for uniform weighting."""
    config = SampleWeightingConfig(type="uniform")
    assert config.type == "uniform"


# =============================================================================
# UniformWeighter Tests
# =============================================================================


def test_uniform_weighter_inherits_from_sample_weighter():
    """Test that UniformWeighter is a SampleWeighter."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    assert isinstance(weighter, SampleWeighter)


def test_uniform_weighter_compute_batch_weights_with_action_key():
    """Test weight computation with 'action' key in batch."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    batch = {"action": torch.randn(8, 10)}

    weights, stats = weighter.compute_batch_weights(batch)

    assert weights.shape == (8,)
    assert torch.allclose(weights, torch.ones(8))
    assert stats["mean_weight"] == 1.0
    assert stats["type"] == "uniform"


def test_uniform_weighter_compute_batch_weights_with_index_key():
    """Test weight computation with 'index' key in batch."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    batch = {"index": torch.arange(16)}

    weights, stats = weighter.compute_batch_weights(batch)

    assert weights.shape == (16,)
    assert torch.allclose(weights, torch.ones(16))


def test_uniform_weighter_compute_batch_weights_no_tensor_keys():
    """Test weight computation with no tensor keys (fallback to size 1)."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    batch = {"other_key": "some_value"}

    weights, stats = weighter.compute_batch_weights(batch)

    assert weights.shape == (1,)
    assert torch.allclose(weights, torch.ones(1))


def test_uniform_weighter_compute_batch_weights_empty_batch_raises():
    """Test that empty batch raises ValueError."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    batch = {}

    with pytest.raises(ValueError, match="empty batch"):
        weighter.compute_batch_weights(batch)


def test_uniform_weighter_compute_batch_weights_scans_all_keys():
    """Test that batch size is determined by scanning all tensor values."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    # Batch with non-standard key containing a tensor
    batch = {"custom_tensor": torch.randn(7, 3)}

    weights, stats = weighter.compute_batch_weights(batch)

    assert weights.shape == (7,)
    assert torch.allclose(weights, torch.ones(7))


def test_uniform_weighter_compute_batch_weights_on_cuda():
    """Test that weights are placed on the correct device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    weighter = UniformWeighter(device=torch.device("cuda"))
    batch = {"action": torch.randn(4, 10)}

    weights, _ = weighter.compute_batch_weights(batch)

    assert weights.device.type == "cuda"


def test_uniform_weighter_get_stats():
    """Test get_stats returns expected structure."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    stats = weighter.get_stats()

    assert stats == {"type": "uniform"}


# =============================================================================
# make_sample_weighter Factory Tests
# =============================================================================


def test_factory_returns_none_for_none_config():
    """Test that None config returns None weighter."""
    policy = Mock()
    device = torch.device("cpu")

    result = make_sample_weighter(None, policy, device)

    assert result is None


def test_factory_creates_uniform_weighter():
    """Test creation of UniformWeighter."""
    config = SampleWeightingConfig(type="uniform")
    policy = Mock()
    device = torch.device("cpu")

    weighter = make_sample_weighter(config, policy, device)

    assert isinstance(weighter, UniformWeighter)
    assert isinstance(weighter, SampleWeighter)


def test_factory_raises_for_unknown_type():
    """Test that unknown type raises ValueError."""
    config = SampleWeightingConfig(type="unknown_type")
    policy = Mock()
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="Unknown sample weighting type"):
        make_sample_weighter(config, policy, device)

