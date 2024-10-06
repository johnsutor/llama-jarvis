# !/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest.mock import Mock, patch

import pytest
import torch

from llama_jarvis import (
    JarvisConfig,
    JarvisModel,
)


@pytest.fixture
def jarvis_config():
    return JarvisConfig(
        language_model="gpt2",
        seamless_model="facebook/seamless-m4t-medium",
        s2e_mapper_downsample=5,
        s2e_mapper_hidden_dim=1024,
        e2s_upsample=25,
        e2s_num_layers=2,
        e2s_linear_dim=11008,
        e2s_num_heads=32,
    )


@pytest.fixture
def jarvis_model(jarvis_config):
    with patch("transformers.AutoModelForCausalLM.from_pretrained"), patch(
        "llama_jarvis.JarvisSeamlessM4TModel.from_pretrained"
    ):
        return JarvisModel(jarvis_config)


def test_jarvis_model_initialization(jarvis_model):
    assert isinstance(jarvis_model, JarvisModel)
    assert hasattr(jarvis_model, "language_model")
    assert hasattr(jarvis_model, "seamless_model")
    assert hasattr(jarvis_model, "s2e_mapper")
    assert hasattr(jarvis_model, "e2s_mapper")
    assert hasattr(jarvis_model, "e2s_head")


def test_downsample_attention_mask(jarvis_model):
    attention_mask = torch.ones(2, 100)
    downsampled_mask = jarvis_model.downsample_attention_mask(attention_mask, 5)

    assert downsampled_mask.shape == (2, 20)
    assert torch.all(downsampled_mask == 1)


def test_jarvis_model_from_pretrained():
    with patch(
        "llama_jarvis.JarvisConfig.from_pretrained"
    ) as mock_config_from_pretrained, patch(
        "llama_jarvis.JarvisModel.__init__", return_value=None
    ) as mock_init:
        mock_config = Mock()
        mock_config_from_pretrained.return_value = mock_config

        JarvisModel.from_pretrained("dummy/path")

        mock_config_from_pretrained.assert_called_once_with("dummy/path")
        mock_init.assert_called_once_with(mock_config)
