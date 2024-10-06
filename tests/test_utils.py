# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from transformers import AutoTokenizer

from llama_jarvis import (
    JarvisSeamlessM4TGenerationOutput,
    _compute_new_attention_mask,
    format_speech_generation_kwargs,
    prepare_tokenizer,
)


def test_compute_new_attention_mask():
    hidden_states = torch.randn(2, 5, 10)  # batch_size=2, seq_len=5
    seq_lens = torch.tensor([3, 4])

    mask = _compute_new_attention_mask(hidden_states, seq_lens)

    assert mask.shape == (2, 5)
    assert torch.all(mask[0, :3] == 1) and torch.all(mask[0, 3:] == 0)
    assert torch.all(mask[1, :4] == 1) and torch.all(mask[1, 4:] == 0)


def test_format_speech_generation_kwargs():
    kwargs = {
        "max_length": 100,
        "text_num_beams": 4,
        "speech_do_sample": True,
        "temperature": 0.7,
    }

    kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)

    assert kwargs_text == {"max_length": 100, "num_beams": 4, "temperature": 0.7}
    assert kwargs_speech == {"max_length": 100, "do_sample": True, "temperature": 0.7}


def test_prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    original_vocab_size = len(tokenizer)

    modified_tokenizer = prepare_tokenizer(tokenizer)

    assert (
        len(modified_tokenizer) == original_vocab_size + 3
    )  # 2 speech tokens + 1 pad token
    assert "<|BEGIN_SPEECH|>" in modified_tokenizer.get_vocab()
    assert "<|END_SPEECH|>" in modified_tokenizer.get_vocab()
    assert modified_tokenizer.pad_token == "<|PAD|>"


def test_jarvis_seamless_m4t_generation_output():
    waveform = torch.randn(2, 1000)
    waveform_lengths = torch.tensor([800, 1000])
    sequences = torch.randint(0, 1000, (2, 10))
    unit_sequences = torch.randint(0, 1000, (2, 20))
    unit_ids = torch.randint(0, 1000, (2, 20))

    output = JarvisSeamlessM4TGenerationOutput(
        waveform=waveform,
        waveform_lengths=waveform_lengths,
        sequences=sequences,
        unit_sequences=unit_sequences,
        unit_ids=unit_ids,
    )

    assert torch.all(output.waveform == waveform)
    assert torch.all(output.waveform_lengths == waveform_lengths)
    assert torch.all(output.sequences == sequences)
    assert torch.all(output.unit_sequences == unit_sequences)
    assert torch.all(output.unit_ids == unit_ids)


@pytest.mark.parametrize(
    "hidden_states, seq_lens",
    [
        (torch.randn(1, 10, 5), torch.tensor([5])),  # Single sequence
        (
            torch.randn(3, 8, 5),
            torch.tensor([3, 5, 8]),
        ),  # Multiple sequences of different lengths
        (torch.randn(2, 1, 5), torch.tensor([1, 1])),  # Very short sequences
    ],
)
def test_compute_new_attention_mask_various_inputs(hidden_states, seq_lens):
    mask = _compute_new_attention_mask(hidden_states, seq_lens)

    assert mask.shape == hidden_states.shape[:2]
    for i, length in enumerate(seq_lens):
        assert torch.all(mask[i, :length] == 1)
        assert torch.all(mask[i, length:] == 0)


def test_format_speech_generation_kwargs_priority():
    kwargs = {
        "max_length": 100,
        "text_max_length": 150,
        "speech_max_length": 200,
        "num_beams": 4,
        "text_temperature": 0.8,
        "speech_temperature": 0.9,
    }

    kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)

    assert kwargs_text["max_length"] == 150
    assert kwargs_speech["max_length"] == 200
    assert kwargs_text["num_beams"] == 4
    assert kwargs_speech["num_beams"] == 4
    assert kwargs_text["temperature"] == 0.8
    assert kwargs_speech["temperature"] == 0.9


def test_prepare_tokenizer_idempotent():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    modified_tokenizer = prepare_tokenizer(tokenizer)
    vocab_size_after_first_modification = len(modified_tokenizer)

    # Modify the tokenizer again
    modified_tokenizer = prepare_tokenizer(modified_tokenizer)

    assert len(modified_tokenizer) == vocab_size_after_first_modification


def test_jarvis_seamless_m4t_generation_output_partial():
    waveform = torch.randn(2, 1000)
    sequences = torch.randint(0, 1000, (2, 10))

    output = JarvisSeamlessM4TGenerationOutput(
        waveform=waveform,
        sequences=sequences,
    )

    assert torch.all(output.waveform == waveform)
    assert torch.all(output.sequences == sequences)
    assert output.waveform_lengths is None
    assert output.unit_sequences is None
    assert output.unit_ids is None
