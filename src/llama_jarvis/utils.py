# !/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import (
    PreTrainedTokenizerBase,
)
from transformers.file_utils import ModelOutput


def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """
    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that
    stops at the corresponding element in `seq_lens`.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, *)`):
            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.
        seq_lens (`torch.Tensor` of shape `(batch)`:
            Each element represents the length of the sequence at the same index in `hidden_states`

    Returns:
        `torch.FloatTensor`: The float attention mask of shape `(batch, seq_len)`
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]

    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    mask = mask.masked_fill(bool_mask, 0)

    return mask


def format_speech_generation_kwargs(kwargs):
    """
    Format kwargs for SeamlessM4T models that generate speech, attribute kwargs to either the text generation or the
    speech generation models.

    Args:
        kwargs (`dict`)`:
             Keyword arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                except for `decoder_input_ids` which will only be passed through the text components.
                - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                text model and speech model respectively. It has the priority over the keywords without a prefix.

                This means you can, for example, specify a generation strategy for one generation but not for the
                other.
    """
    # attribute kwargs to models
    kwargs_text = {}
    kwargs_speech = {}
    for key, value in kwargs.items():
        if key.startswith("text_"):
            key = key[len("text_") :]
            kwargs_text[key] = value
        elif key.startswith("speech_"):
            key = key[len("speech_") :]
            kwargs_speech[key] = value
        else:
            # If the key is already in a specific config, then it's been set with a
            # submodules specific value and we don't override
            if key not in kwargs_text:
                kwargs_text[key] = value
            if key not in kwargs_speech:
                kwargs_speech[key] = value
    return kwargs_text, kwargs_speech


def prepare_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """
    Modifies the tokenizer to include special tokens for speech synthesis. Also adds
    a padding token if it is not already in the tokenizer.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer to modify.

    Returns:
        `PreTrainedTokenizerBase`: The modified tokenizer.

    """
    tokenizer.add_tokens(["<|BEGIN_SPEECH|>", "<|END_SPEECH|>"], special_tokens=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})

    return tokenizer


@dataclass
class JarvisSeamlessM4TGenerationOutput(ModelOutput):
    """
    Class defining the generated outputs from [`SeamlessM4TModel`], [`SeamlessM4TForTextToText`],
    [`SeamlessM4TForTextToSpeech`], [`SeamlessM4TForSpeechToSpeech`] and [`SeamlessM4TForTextToSpeech`].

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The final audio waveform predicted by the model.
        waveform_lengths (`torch.IntTensor` of shape `(batch_size,)`, *optional*):
            The length in samples of each element in the `waveform` batch.
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The generated translated sequences. This is the output of the text-to-text or the speech-to-text models.
            The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished
            early due to the `eos_token_id`.
        unit_sequences (`torch.LongTensor` of shape `(batch_size, unit_sequence_length)`, *optional*):
            The generated translated unit sequences. This is the output of the text-to-units model. The second
            dimension (unit_sequence_length) is either equal to `t2u_max_length` or shorter if all batches finished
            early due to the `t2u_eos_token_id`.
    """

    waveform: Optional[torch.FloatTensor] = None
    waveform_lengths: Optional[torch.IntTensor] = None
    sequences: Optional[Tuple[torch.FloatTensor]] = None
    unit_sequences: Optional[Tuple[torch.FloatTensor]] = None
    unit_ids: Optional[Tuple[torch.FloatTensor]] = None
