# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import (
    AutoModelForCausalLM,
    SeamlessM4TModel,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional


class EmbeddingMapper(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        downsample: int = 5,
    ):
        super().__init__()
        self.downsample = downsample
        self.net = nn.Sequential(
            nn.Linear(input_dim * downsample, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.downsample
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.downsample, dim * self.downsample)
        return self.net(x)


class JarvisConfig(PretrainedConfig):
    model_type = "jarvis"

    def __init__(
        self,
        language_model: str,
        seamless_model: str,
        speech_mapper_downsample: int,
        speech_mapper_hidden_dim: int,
        **kwargs,
    ):
        self.language_model = language_model
        self.seamless_model = seamless_model
        self.speech_mapper_downsample = speech_mapper_downsample
        self.speech_mapper_hidden_dim = speech_mapper_hidden_dim
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, config_dict):
        return cls(**config_dict)


class JarvisModel(PreTrainedModel):
    config_class = JarvisConfig

    def __init__(self, config: JarvisConfig):
        super().__init__(config)
        self.language_model = AutoModelForCausalLM.from_pretrained(config.language_model)
        self.seamless_model = SeamlessM4TModel.from_pretrained(config.seamless_model)
        self.speech_mapper = EmbeddingMapper(
            self.seamless_model.config.hidden_size,
            self.language_model.config.hidden_size,
            hidden_dim=config.speech_mapper_hidden_dim,
            downsample=config.speech_mapper_downsample,
        )

    def resize_token_embeddings(self, new_num_tokens):
        self.language_model.resize_token_embeddings(new_num_tokens)

    def forward(
        self,
        input_ids: torch.Tensor,
        speech_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speech_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        speech_embeddings = self.seamless_model.text_encoder(
            input_ids=speech_input_ids,
            attention_mask=speech_attention_mask
        ).last_hidden_state
        
        mapped_embeddings = self.speech_mapper(speech_embeddings)
        
        input_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        combined_embeds = torch.cat((input_embeddings, mapped_embeddings), dim=1)
        
        if attention_mask is not None and speech_attention_mask is not None:
            combined_attention_mask = torch.cat((attention_mask, speech_attention_mask), dim=1)
        else:
            combined_attention_mask = None
        
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
        )
        
        if labels is not None:
            shifted_logits = outputs.logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            loss = F.cross_entropy(
                shifted_logits.transpose(1, 2),
                shifted_labels,
                ignore_index=-100, 
            )
            outputs.loss = loss
        
        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = JarvisConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(config, **kwargs)