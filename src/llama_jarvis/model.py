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


from typing import Literal, Optional


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
            nn.ReLU(inplace=True),
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

    def __init__(self, config: JarvisConfig, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model
        )
        self.seamless_model = SeamlessM4TModel.from_pretrained(config.seamless_model)
        self.speech_mapper = EmbeddingMapper(
            self.seamless_model.config.hidden_size,
            self.language_model.config.hidden_size,
            hidden_dim=config.speech_mapper_hidden_dim,
            downsample=config.speech_mapper_downsample,
        )
        tokenizer.add_tokens(
            ["<|BEGIN_SPEECH|>", "<|END_SPEECH|>"], special_tokens=True
        )
        self.language_model.resize_token_embeddings(len(tokenizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeddings: torch.Tensor,
        speech_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        phase: Literal["one", "two"] = "one",
    ):
        if input_ids is None and input_embeddings is None:
            raise ValueError("Either input_ids or input_embeddings must be provided")

        elif input_ids is not None and input_embeddings is not None:
            raise ValueError(
                "Only one of input_ids or input_embeddings can be provided"
            )

        elif input_ids is not None:
            input_embeddings = self.language_model.get_input_embeddings()(input_ids)

        mapped_embeddings = self.speech_mapper(speech_embeddings)

        if labels is not None:
            label_embeddings = self.language_model.get_input_embeddings()(labels)
            combined_embeds = torch.cat(
                (input_embeddings, mapped_embeddings, label_embeddings), dim=1
            )

            outputs = self.language_model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
            )

            shifted_logits = outputs.logits[:, :-1, :]
            shifted_labels = batch["labels"][:, 1:]

            loss = F.cross_entropy(
                shifted_logits[:, -shifted_labels.size(1) :, :].transpose(1, 2),
                shifted_labels,
            )

            return {**outputs, "loss": loss}

        combined_embeds = torch.cat((input_embeddings, mapped_embeddings), dim=1)

        return self.language_model(
            inputs_embeds=combined_embeds,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = JarvisConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(config, **kwargs)
