# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from torch.utils.data import Dataset


class JarvisPhaseOneDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        seamless_processor,
        seamless_model,
        precompute_embeddings=False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seamless_processor = seamless_processor
        self.seamless_model = seamless_model
        self.precompute_embeddings = precompute_embeddings

        if self.precompute_embeddings:
            self.dataset = self.precompute_speech_embeddings()

    @torch.no_grad()
    def compute_speech_embedding(self, example):
        speech_input = self.seamless_processor(
            text=[example["question"]], return_tensors="pt", padding=True
        )

        speech_embeddings = self.seamless_model.text_encoder(
            **speech_input
        ).last_hidden_state.squeeze(0)
        example["speech_embeddings"] = speech_embeddings.cpu().numpy().tolist()

        return example

    def precompute_speech_embeddings(self):
        return self.dataset.map(self.compute_speech_embedding)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        text_input = self.tokenizer(
            example["system_prompt"] + "<|BEGIN_SPEECH|>",
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if self.precompute_embeddings:
            speech_embeddings = torch.tensor(example["speech_embeddings"])
        else:
            speech_input = self.seamless_processor(
                text=[example["question"]], return_tensors="pt", padding=True
            )
            with torch.no_grad():
                speech_embeddings = self.seamless_model.text_encoder(
                    **speech_input
                ).last_hidden_state.squeeze(0)

        labels = self.tokenizer(
            "<|END_SPEECH|>" + example["response"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return {
            "input_ids": text_input.input_ids.squeeze(0),
            "input_attention_mask": text_input.attention_mask.squeeze(0),
            "speech_embeddings": speech_embeddings,
            "labels": labels.input_ids.squeeze(0),
            "labels_attention_mask": labels.attention_mask.squeeze(0),
        }
