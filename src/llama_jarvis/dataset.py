# !/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizerBase

class JarvisDataset(Dataset):
    def __init__(self, dataset, tokenizer: PreTrainedTokenizerBase):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        text_input = self.tokenizer(
            example["system_prompt"] + "<|BEGIN_SPEECH|>",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        
        speech_input = self.tokenizer(
            example["question"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        
        labels = self.tokenizer(
            "<|END_SPEECH|>" + example["response"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,  # Adjust as needed
        )
        
        return {
            "input_ids": text_input.input_ids.squeeze(0),
            "attention_mask": text_input.attention_mask.squeeze(0),
            "speech_input_ids": speech_input.input_ids.squeeze(0),
            "speech_attention_mask": speech_input.attention_mask.squeeze(0),
            "labels": labels.input_ids.squeeze(0),
        }

def prepare_tokenizer(tokenizer: PreTrainedTokenizerBase):
    tokenizer.add_tokens(["<|BEGIN_SPEECH|>", "<|END_SPEECH|>"], special_tokens=True)
    return tokenizer