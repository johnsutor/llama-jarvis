# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Code within this file is derived from both the HuggingFace Transformers library and the Llama-Omni project.
#
# The original license for the Hugging Face Transformers library is included below.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
#
# The original license for the Llama-Omni project is included below.
# Copyright 2023 The Llama-Omni team. All rights reserved.
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


import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaModel,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    SeamlessM4TModel,
    SeamlessM4TProcessor,
)
from transformers.file_utils import ModelOutput

from .utils import (
    JarvisSeamlessM4TGenerationOutput,
    _compute_new_attention_mask,
    format_speech_generation_kwargs,
    prepare_tokenizer,
)

logger = logging.getLogger(__name__)


class JarvisSeamlessM4TModel(SeamlessM4TModel):
    """
    This subclass of `SeamlessM4TModel` adds a method to generate translated speech from text with utilities
    to get the target vocoder token ids.

    Args:
        config (`PretrainedConfig`):
            The configuration for the model.
    """

    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = "eng",
        spkr_id: Optional[int] = 0,
        generate_speech: Optional[bool] = False,
        generate_target_ids: Optional[bool] = False,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        JarvisSeamlessM4TGenerationOutput,
    ]:
        """
        Generates translated token ids and/or translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_ids=input_ids, num_beams=4, speech_do_sample=True)` will successively
        perform beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>


        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`, *optional*):
                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio. Note that if `generate_speech=True`, this parameter will be
                ignored.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            spkr_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            generate_speech (`bool`, *optional*, defaults to `True`):
                If `False`, will only returns the text tokens and won't generate speech.
            generate_target_ids (`bool`, *optional*, defaults to `False`):
                If `True`, will return the target token ids for the vocoder. Setting this to `True` will override the
                parameter `generate_speech`, but will only be applied if `return_intermediate_token_ids` is also true.
                This is useful for training the model with the `SeamlessM4TProcessor`.

            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                    except for `decoder_input_ids` which will only be passed through the text components.
                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                    text model and speech model respectively. It has the priority over the keywords without a prefix.

                    This means you can, for example, specify a generation strategy for one generation but not for the
                    other.

        Returns:
            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor], ModelOutput]`:
            - If `generate_speech` and `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].
            - If `generate_speech` and not `return_intermediate_token_ids`, returns a tuple composed of waveforms of
              shape `(batch_size, sequence_length)`and and `waveform_lengths` which gives the length of each sample.
            - If `generate_speech=False`, it will returns `ModelOutput`.
        """
        if (
            input_ids is None
            and input_features is None
            and kwargs.get("inputs_embeds") is None
        ):
            raise ValueError(
                "`input_ids`,`input_features` and `inputs_embeds` are all empty. Make sure at least one of them is not."
            )

        if generate_speech and tgt_lang is None:
            raise ValueError(
                "You must specify a `tgt_lang` to generate translated speech."
            )

        if tgt_lang is not None:
            # also accept __xxx__
            tgt_lang = tgt_lang.replace("__", "")
            for key in [
                "text_decoder_lang_to_code_id",
                "t2u_lang_code_to_id",
                "vocoder_lang_code_to_id",
            ]:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(
                        f"""This model generation config doesn't have a `{key}` key which maps the target language
                        to the right token id. Make sure to load the right generation config."""
                    )
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model.
                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports
                    more languages for text translation than for speech synthesis."""
                    )

        batch_size = (
            len(input_features)
            if input_features is not None
            else (
                len(input_ids)
                if input_ids is not None
                else len(kwargs.get("inputs_embeds"))
            )
        )

        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text["output_hidden_states"] = True
        kwargs_text["return_dict_in_generate"] = True
        kwargs_text["output_scores"] = True

        text_decoder_input_ids = kwargs_text.get("decoder_input_ids")
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        if tgt_lang is not None:
            # tgt_lang gets priority over decoder input ids
            text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(
                tgt_lang
            )
            text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(
                self.device
            )

        kwargs_text["decoder_input_ids"] = text_decoder_input_ids

        # first generation, make sure to call grandparent class
        if input_features is not None:
            self.set_modality("speech")
            if input_ids is not None:
                logger.warning(
                    "`input_features` and `input_ids` are both non empty. `input_features` will be used in priority "
                    "through the speech encoder. Make sure `input_features=None` if you want to use the text encoder."
                )
            text_generation_output = super(SeamlessM4TModel, self).generate(
                input_features=input_features, **kwargs_text
            )
        else:
            self.set_modality("text")
            text_generation_output = super(SeamlessM4TModel, self).generate(
                input_ids=input_ids, input_features=None, **kwargs_text
            )
        sequences = text_generation_output.sequences

        if not generate_target_ids and not generate_speech:
            return text_generation_output

        # prepare second generation
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get(
            "attention_mask", kwargs_text.get("attention_mask", None)
        )

        # get encoder last hidden states
        if self.current_modality == "speech":
            # get last_hidden_state from encoder - must do a pass through the speech encoder
            encoder_hidden_states = self.speech_encoder(
                input_features=input_features, attention_mask=attention_mask
            ).last_hidden_state

            # input modality = speech so new attention mask for the decoder
            if attention_mask is not None:
                sub_sampled_lengths = (
                    self._compute_sub_sample_lengths_from_attention_mask(
                        attention_mask
                    ).to(encoder_hidden_states.device)
                )
                attention_mask = _compute_new_attention_mask(
                    hidden_states=encoder_hidden_states, seq_lens=sub_sampled_lengths
                )
        else:
            encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]

        # take care of num_return_sequences
        # take most probable hidden states per batch of return_sequences
        # (batch_size*num_return_sequences, ...) -> (batch_size,...)
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = (
                text_generation_output.sequences_scores.view(batch_size, -1)
            )
            idx_most_probable_sequences_per_batch = (
                idx_most_probable_sequences_per_batch.argmax(-1)
            )
            idx_most_probable_sequences_per_batch = (
                idx_most_probable_sequences_per_batch
                + torch.arange(batch_size).to(self.device) * num_return_sequences
            )
            sequences = sequences[idx_most_probable_sequences_per_batch]

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(
            t2u_input_embeds, seq_lens
        )
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # Compute t2u decoder_input_ids
        t2u_decoder_input_ids = kwargs_speech.get("decoder_input_ids")
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = torch.tensor(
            [[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size
        ).to(self.device)
        kwargs_speech["decoder_input_ids"] = t2u_decoder_input_ids

        # second generation
        unit_ids = self.t2u_model.generate(
            inputs_embeds=t2u_input_embeds, **kwargs_speech
        )
        output_unit_ids = unit_ids.detach().clone()

        # get rid of t2u_decoder_input_ids
        unit_ids = unit_ids[:, kwargs_speech["decoder_input_ids"].shape[1] :]
        # replace eos per pad
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = (
            self.config.t2u_pad_token_id
        )
        # offset of control symbols
        unit_ids = torch.where(
            unit_ids == self.config.t2u_pad_token_id,
            unit_ids,
            unit_ids - self.config.vocoder_offset,
        )

        if generate_target_ids:
            return JarvisSeamlessM4TGenerationOutput(
                sequences=sequences,
                unit_sequences=output_unit_ids,
                unit_ids=unit_ids.detach().clone(),
            )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(
            tgt_lang
        )
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids)).to(
            self.device
        )

        spkr_id = torch.tensor([[spkr_id]] * len(unit_ids)).to(self.device)

        waveform, waveform_lengths = self.vocoder(
            input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id
        )

        if return_intermediate_token_ids:
            return JarvisSeamlessM4TGenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths


class SpeechToEmbeddingMapper(nn.Module):
    """
    A simple feedforward network that maps the speech embeddings to the LLM embeddings, while downsampling the sequence
    length by a factor of `downsample`.

    Args:
        input_dim (`int`):
            The dimensionality of the speech embeddings.
        output_dim (`int`):
            The dimensionality of the LLM embeddings.
        hidden_dim (`int`, *optional*, defaults to 1024):
            The dimensionality of the hidden layer in the feedforward network.
        downsample (`int`, *optional*, defaults to 5):
            The factor by which to downsample the sequence length.

    Shape:
        - Input: `(batch_size, seq_len, input_dim)`
        - Output: `(batch_size, seq_len // downsample, output_dim)`
    """

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
    """
    This class defines the configuration for the Jarvis model.

    Args:
        language_model (`str`):
            The name or path of the pretrained language model.
        seamless_model (`str`):
            The name or path of the pretrained SeamlessM4T model.
        s2e_mapper_downsample (`int`, *optional*, defaults to 5):
            The factor by which to downsample the sequence length in the speech to embedding mapper.
        s2e_mapper_hidden_dim (`int`, *optional*, defaults to 1024):
            The dimensionality of the hidden layer in the speech to embedding mapper.
        e2s_upsample (`int`, *optional*, defaults to 25):
            The factor by which to upsample the sequence length in the embedding to speech mapper.
        e2s_num_layers (`int`, *optional*, defaults to 2):
            The number of layers in the embedding to speech mapper.
        e2s_linear_dim (`int`, *optional*, defaults to 11008):
            The dimensionality of the intermediate layer in the embedding to speech mapper.
        e2s_num_heads (`int`, *optional*, defaults to 32):
            The number of attention heads in the embedding to speech mapper.
    """

    model_type = "jarvis"

    def __init__(
        self,
        language_model: str,
        seamless_model: str,
        s2e_mapper_downsample: int = 5,
        s2e_mapper_hidden_dim: int = 1024,
        e2s_upsample: int = 25,
        e2s_num_layers: int = 2,
        e2s_linear_dim: int = 11008,
        e2s_num_heads: int = 32,
        **kwargs,
    ):
        self.language_model = language_model
        self.seamless_model = seamless_model
        self.s2e_mapper_downsample = s2e_mapper_downsample
        self.s2e_mapper_hidden_dim = s2e_mapper_hidden_dim
        self.e2s_upsample = e2s_upsample
        self.e2s_num_layers = e2s_num_layers
        self.e2s_linear_dim = e2s_linear_dim
        self.e2s_num_heads = e2s_num_heads

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, config_dict):
        return cls(**config_dict)


class JarvisProcessor(ProcessorMixin):
    """
    This class processes the inputs for the Jarvis model.

    Args:
        tokenizer (`Union[PreTrainedTokenizerBase, str]`):
            The name or instance of the pretrained tokenizer.
        seamless_processor (`Union[SeamlessM4TProcessor, str]`):
            The name or instance of the pretrained SeamlessM4T processor.
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizerBase, str],
        seamless_processor: Union[SeamlessM4TProcessor, str],
    ):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        if isinstance(seamless_processor, str):
            seamless_processor = SeamlessM4TProcessor.from_pretrained(
                seamless_processor
            )

        self.seamless_processor = seamless_processor
        self.tokenizer = prepare_tokenizer(tokenizer)

    def __call__(
        self,
        instruction: Optional[List[str]] = None,
        text: Optional[List[str]] = None,
        speech: Optional[List[str]] = None,
        label: Optional[List[str]] = None,
        src_lang: str = "eng",
        **kwargs,
    ):
        return_dict = {}

        if instruction is not None:
            instruction_tokens = self.tokenizer(instruction, **kwargs)

            return_dict.update(
                {"instruction_" + k: v for k, v in instruction_tokens.items()}
            )

        if text is not None:
            text_tokens = self.seamless_processor(
                text=text, src_lang=src_lang, **kwargs
            )

            return_dict.update({"text_" + k: v for k, v in text_tokens.items()})

        if speech is not None:
            speech_tokens = self.seamless_processor(
                text=speech, src_lang=src_lang, **kwargs
            )

            return_dict.update({"speech_" + k: v for k, v in speech_tokens.items()})

        if label is not None:
            phase_one_label_tokens = self.tokenizer(label, **kwargs)
            phase_two_label_tokens = self.seamless_processor(
                text=label, src_lang=src_lang, **kwargs
            )

            return_dict.update(
                {"phase_one_label_" + k: v for k, v in phase_one_label_tokens.items()}
            )

            return_dict.update(
                {"phase_two_label_" + k: v for k, v in phase_two_label_tokens.items()}
            )

        # Make sure batch dimensions are consistent
        batch_size = None
        for k, v in return_dict.items():
            if batch_size is None:
                batch_size = v.size(0)
            else:
                assert v.size(0) == batch_size, f"Batch size mismatch in key: {k}"

        return return_dict


class JarvisModel(PreTrainedModel):
    """
    This class defines the Jarvis model, which combines a language model with a SeamlessM4T model.

    Args:
        config (`JarvisConfig`):
            The configuration for the model.
    """

    config_class = JarvisConfig

    def __init__(self, config: JarvisConfig):
        super().__init__(config)
        self.config = config
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model
        )
        self.seamless_model = JarvisSeamlessM4TModel.from_pretrained(
            config.seamless_model
        )
        self.s2e_mapper = SpeechToEmbeddingMapper(
            self.seamless_model.config.hidden_size,
            self.language_model.config.hidden_size,
            hidden_dim=config.s2e_mapper_hidden_dim,
            downsample=config.s2e_mapper_downsample,
        )
        self.e2s_mapper = LlamaModel(
            LlamaConfig(
                vocab_size=self.language_model.config.vocab_size,
                num_hidden_layers=config.e2s_num_layers,
                hidden_size=self.language_model.config.hidden_size,
                intermediate_size=config.e2s_linear_dim,
                num_attention_heads=config.e2s_num_heads,
            )
        )
        # Add an extra token for the CTC blank token
        self.e2s_head = nn.Linear(
            self.language_model.config.hidden_size,
            self.seamless_model.config.unit_hifi_gan_vocab_size + 1,
        )

    def resize_token_embeddings(self, new_num_tokens):
        """Resize the token embeddings of the language model. Necessary for adding the speech tokens."""
        self.language_model.resize_token_embeddings(new_num_tokens)

    def downsample_attention_mask(self, attention_mask, downsample):
        """Downsample an attention mask by a factor of `downsample`."""
        _, seq_len = attention_mask.size()

        num_frames_to_discard = seq_len % downsample

        if num_frames_to_discard > 0:
            attention_mask = attention_mask[:, :-num_frames_to_discard]

        return attention_mask[:, ::downsample]

    def forward(
        self,
        instruction_input_ids: Optional[torch.Tensor] = None,
        instruction_attention_mask: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        speech_input_ids: Optional[torch.Tensor] = None,
        speech_attention_mask: Optional[torch.Tensor] = None,
        phase_one_label_input_ids: Optional[torch.Tensor] = None,
        phase_one_label_attention_mask: Optional[torch.Tensor] = None,
        phase_two_label_input_ids: Optional[torch.Tensor] = None,
        phase_two_label_attention_mask: Optional[torch.Tensor] = None,
        train_phase: Optional[int] = 1,
        tgt_lang: Optional[str] = "eng",
        spkr_id: Optional[int] = 0,
    ):
        if (
            instruction_input_ids is None
            and text_input_ids is None
            and speech_input_ids is None
        ):
            raise ValueError(
                "At least one of instruction_input_ids, text_input_ids, or speech_input_ids must be provided."
            )

        combined_embeds = None
        combined_attention_mask = None

        if instruction_input_ids is not None:
            combined_embeds = self.language_model.get_input_embeddings()(
                instruction_input_ids
            )
            combined_attention_mask = instruction_attention_mask

        if text_input_ids is not None:
            text_embeddings = self.seamless_model.text_encoder(
                input_ids=text_input_ids, attention_mask=text_attention_mask
            ).last_hidden_state

            mapped_text = self.s2e_mapper(text_embeddings)
            text_attention_mask = self.downsample_attention_mask(
                text_attention_mask, self.config.s2e_mapper_downsample
            )

            if combined_embeds is not None:
                combined_embeds = torch.cat((combined_embeds, mapped_text), dim=1)
                combined_attention_mask = torch.cat(
                    (combined_attention_mask, text_attention_mask), dim=1
                )

            else:
                combined_embeds = mapped_text
                combined_attention_mask = text_attention_mask

        if speech_input_ids is not None:
            speech_embeddings = self.seamless_model.speech_encoder(
                input_ids=speech_input_ids, attention_mask=speech_attention_mask
            ).last_hidden_state

            mapped_speech = self.s2e_mapper(speech_embeddings)
            speech_attention_mask = self.downsample_attention_mask(
                speech_attention_mask, self.config.s2e_mapper_downsample
            )

            if combined_embeds is not None:
                combined_embeds = torch.cat((combined_embeds, mapped_speech), dim=1)
                combined_attention_mask = torch.cat(
                    (combined_attention_mask, speech_attention_mask), dim=1
                )

            else:
                combined_embeds = mapped_speech
                combined_attention_mask = speech_attention_mask

        if phase_one_label_input_ids is not None:
            label_embeds = self.language_model.get_input_embeddings()(
                phase_one_label_input_ids
            )
            combined_embeds = torch.cat((combined_embeds, label_embeds), dim=1)
            combined_attention_mask = torch.cat(
                (combined_attention_mask, phase_one_label_attention_mask), dim=1
            )

        lm_outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            output_hidden_states=True,
        )

        if train_phase == 1 and phase_one_label_input_ids is not None:
            shifted_logits = lm_outputs.logits[:, :-1, :].contiguous()
            shifted_labels = phase_one_label_input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits[:, -shifted_labels.shape[1] :].transpose(1, 2),
                shifted_labels,
                ignore_index=-100,
            )

            return ModelOutput(loss=loss, **lm_outputs)

        if train_phase == 2 and phase_two_label_input_ids is not None:
            # Get the last hidden state of the language model
            # and upsample it
            try:
                last_hidden_state = lm_outputs.hidden_states[-1]
            except AttributeError:
                last_hidden_state = lm_outputs.last_hidden_state
            except IndexError:
                last_hidden_state = lm_outputs.last_hidden_state

            last_hidden_state = torch.repeat_interleave(
                last_hidden_state, self.config.e2s_upsample, dim=1
            )

            # Pass the upsampled hidden state through the e2s mapper
            e2s_embeds = self.e2s_mapper(
                inputs_embeds=last_hidden_state
            ).last_hidden_state
            e2s_logits = self.e2s_head(e2s_embeds)

            unit_labels = self.seamless_model.generate(
                input_ids=phase_two_label_input_ids,
                text_attention_mask=phase_two_label_attention_mask,
                tgt_lang=tgt_lang,
                spkr_id=spkr_id,
                generate_speech=False,
                generate_target_ids=True,
            ).unit_ids

            # Logits are shape (batch_size, seq_len, vocab_size), need
            # to be (seq_len, batch_size, vocab_size) for CTC loss
            loss = F.ctc_loss(
                e2s_logits.permute(1, 0, 2).log_softmax(-1),
                unit_labels,
                input_lengths=[e2s_logits.size(1)] * e2s_logits.size(0),
                target_lengths=[unit_labels.size(1)] * unit_labels.size(0),
                blank=self.seamless_model.config.unit_hifi_gan_vocab_size,
                zero_infinity=True,
            )

            return ModelOutput(loss=loss, **lm_outputs)

        return lm_outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = JarvisConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(config, **kwargs)
