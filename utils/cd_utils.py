# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from types import MethodType
import os
import numpy as np

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from transformers import __version__
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import (
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    GenerationConfig,
    CACHE_CONFIG_MAPPING,
    BaseWatermarkingConfig,
    WatermarkingConfig,
    CompileConfig,
)

from transformers.generation.streamers import BaseStreamer
from transformers.generation.beam_constraints import (
    DisjunctiveConstraint,
    PhrasalConstraint,
)
from transformers.generation.beam_search import (
    BeamSearchScorer,
    ConstrainedBeamSearchScorer,
)
from transformers.generation.utils import (
    _relative_top_filter,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    GenerateOutput,
    GenerationMixin,
)
from transformers.utils import (
    is_torchdynamo_compiling,
    ExplicitEnum,
    PushToHubMixin,
    logging,
)

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    CONTRASTIVE_GENERATION = "contrastive_generation"
    ATTENTION_CONTRASTIVE_DECODING = "attention_contrastive_decoding"
    ACTIVATION_DECODING = "activation_decoding"
    ENTROPY_ENHANCED_DECODING = "entropy_enhanced_decoding"
    DOLA_GENERATION = "dola_generation"
    # Beam methods
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"


from dataclasses import dataclass
from transformers.utils import (
    ModelOutput,
)


@dataclass
class DolaGenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    dola_logits: Optional[List[List[torch.FloatTensor]]] = None


class GenerationConfigCustom(PushToHubMixin):
    extra_output_flags = (
        "output_attentions",
        "output_hidden_states",
        "output_scores",
        "output_logits",
    )

    def __init__(self, **kwargs):
        # Parameters that control the length of the output
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.dola_layers = kwargs.pop("dola_layers", None)

        # Parameters that control the cache
        self.use_cache = kwargs.pop("use_cache", True)
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)
        if (
            self.cache_implementation is not None
            and self.cache_implementation in CACHE_CONFIG_MAPPING
        ):
            cache_config_class = CACHE_CONFIG_MAPPING[self.cache_implementation]
            if isinstance(self.cache_config, dict):
                self.cache_config = cache_config_class.from_dict(self.cache_config)
        self.return_legacy_cache = kwargs.pop("return_legacy_cache", None)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.min_p = kwargs.pop("min_p", None)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
        self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)
        self.constraints = kwargs.pop("constraints", None)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop(
            "exponential_decay_length_penalty", None
        )
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.forced_decoder_ids = kwargs.pop("forced_decoder_ids", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.token_healing = kwargs.pop("token_healing", False)
        self.guidance_scale = kwargs.pop("guidance_scale", None)
        self.low_memory = kwargs.pop("low_memory", None)
        watermarking_config = kwargs.pop("watermarking_config", None)
        if watermarking_config is None:
            self.watermarking_config = None
        elif isinstance(watermarking_config, BaseWatermarkingConfig):
            self.watermarking_config = watermarking_config
        else:
            self.watermarking_config = WatermarkingConfig.from_dict(watermarking_config)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Generation parameters exclusive to encoder-decoder models
        self.encoder_no_repeat_ngram_size = kwargs.pop(
            "encoder_no_repeat_ngram_size", 0
        )
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # Assistant generation
        self.is_assistant = False
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 20)
        self.num_assistant_tokens_schedule = kwargs.pop(
            "num_assistant_tokens_schedule", "constant"
        )
        self.assistant_confidence_threshold = kwargs.pop(
            "assistant_confidence_threshold", 0.4
        )
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)
        self.assistant_early_exit = kwargs.pop("assistant_early_exit", None)
        ## assistant generation for different tokenizers, the windows size for assistant/target model
        self.assistant_lookbehind = kwargs.pop("assistant_lookbehind", 10)
        self.target_lookbehind = kwargs.pop("target_lookbehind", 10)

        # Performances
        self.compile_config = kwargs.pop("compile_config", CompileConfig())
        self.disable_compile = kwargs.pop("disable_compile", False)
        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # added for constrative decoding
        self.cd_temperature = kwargs.pop("cd_temperature", 0.0)
        self.relative_top = kwargs.pop("relative_top", 0.1)

        # added for activation decoding
        self.alpha = kwargs.pop("alpha", 0.0)
        self.info_layer = kwargs.pop("info_layer", None)

        # added for attention contrastive decoding
        self.layer_indices = kwargs.pop("layer_indices", None)
        self.head_indices = kwargs.pop("head_indices", None)
        self.mature_layer = kwargs.pop("mature_layer", None)
        self.cd_temperature_two = kwargs.pop("cd_temperature_two", None)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def get_generation_mode(
        self, assistant_model: Optional["PreTrainedModel"] = None
    ) -> GenerationMode:
        """
        Returns the generation mode triggered by the [`GenerationConfig`] instance.

        Arg:
            assistant_model (`PreTrainedModel`, *optional*):
                The assistant model to be used for assisted generation. If set, the generation mode will be
                assisted generation.

        Returns:
            `GenerationMode`: The generation mode triggered by the instance.
        """
        # TODO joao: find out a way of not depending on external fields (e.g. `assistant_model`), then make this a
        # property and part of the `__repr__`
        if self.constraints is not None or self.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif self.num_beams == 1:
            if self.do_sample is False:
                if (
                    self.top_k is not None
                    and self.top_k > 1
                    and self.penalty_alpha is not None
                    and self.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if self.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif self.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        if assistant_model is not None:
            # Assisted generation may extend some generation modes
            if (
                self.prompt_lookup_num_tokens is not None
                or self.assistant_early_exit is not None
            ):
                if generation_mode in ("greedy_search", "sample"):
                    generation_mode = GenerationMode.ASSISTED_GENERATION
                else:
                    raise ValueError(
                        "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                        "is only supported with Greedy Search and Sample."
                    )

        # Contrastive decoding
        if (
            self.cd_temperature > 0.0
            and self.layer_indices is None
            and self.head_indices is None
        ):
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.CONTRASTIVE_GENERATION
            else:
                raise ValueError(
                    "You've set `cd_temperature`, which triggers Contrastive decoding."
                    "is only supported with Greedy Search and Sample."
                )

        # Attention Contrastive decoding
        if self.cd_temperature > 0.0 and self.layer_indices is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ATTENTION_CONTRASTIVE_DECODING
                # print("generation_mode", generation_mode)
            else:
                raise ValueError(
                    "You've set `cd_temperature`, `layer_indices` and `head_indices`, which triggers Attention Contrastive decoding."
                    "is only supported with Greedy Search and Sample."
                )

        # Activation decoding
        if self.alpha > 0.0 and self.info_layer is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ACTIVATION_DECODING
            else:
                raise ValueError(
                    "You've set `alpha` and `info_layer`, which triggers Activation decoding"
                    "is only supported with Greedy Search and Sample."
                )

        # Activation decoding
        if self.alpha > 0.0 and self.info_layer is None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ENTROPY_ENHANCED_DECODING
            else:
                raise ValueError(
                    "You've set `alpha`, which triggers END"
                    "is only supported with Greedy Search and Sample."
                )

        # DoLa generation may extend some generation modes
        if self.dola_layers is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.DOLA_GENERATION
            else:
                raise ValueError(
                    "You've set `dola_layers`, which triggers DoLa generate. Currently, DoLa generate "
                    "is only supported with Greedy Search and Sample."
                )

        return generation_mode


@torch.no_grad()
def _contrastive_decoding(
    self,
    input_ids: torch.LongTensor,
    assistant_model: "PreTrainedModel",
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    print("contrastive decdoing with assistant model", flush=True)

    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    # added for contrastive decoding
    c_model_kwargs = copy.deepcopy(model_kwargs)
    cd_logits_processor = (
        lambda scores, c_scores: scores - generation_config.cd_temperature * c_scores
    )
    model_forward = self.__call__

    is_prefill = True
    while self._has_unfinished_sequences(
        this_peer_finished,
        synced_gpus,
        device=input_ids.device,
        cur_len=cur_len,
        max_length=max_length,
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # ----------------------------Contrastive--------------------------
        c_model_inputs = self.prepare_inputs_for_generation(input_ids, **c_model_kwargs)

        c_model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        c_model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        c_outputs = assistant_model(**c_model_inputs, return_dict=True)

        c_model_kwargs = self._update_model_kwargs_for_generation(
            c_outputs,
            c_model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        c_next_token_logits = c_outputs.logits[:, -1, :].clone().float()
        c_next_token_logits = c_next_token_logits.to(input_ids.device)

        next_token_logits, c_next_token_logits = _relative_top_filter(
            next_token_logits, c_next_token_logits
        )
        next_token_logits = cd_logits_processor(next_token_logits, c_next_token_logits)
        # ------------------------------------------------------------------
        if synced_gpus and this_peer_finished:
            continue

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


@torch.no_grad()
def _attention_contrastive_decoding(
    self,
    input_ids: torch.LongTensor,
    assistant_model: "PreTrainedModel",
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    assistant_model_two: Optional["PreTrainedModel"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """

    def calculate_entropy(logits):
        scale_alpha = False

        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        if scale_alpha:
            entropy = entropy / np.log(probs.shape[-1])

        return entropy

    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    # added for contrastive decoding
    c_model_kwargs = copy.deepcopy(model_kwargs)
    cd_logits_processor = (
        lambda scores, c_scores: (1 + generation_config.cd_temperature) * scores
        - generation_config.cd_temperature * c_scores
    )
    cd_logits_processor_alpha = (
        lambda scores, c_scores, alpha: (1 + alpha) * scores - alpha * c_scores
    )
    model_forward = self.__call__

    is_prefill = True
    while self._has_unfinished_sequences(
        this_peer_finished,
        synced_gpus,
        device=input_ids.device,
        cur_len=cur_len,
        max_length=max_length,
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        os.environ.pop("LAYERS_TO_ABLATE", None)
        os.environ.pop("LAYER_TO_HEAD_MAP_PATH", None)

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True, output_hidden_states=True)
            is_prefill = False
        else:
            outputs = model_forward(
                **model_inputs, return_dict=True, output_hidden_states=True
            )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # ----------------------------Contrastive--------------------------
        c_model_inputs = self.prepare_inputs_for_generation(input_ids, **c_model_kwargs)

        c_model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        c_model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        os.environ["LAYERS_TO_ABLATE"] = " ".join(
            map(str, generation_config.layer_indices)
        )
        if generation_config.head_indices:
            os.environ["LAYER_TO_HEAD_MAP_PATH"] = generation_config.head_indices

        c_outputs = assistant_model(
            **c_model_inputs, return_dict=True, output_hidden_states=True
        )
        c_model_kwargs = self._update_model_kwargs_for_generation(
            c_outputs,
            c_model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        c_next_token_logits = c_outputs.logits[:, -1, :].clone().float()
        c_next_token_logits = c_next_token_logits.to(input_ids.device)

        next_token_logits, c_next_token_logits = _relative_top_filter(
            next_token_logits,
            c_next_token_logits,
            relative_top=generation_config.relative_top,
        )

        if os.environ.get("ATT_CD_MODE") == "decore":
            # print("att_cd decore")
            alpha = calculate_entropy(outputs.logits[0, -1])
            next_token_logits = cd_logits_processor_alpha(
                next_token_logits, c_next_token_logits, alpha
            )
        else:
            # print("att_cd default")
            next_token_logits = cd_logits_processor(
                next_token_logits, c_next_token_logits
            )

        # ------------------------------------------------------------------
        if synced_gpus and this_peer_finished:
            continue

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def relative_top_filter(
    scores: torch.FloatTensor,
    relative_top: float = 0.1,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    scores_normalized[scores_normalized < probs_thresh] = filter_value
    return scores_normalized


from transformers.cache_utils import DynamicCache, Cache


class DynamicCache_Last(Cache):
    def __init__(self, cache: DynamicCache):
        super().__init__()
        self._seen_tokens = cache._seen_tokens
        self.key_cache: List[torch.Tensor] = cache.key_cache[-1][
            :, :, :-1, :
        ].clone()  # b h prev_s d
        self.value_cache: List[torch.Tensor] = cache.value_cache[-1][
            :, :, :-1, :
        ].clone()  # b h prev_s d

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_cache = torch.cat([self.key_cache, key_states], dim=-2)
        value_cache = torch.cat([self.value_cache, value_states], dim=-2)

        return key_cache, value_cache


def _dola_decoding(
    self,
    input_ids: torch.LongTensor,
    dola_layers: Union[str, List[int]],
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: "BaseStreamer",
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **dola decoding** and can be
    used for decoder-only text models.
    The method is based on the paper "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language
    Models" (https://arxiv.org/abs/2309.03883) in ICLR 2024.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        dola_layers (`Union[str, List[int]]`):
            The candidate layers used in contrasting layers of DoLa. It can be either 1) 'low' or 'high', which
            means the lower part or higher part of the model layers, respectively, or 2) a list of layer indices
            to be used for candidate layers. The 0-th layer is the word embedding layer of the model.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`]
        or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """

    if self.config.is_encoder_decoder:
        raise ValueError("DoLa decoding is only available for decoder-only models.")
    # init values

    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    debug_list = []

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    this_peer_finished = False

    # prepare layers for DoLa decoding
    final_layer = self.config.get_text_config().num_hidden_layers
    # if the model has tied word embeddings, we skip the word embeddings (0-th) layer and start from the 2nd layer,
    # as the early exit from word embeddings will become identity function
    # if the model is really shallow (<=2 layers), we use the 1st layer if it's not the final layer and the 0-th
    # layer otherwise. Notice that DoLa does not help shallow models much.
    if not self.config.tie_word_embeddings:
        start_layer = 0
    elif final_layer > 2:
        start_layer = 2
    elif final_layer == 2:
        start_layer = 1
    else:
        start_layer = 0

    # For `N`-layer models with `N <= 40` layers, the layers of `range(0, N // 2, 2)` and `range(N // 2, N, 2)`
    # are used for `'low'` and `'high'` layers, respectively.
    # For models with `N > 40` layers, the layers of `range(0, 20, 2)` and `range(N - 20, N, 2)` are used for
    # `'low'` and `'high'` layers, respectively.
    if isinstance(dola_layers, str) and dola_layers == "low":
        if start_layer == final_layer // 2:
            candidate_premature_layers = [start_layer]
        else:
            candidate_premature_layers = (
                list(range(start_layer, final_layer // 2, 2))
                if final_layer <= 40
                else list(range(start_layer, 20, 2))
            )
    elif isinstance(dola_layers, str) and dola_layers == "high":
        candidate_premature_layers = (
            list(range(final_layer // 2, final_layer, 2))
            if final_layer <= 40
            else list(range(final_layer - 20, final_layer, 2))
        )
    # Set the `dola_layers` to a list of integers for layer indices to contrast manually specified layers.
    elif isinstance(dola_layers, list):
        candidate_premature_layers = [i for i in dola_layers if i < final_layer]
    else:
        raise ValueError(
            "dola_layers must be either 'low', 'high' or a list of integers."
        )

    lm_head = self.get_output_embeddings()
    if lm_head is None:
        raise ValueError(
            "DoLa is not supported for models that don't have output embeddings."
        )

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        if os.getenv("WITH_LAST") is not None:
            last_block = self.model.layers[-1]
            temp_cache = DynamicCache_Last(outputs.past_key_values)
            last_block.cached_kwargs["past_key_value"] = temp_cache
            last_block.cached_kwargs["position_ids"] = last_block.cached_kwargs[
                "position_ids"
            ][:, -1:]
            last_block.cached_kwargs["cache_position"] = last_block.cached_kwargs[
                "cache_position"
            ][-1:]
            last_block.cached_kwargs["position_embeddings"] = (
                last_block.cached_kwargs["position_embeddings"][0][:, -1:, :],
                last_block.cached_kwargs["position_embeddings"][1][:, -1:, :],
            )
        # .float() is needed to retain precision for later logits manipulations
        final_layer_next_token_logits = (
            outputs.logits[:, -1, :].detach().clone().float()
        )
        final_logits = outputs.logits[:, -1, :].float()
        candidate_premature_logits = {}
        for candidate_premature_layer in candidate_premature_layers:
            candidate_hidden_states = outputs.hidden_states[candidate_premature_layer][
                :, -1, :
            ]
            if os.getenv("WITH_LAST") is not None:
                candidate_hidden_states = last_block(
                    candidate_hidden_states.unsqueeze(-2), **last_block.cached_kwargs
                )[0].squeeze(-2)
            if os.getenv("NORM") is not None:
                candidate_hidden_states = self.model.norm(candidate_hidden_states)
            candidate_premature_logits[candidate_premature_layer] = lm_head(
                candidate_hidden_states
            ).to(final_logits.device)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # import code; code.interact('dola line 679', local=dict(globals(), **locals()))
        next_token_logits, debug_logits = _dola_select_contrast(
            candidate_premature_layers, candidate_premature_logits, final_logits
        )
        debug_list.append(
            [final_logits.clone().detach(), debug_logits[0], debug_logits[1]]
        )

        next_token_logits = next_token_logits.to(input_ids.device)
        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (final_layer_next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if do_sample:  # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:  # argmax
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        # stop when each sentence is finished
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return DolaGenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
            dola_logits=debug_list,
        )
    else:
        return input_ids


def _dola_select_contrast(
    candidate_premature_layers: List[int],
    candidate_premature_logits: Dict[int, torch.FloatTensor],
    final_logits: torch.FloatTensor,
) -> torch.FloatTensor:
    if len(candidate_premature_layers) == 1:
        base_logits = candidate_premature_logits[candidate_premature_layers[0]]
        final_logits, base_logits = _relative_top_filter(final_logits, base_logits)
        logits = final_logits - base_logits
        return logits

    # 1. Stacking all premature_layers into a new dimension
    stacked_premature_layers = torch.stack(
        [candidate_premature_logits[i] for i in candidate_premature_layers], dim=0
    )

    # 2. Calculate the softmax values for mature_layer and all premature_layers
    # shape: (batch_size, vocab_size)
    softmax_mature_layer = F.softmax(final_logits, dim=-1)
    # shape: (num_premature_layers, batch_size, vocab_size)
    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)

    # 3. Calculate the average distribution
    # shape: (num_premature_layers, batch_size, vocab_size)
    avg_dist = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)

    # 4. Calculate log-softmax for the KL divergence
    # shape: (batch_size, vocab_size)
    log_softmax_mature_layer = F.log_softmax(final_logits, dim=-1)
    # shape: (num_premature_layers, batch_size, vocab_size)
    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)

    # 5. Calculate the KL divergences and then the JS divergences
    # shape: (num_premature_layers, batch_size)
    kl1 = F.kl_div(
        log_softmax_mature_layer[None, :, :], avg_dist, reduction="none"
    ).mean(-1)
    # shape: (num_premature_layers, batch_size)
    kl2 = F.kl_div(log_softmax_premature_layers, avg_dist, reduction="none").mean(-1)
    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

    # 6. Reduce the batchmean
    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
    base_logits = candidate_premature_logits[premature_layer]
    debug = (premature_layer, base_logits.clone().detach())

    final_logits, base_logits = _relative_top_filter(final_logits, base_logits)
    logits = final_logits - base_logits
    return logits, debug


@torch.no_grad()
def _activation_decoding(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    model_forward = self.__call__

    entropy = None
    is_prefill = True
    while self._has_unfinished_sequences(
        this_peer_finished,
        synced_gpus,
        device=input_ids.device,
        cur_len=cur_len,
        max_length=max_length,
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        # model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            model_inputs.update(
                {"output_hidden_states": True}
            )  # True for Activation decoding
            outputs = self(**model_inputs, return_dict=True)
            info_layer_hidden_states = outputs.hidden_states[
                generation_config.info_layer
            ]  # [b, s, d]
            if os.getenv("NORM") is not None:
                info_layer_hidden_states = self.model.norm(info_layer_hidden_states)
            info_layer_score = self.get_output_embeddings()(
                info_layer_hidden_states
            ).float()  # [b, s, v]
            info_layer_probs = F.softmax(
                info_layer_score.transpose(-2, -1), dim=-1
            )  # [b, v, s]
            entropy = torch.distributions.Categorical(
                probs=info_layer_probs, validate_args=False
            ).entropy()  # [b, v, s] -> [b, v]
            is_prefill = False
        else:
            model_inputs.update({"output_hidden_states": False})
            outputs = model_forward(**model_inputs, return_dict=True)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)

        if synced_gpus and this_peer_finished:
            continue

        # import code; code.interact('entropy line 900', local=dict(globals(), **locals()))
        next_token_logits = next_token_logits.log_softmax(dim=-1)  # normalize
        relative_top = 0.1  # this value can be arguments
        if relative_top > 0.0:
            next_token_logits = relative_top_filter(next_token_logits, relative_top)
        mask = next_token_logits < -1e3

        final_entropy = entropy.clone()
        final_entropy[mask] = float("Inf")
        next_token_logits -= generation_config.alpha * final_entropy

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


@torch.no_grad()
def _entropy_enhanced_decoding(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    model_forward = self.__call__

    entropy = None
    is_prefill = True

    # prepare layers for END
    final_layer = self.config.get_text_config().num_hidden_layers
    candidate_premature_layers = list(range(final_layer // 2, final_layer))
    lm_head = self.get_output_embeddings()
    # import code; code.interact('END', local=dict(globals(), **locals()))

    while self._has_unfinished_sequences(
        this_peer_finished,
        synced_gpus,
        device=input_ids.device,
        cur_len=cur_len,
        max_length=max_length,
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update({"output_hidden_states": True})
        # model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)

        final_probs = F.softmax(next_token_logits, dim=-1)  # [b, V]
        max_probs, _ = final_probs.max(dim=-1, keepdim=True)  # [b, 1]
        vhead_mask = (
            final_probs >= generation_config.relative_top * max_probs
        )  # [b, V] bool

        candidate_premature_logits = []
        for candidate_premature_layer in candidate_premature_layers:
            candidate_hidden_states = outputs.hidden_states[candidate_premature_layer][
                :, -1, :
            ]
            # if os.getenv('NORM') is not None:
            #     candidate_hidden_states = self.model.norm(candidate_hidden_states)
            candidate_logits = F.softmax(
                lm_head(candidate_hidden_states).float(), dim=-1
            ).to(next_token_logits.device)
            candidate_premature_logits.append(candidate_logits)

        candidate_premature_probs = torch.stack(
            candidate_premature_logits, dim=-1
        )  # b, v, L
        candidate_premature_probs *= vhead_mask.unsqueeze(-1)
        candidate_premature_probs /= candidate_premature_probs.sum(dim=-1, keepdim=True)
        entropy = torch.distributions.Categorical(
            probs=candidate_premature_probs, validate_args=False
        ).entropy()  # b, v
        entropy = entropy.masked_fill(~vhead_mask, float("inf"))

        next_token_logits = next_token_logits.log_softmax(dim=-1)  # normalize
        next_token_logits -= generation_config.alpha * entropy

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        if synced_gpus and this_peer_finished:
            continue

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


@torch.no_grad()
def generate_custom(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    assistant_model_two: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config ([`~generation.GenerationConfig`], *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which has the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complements the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
            to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
            deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
    """

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    tokenizer = kwargs.pop(
        "tokenizer", None
    )  # Pull this out first, we only use it for stopping criteria
    assistant_tokenizer = kwargs.pop(
        "assistant_tokenizer", None
    )  # only used for assisted generation

    generation_config, model_kwargs = self._prepare_generation_config(
        generation_config, **kwargs
    )
    # added for contrastive decoding
    generation_config.cd_temperature = model_kwargs.pop("cd_temperature", 0.0) or 0.0
    generation_config.relative_top = model_kwargs.pop("relative_top", 0.1) or 0.1

    # added for activation decoding
    generation_config.alpha = model_kwargs.pop("alpha", 0.0)
    generation_config.info_layer = model_kwargs.pop("info_layer", None)

    # added for attention contrastive decoding
    generation_config.layer_indices = model_kwargs.pop("layer_indices", 0.0)
    generation_config.head_indices = model_kwargs.pop("head_indices", None)
    generation_config.mature_layer = model_kwargs.pop("mature_layer", None)
    generation_config.cd_temperature_two = model_kwargs.pop("cd_temperature_two", None)

    self._validate_model_kwargs(model_kwargs.copy())
    self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        synced_gpus = (
            is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)
        ) and dist.get_world_size() > 1

    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )

    accepts_attention_mask = "attention_mask" in set(
        inspect.signature(self.forward).parameters.keys()
    )
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(
        generation_config, kwargs_has_attention_mask, device=device
    )

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor)
            > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        generation_config.use_cache = True

    if (
        not kwargs_has_attention_mask
        and requires_attention_mask
        and accepts_attention_mask
    ):
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if (
            model_input_name == "input_ids"
            and len(model_kwargs["attention_mask"].shape) > 2
        ):
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = (
            inputs_tensor
            if model_input_name == "input_ids"
            else model_kwargs.pop("input_ids")
        )

    if generation_config.token_healing:
        input_ids = self.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = (
        kwargs.get("max_length") is None and generation_config.max_length is not None
    )
    has_default_min_length = (
        kwargs.get("min_length") is None and generation_config.min_length is not None
    )
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
        model_kwargs["logits_to_keep"] = 1

    self._validate_generated_length(
        generation_config, input_ids_length, has_default_max_length
    )

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    max_cache_length = generation_config.max_length - 1
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not self.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    self._prepare_cache_for_generation(
        generation_config,
        model_kwargs,
        assistant_model,
        batch_size,
        max_cache_length,
        device,
    )

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        tokenizer=tokenizer,
        **kwargs,
    )

    # Set model_kwargs `use_cache` so we can use it later in forward runs
    model_kwargs["use_cache"] = generation_config.use_cache

    # 10. go into different generation modes
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing assisted generate, "
                f"but is {generation_config.num_return_sequences}."
            )
        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")
        if generation_config.cache_implementation in [
            "static",
            "hybrid",
            "sliding_window",
        ]:
            raise ValueError(
                "assisted generate is not supported with Static cache classes`"
            )
        if self._is_stateful:
            # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
            # which is not possible with stateful models (they can't reset to a previous subset of generated text)
            raise ValueError(
                f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
            )

        # 11. Get the candidate generator, given the parameterization
        candidate_generator = self._get_candidate_generator(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            assistant_model=assistant_model,
            logits_processor=logits_processor,
            target_tokenizer=tokenizer,
            assistant_tokenizer=assistant_tokenizer,
            model_kwargs=model_kwargs,
        )

        # 12. run assisted generate
        result = self._assisted_decoding(
            input_ids,
            candidate_generator=candidate_generator,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    elif generation_mode == GenerationMode.CONTRASTIVE_GENERATION:
        if not model_kwargs["use_cache"]:
            raise ValueError("contrastive generate requires `use_cache=True`")
        if self._is_stateful:
            raise ValueError(
                f"contrastive generation is not supported with stateful models, such as {self.__class__.__name__}"
            )

        result = self._contrastive_decoding(
            input_ids,
            assistant_model=assistant_model,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    elif generation_mode == GenerationMode.ATTENTION_CONTRASTIVE_DECODING:
        if not model_kwargs["use_cache"]:
            raise ValueError("Attention contrastive decoding requires `use_cache=True`")
        if self._is_stateful:
            raise ValueError(
                f"Attention contrastive decoding is not supported with stateful models, such as {self.__class__.__name__}"
            )

        result = self._attention_contrastive_decoding(
            input_ids,
            assistant_model=assistant_model,
            assistant_model_two=assistant_model_two,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    elif generation_mode == GenerationMode.ACTIVATION_DECODING:
        if not model_kwargs["use_cache"]:
            raise ValueError("Activation decoding requires `use_cache=True`")
        if self._is_stateful:
            raise ValueError(
                f"Activation decoding is not supported with stateful models, such as {self.__class__.__name__}"
            )

        result = self._activation_decoding(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    elif generation_mode == GenerationMode.ENTROPY_ENHANCED_DECODING:
        if not model_kwargs["use_cache"]:
            raise ValueError("Entropy enhanced decoding requires `use_cache=True`")
        if self._is_stateful:
            raise ValueError(
                f"Entropy enhanced decoding is not supported with stateful models, such as {self.__class__.__name__}"
            )

        result = self._entropy_enhanced_decoding(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    elif generation_mode == GenerationMode.DOLA_GENERATION:
        if self._is_stateful:
            # DoLa decoding was not designed for stateful models, and would require some changes
            raise ValueError(
                f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
            )
        result = self._dola_decoding(
            input_ids,
            dola_layers=generation_config.dola_layers,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
        if not model_kwargs["use_cache"]:
            raise ValueError("Contrastive search requires `use_cache=True`")
        if self._is_stateful:
            # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
            raise ValueError(
                f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
            )

        result = self._contrastive_search(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = self._sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run beam sample
        result = self._beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            num_beam_groups=generation_config.num_beam_groups,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._group_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
        final_constraints = []
        if generation_config.constraints is not None:
            final_constraints = generation_config.constraints

        if generation_config.force_words_ids is not None:

            def typeerror():
                raise ValueError(
                    "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                    f"of positive integers, but is {generation_config.force_words_ids}."
                )

            if (
                not isinstance(generation_config.force_words_ids, list)
                or len(generation_config.force_words_ids) == 0
            ):
                typeerror()

            for word_ids in generation_config.force_words_ids:
                if isinstance(word_ids[0], list):
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any(not isinstance(token_ids, list) for token_ids in word_ids):
                        typeerror()
                    if any(
                        any(
                            (not isinstance(token_id, int) or token_id < 0)
                            for token_id in token_ids
                        )
                        for token_ids in word_ids
                    ):
                        typeerror()

                    constraint = DisjunctiveConstraint(word_ids)
                else:
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any(
                        (not isinstance(token_id, int) or token_id < 0)
                        for token_id in word_ids
                    ):
                        typeerror()

                    constraint = PhrasalConstraint(word_ids)
                final_constraints.append(constraint)

        # 11. prepare beam search scorer
        constrained_beam_scorer = ConstrainedBeamSearchScorer(
            constraints=final_constraints,
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._constrained_beam_search(
            input_ids,
            constrained_beam_scorer=constrained_beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is True
        and not is_torchdynamo_compiling()
        and hasattr(result, "past_key_values")
        and getattr(result.past_key_values, "to_legacy_cache") is not None
    ):
        result.past_key_values = result.past_key_values.to_legacy_cache()
    return result


def monkeypatch_cd():
    GenerationMixin._contrastive_decoding = _contrastive_decoding
    GenerationMixin._attention_contrastive_decoding = _attention_contrastive_decoding
    GenerationMixin._dola_decoding = _dola_decoding
    GenerationMixin._activation_decoding = _activation_decoding
    GenerationMixin._entropy_enhanced_decoding = _entropy_enhanced_decoding
    GenerationMixin.generate = generate_custom
    GenerationConfig.__init__ = GenerationConfigCustom.__init__
    GenerationConfig.get_generation_mode = GenerationConfigCustom.get_generation_mode
