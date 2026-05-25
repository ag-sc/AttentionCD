import torch
import torch.nn as nn

import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    _relative_top_filter, 
    GenerateDecoderOnlyOutput, 
    GenerateEncoderDecoderOutput, 
    GenerateNonBeamOutput,
    GenerationMixin,
)

from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.utils import add_start_docstrings_to_model_forward, logging
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING, 
    LlamaForCausalLM, 
    apply_rotary_pos_emb, 
    eager_attention_forward, 
    ALL_ATTENTION_FUNCTIONS, 
)

from .cd_utils import monkeypatch_cd
from accelerate import init_empty_weights

logger = logging.get_logger(__name__)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from types import MethodType

from dataclasses import dataclass
from transformers.utils import ModelOutput

@dataclass
class CDGenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    cd_logits: Optional[List[List[torch.FloatTensor]]] = None

@torch.no_grad()
def prunecd(
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
    # print(f"PruneCD generation with cd_temp: {generation_config.cd_temperature} relative_top: {generation_config.relative_top}", flush=True)

    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )
        
    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    cd_logits_processor = lambda scores, c_scores, cd_temp: scores - cd_temp * c_scores

    model_forward = self.__call__

    is_prefill = True

    debug = True
    if debug:
        logits_list = []

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
    ):
        # prepare model inputs
        # import code; code.interact('monkeypatch _contrastive_decoding line 116', local=dict(globals(), **locals()))
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

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
        
        batch_org = outputs.logits.shape[0] // 2
        
        next_token_logits = outputs.logits[:batch_org, -1, :].clone().float()
        c_next_token_logits = outputs.logits[batch_org:, -1, :].clone().float()
        
        if debug:
            logits_list.append([next_token_logits.clone().detach(), c_next_token_logits.clone().detach()])
        
        next_token_logits = next_token_logits.to(input_ids.device)
        c_next_token_logits = c_next_token_logits.to(input_ids.device)
        
        next_token_logits, c_next_token_logits = _relative_top_filter(next_token_logits, c_next_token_logits, relative_top=generation_config.relative_top)
        next_token_logits = cd_logits_processor(next_token_logits, c_next_token_logits, generation_config.cd_temperature)

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
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
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
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
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
            if debug:
                return CDGenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    cd_logits = logits_list
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

def forward_block_naive(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if not getattr(self, 'attn_skip', False):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        import code; code.interact(f'Layer {self.self_attn.layer_idx} forward_block_naive transform 1 ', local=dict(globals(), **locals()))
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
    elif os.getenv("TRANSFORM"):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        n_repeat = self.self_attn.config.num_attention_heads // self.self_attn.config.num_key_value_heads
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.self_attn.head_dim)
        hidden_states = self.self_attn.o_proj(self.self_attn.v_proj(hidden_states).view(hidden_shape).repeat_interleave(n_repeat, dim=-2).reshape(*input_shape,-1))
        
        hidden_states = residual + hidden_states
    
    if not getattr(self, 'mlp_skip', False):
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        raise ValueError

    return outputs

def forward_block_naive_keepnorm(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    if not getattr(self, 'attn_skip', False):
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    elif os.getenv("TRANSFORM"):
        n_repeat = self.self_attn.config.num_attention_heads // self.self_attn.config.num_key_value_heads
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.self_attn.head_dim)
        hidden_states = self.self_attn.o_proj(self.self_attn.v_proj(hidden_states).view(hidden_shape).repeat_interleave(n_repeat, dim=-2).reshape(*input_shape,-1))
    hidden_states = residual + hidden_states
    
    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    if not getattr(self, 'mlp_skip', False):
        hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        raise ValueError

    return outputs

def update_cache_share(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

    Parameters:
        key_states (`torch.Tensor`):
            The new key states to cache.
        value_states (`torch.Tensor`):
            The new value states to cache.
        layer_idx (`int`):
            The index of the layer to cache the states for.
        cache_kwargs (`Dict[str, Any]`, `optional`):
            Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

    Return:
        A tuple containing the updated key and value states.
    """
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += key_states.shape[-2]
    b = key_states.shape[0]
    attn_skip = cache_kwargs.get('attn_skip', False)
    # Update the cache
    if key_states is not None:
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            if attn_skip:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache.append(key_states[:b//2])
                self.value_cache.append(value_states[:b//2])
                
                return key_states, value_states
        elif (
            len(self.key_cache[layer_idx]) == 0
        ):  # fills previously skipped layers; checking for tensor causes errors
            if attn_skip:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = key_states[:b//2]
                self.value_cache[layer_idx] = value_states[:b//2]
                
                return key_states, value_states
        else:
            if attn_skip:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                
            else:
                c_key_state = torch.cat([self.key_cache[layer_idx], key_states[b//2:]], dim=-2)
                c_value_state = torch.cat([self.value_cache[layer_idx], value_states[b//2:]], dim=-2)
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states[:b//2]], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states[:b//2]], dim=-2)
                
                return torch.concat([self.key_cache[layer_idx], c_key_state], dim=0), \
                        torch.concat([self.value_cache[layer_idx], c_value_state], dim=0)
    
    return self.key_cache[layer_idx], self.value_cache[layer_idx]

def forward_attn(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        if kwargs.get('cache_share', False):
            past_key_value.update = MethodType(update_cache_share, past_key_value)
        cache_kwargs['attn_skip'] = kwargs.get('attn_skip', False)
        
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def forward_block(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    b, _, _ = hidden_states.shape
    
    calc_b = b // 2 if self.attn_skip else b
    hidden_states_ = hidden_states[:calc_b]
    attention_mask_ = attention_mask[:calc_b]
    if isinstance(position_embeddings, Tuple):
        temp = []
        for i in range(len(position_embeddings)):
            temp.append(position_embeddings[i][:calc_b])
        position_embeddings_ = tuple(temp)
    # hidden_states_ = hidden_states
    # attention_mask_ = attention_mask
    # position_embeddings_ = position_embeddings
        
    residual = hidden_states_
    
    hidden_states_ = self.input_layernorm(hidden_states_)
    
    # Self Attention
    hidden_states_, self_attn_weights = self.self_attn(
        hidden_states=hidden_states_,
        attention_mask=attention_mask_,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings_,
        attn_skip=self.attn_skip,
        cache_share=self.cache_share,
        **kwargs,
    )
    hidden_states_ = residual + hidden_states_
    
    # update result
    if self.attn_skip:
        hidden_states[:calc_b] = hidden_states_
    else:
        hidden_states = hidden_states_
    # hidden_states = hidden_states_
    
    # import code; code.interact(f'Layer {self.self_attn.layer_idx} forward_block mid',local=dict(globals(), **locals()))
    
    # Fully Connected
    calc_b = b // 2 if self.mlp_skip else b
    hidden_states_ = hidden_states[:calc_b]
        
    residual = hidden_states_
    hidden_states_ = self.post_attention_layernorm(hidden_states_)
    hidden_states_ = self.mlp(hidden_states_)
    hidden_states_ = residual + hidden_states_

    if self.mlp_skip:
        hidden_states[:calc_b] = hidden_states_
    else:
        hidden_states = hidden_states_
    
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs

@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def forward_model(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    
    position_ids = position_ids.repeat([2 if i==0 else 1 for i in range(len(position_ids.size()))])
    causal_mask = causal_mask.repeat([2 if i==0 else 1 for i in range(len(causal_mask.size()))])
    hidden_states = hidden_states.repeat([2 if i==0 else 1 for i in range(len(hidden_states.size()))])
    if isinstance(position_embeddings, Tuple):
        temp = []
        for i in range(len(position_embeddings)):
            temp.append(position_embeddings[i].repeat([2 if i==0 else 1 for i in range(len(hidden_states.size()))]))
        position_embeddings = tuple(temp)
        
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    output = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    return output if return_dict else output.to_tuple()

def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and (attention_mask == 0.0).any():
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    # if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
    #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
    #         attention_mask,
    #         inputs_embeds=input_tensor,
    #         past_key_values_length=past_seen_tokens,
    #         is_training=self.training,
    #     ):
    #         return None

    dtype, device = input_tensor.dtype, input_tensor.device
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_cache_shape()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
    )

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type in ["cuda", "xpu"]
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        min_dtype = torch.finfo(dtype).min
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask

def monkeypatch_prunecd_for_analysis(model, skip_list):
    assert isinstance(model, LlamaForCausalLM)
    
    if len(skip_list) == 1:
        attn_skip_list = mlp_skip_list = skip_list[0]
    elif len(skip_list) == 2:
        attn_skip_list = skip_list[0]
        mlp_skip_list = skip_list[1]
        if -1 in attn_skip_list:
            attn_skip_list = []
        if -1 in mlp_skip_list:
            mlp_skip_list = []
    else:
        raise ValueError
    
    skip_block_list = attn_skip_list + mlp_skip_list
    skip_block_list = list(set(skip_block_list))
    # import code; code.interact(local=dict(globals(), **locals()))
    with init_empty_weights():
        assistant_model = transformers.AutoModelForCausalLM.from_config(model.config)
    assistant_model.load_state_dict(model.state_dict(), strict=False, assign=True)
    
    for skip_idx in attn_skip_list:
        layer = assistant_model.model.layers[skip_idx]
        layer.attn_skip = True
        layer.forward = MethodType(forward_block_naive, layer)
        
    for skip_idx in mlp_skip_list:
        layer = assistant_model.model.layers[skip_idx]
        layer.mlp_skip = True
        layer.forward = MethodType(forward_block_naive, layer)
    
    assistant_model.to(model.device)
    
    return assistant_model

def monkeypatch_prunecd_llama_naive(model, assistant_model_name, skip_list_txt, **kwargs):
    assert isinstance(model, LlamaForCausalLM)
    if assistant_model_name is None:
        with open(skip_list_txt, 'r') as f:
            lines = f.readlines()
            
        skip_list = []
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                skip_list.append(list(map(int, line.split(','))))
        
        if len(skip_list) == 1:
            attn_skip_list = mlp_skip_list = skip_list[0]
        elif len(skip_list) == 2:
            attn_skip_list = skip_list[0]
            mlp_skip_list = skip_list[1]
            if -1 in attn_skip_list:
                attn_skip_list = []
            if -1 in mlp_skip_list:
                mlp_skip_list = []
        else:
            raise ValueError
        
        skip_block_list = attn_skip_list + mlp_skip_list
        skip_block_list = list(set(skip_block_list))
        # import code; code.interact(local=dict(globals(), **locals()))
        with init_empty_weights():
            assistant_model = transformers.AutoModelForCausalLM.from_config(model.config)
        assistant_model.load_state_dict(model.state_dict(), strict=False, assign=True)
        
        forward_fn = forward_block_naive_keepnorm if os.getenv("KEEPNORM", False) else forward_block_naive
        for layer_idx, layer in enumerate(assistant_model.model.layers):
            layer.attn_skip = layer_idx in attn_skip_list
            layer.mlp_skip = layer_idx in mlp_skip_list
            layer.forward = MethodType(forward_fn, layer)
    
    else:
        print(f"assistant model loading... {assistant_model_name}", flush=True)
        try:
            assistant_model = transformers.AutoModelForCausalLM.from_pretrained(assistant_model_name, torch_dtype=torch.bfloat16)
        except Exception as e:
            print(f"Error loading assistant model: {e}")
            raise

    assistant_model.to(model.device)
    monkeypatch_cd()
    return assistant_model
    
def monkeypatch_prunecd_llama_efficient(model, skip_list_txt, cache_share=False):
    assert isinstance(model, LlamaForCausalLM)
    print(f"monkeypatching efficient with skip_list_txt: {skip_list_txt}")
    
    with open(skip_list_txt, 'r') as f:
        lines = f.readlines()
        
    skip_list = []
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            skip_list.append(list(map(int, line.split(','))))
    
    if len(skip_list) == 1:
        attn_skip_list = mlp_skip_list = skip_list[0]
    elif len(skip_list) == 2:
        attn_skip_list = skip_list[0]
        mlp_skip_list = skip_list[1]
        if -1 in attn_skip_list:
            attn_skip_list = []
        if -1 in mlp_skip_list:
            mlp_skip_list = []
    else:
        raise ValueError
    
    for layer_idx, layer in enumerate(model.model.layers):
        layer.self_attn.forward = MethodType(forward_attn, layer.self_attn)
        layer.forward = MethodType(forward_block, layer)
        layer.attn_skip = (layer_idx in attn_skip_list)
        layer.cache_share = cache_share
        layer.mlp_skip = (layer_idx in mlp_skip_list)
        if layer_idx in attn_skip_list:
            print(f"skip attn layer {layer_idx}")
        if layer_idx in mlp_skip_list:
            print(f"skip mlp layer {layer_idx}")

    
    model.model._update_causal_mask = MethodType(_update_causal_mask, model.model)
    model.model.forward = MethodType(forward_model, model.model)
    
    monkeypatch_cd()
    GenerationMixin._contrastive_decoding = prunecd
    