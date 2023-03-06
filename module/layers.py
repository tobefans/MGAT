import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import torch_int_div
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from timm.models.layers import DropPath, trunc_normal_

logger = logging.get_logger(__name__)

@dataclass
class MMBaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->MM
class MMPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        self.padding = MMSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer with Wav2Vec2->MM
class MMSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class MMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        num_buckets: int = 320,
        max_distance: int = 800,
        has_relative_position_bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        if has_relative_position_bias:
            self.rel_attn_embed = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        index=0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Attention layer with relative attention"""
        bsz, tgt_len, _ = hidden_states.size()
        src_len = key_value_states.size(1) if key_value_states is not None else tgt_len

        # first pass of attention layer creates position bias
        if position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len) # num_heads x query_length x key_length
            position_bias = (
                position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)
            )

        # Compute relative position bias:
        # 1) get reshape hidden_states, BxTx(H*E) -> BxTxHxE -> BxHxTxE
        gated_hidden_states = hidden_states.view(hidden_states.shape[:-1] + (self.num_heads, -1))
        gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)

        # 2) project hidden states, BxHxTxE -> BxHxTx2
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        relative_position_proj = relative_position_proj.view(gated_hidden_states.shape[:-1] + (2, 4)).sum(-1)

        # 3) compute gate for position bias from projected hidden states, BxHxTx1
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0

        # 4) apply gate to position bias to compute gated position_bias, (B*H)xTxS
        gated_position_bias = gate_output.view(bsz * self.num_heads, -1, 1) * position_bias
        gated_position_bias = gated_position_bias.view((-1, tgt_len, src_len))

        attn_output, attn_weights = self.torch_multi_head_self_attention(
            hidden_states, key_value_states, attention_mask, gated_position_bias, output_attentions
        )

        return attn_output, attn_weights, position_bias

    def torch_multi_head_self_attention(
        self,
        hidden_states: torch.FloatTensor,
        key_value_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[Union[torch.LongTensor, torch.BoolTensor]] = None,
        gated_position_bias: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """simple wrapper around torch's multi_head_attention_forward function"""
        # self-attention assumes q = k = v
        if key_value_states is not None:
            key = value = key_value_states.transpose(0, 1)
        else:
            key = value = hidden_states.transpose(0, 1)
        query = hidden_states.transpose(0, 1)
        key_padding_mask = attention_mask.ne(1) if attention_mask is not None else None

        # disable bias and add_zero_attn
        bias_k = bias_v = None
        add_zero_attn = False

        # PyTorch 1.3.0 has F.multi_head_attention_forward defined
        # so no problem with backwards compatibility
        attn_output, attn_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            torch.empty([0]),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k,
            bias_v,
            add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            key_padding_mask,
            output_attentions,
            gated_position_bias,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )

        # [Seq_Len, Batch Size, ...] -> [Batch Size, Seq_Len, ...]
        attn_output = attn_output.transpose(0, 1)

        if attn_weights is not None:
            # IMPORTANT: Attention weights are averaged weights
            # here which should not be the case. This is an open issue
            # on PyTorch: https://github.com/pytorch/pytorch/issues/32590
            attn_weights = attn_weights[:, None].broadcast_to(
                attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:]
            )

        return attn_output, attn_weights

    def compute_bias(self, query_length: int, key_length: int) -> torch.FloatTensor:
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # query_length x key_length
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        values = self.rel_attn_embed(relative_position_bucket)  # query_length x key_length x num_heads
        values = values.permute([2, 0, 1])  # num_heads x query_length x key_length
        return values

    def _relative_positions_bucket(self, relative_positions: torch.FloatTensor) -> torch.FloatTensor:
        num_buckets = self.num_buckets // 2

        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        relative_positions = torch.abs(relative_positions)

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_positions_if_large = torch.log(relative_positions.float() / max_exact)
        relative_positions_if_large = relative_positions_if_large / math.log(self.max_distance / max_exact)
        relative_positions_if_large = relative_positions_if_large * (num_buckets - max_exact)
        relative_postion_if_large = (max_exact + relative_positions_if_large).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward with Wav2Vec2->MM
class MMFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class MMEncoderLayer(nn.Module):
    def __init__(self, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = MMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = MMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, key_value_states=None, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            key_value_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)

        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MMEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = MMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = MMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, key_value_states=None, attention_mask=None, position_bias=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            key_value_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = MMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [MMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        attention_mask=None,
        attention_mask_kv=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states[~attention_mask] = 0.0

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if key_value_states is not None:
            position_embeddings = self.pos_conv_embed(key_value_states)
            key_value_states = key_value_states + position_embeddings
            key_value_states = self.layer_norm(key_value_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                        position_bias,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        key_value_states,
                        attention_mask=attention_mask_kv,
                        position_bias=position_bias,
                        output_attentions=output_attentions,
                        index=i,
                    )

                hidden_states, position_bias = layer_outputs[:2]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MMEncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = MMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [
                MMEncoderLayerStableLayerNorm(config, has_relative_position_bias=(i == 0))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        attention_mask=None,
        attention_mask_kv=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states[~attention_mask] = 0

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        if key_value_states is not None:
            position_embeddings = self.pos_conv_embed(key_value_states)
            key_value_states = key_value_states + position_embeddings

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                        position_bias,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        key_value_states,
                        attention_mask=attention_mask_kv,
                        output_attentions=output_attentions,
                        position_bias=position_bias,
                    )
                hidden_states, position_bias = layer_outputs[:2]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Adapter with Wav2Vec2->MM
class MMAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # feature dim might need to be down-projected
        if config.output_hidden_size != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        self.layers = nn.ModuleList(MMAdapterLayer(config) for _ in range(config.num_adapter_layers))
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        # down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AdapterLayer with Wav2Vec2->MM
class MMAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
        )

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model with Wav2Vec2->MM, wav2vec2->MM, WAV_2_VEC_2->MM
class TransformerLayers(PreTrainedModel):
    def __init__(self, cfg, config):
        super().__init__(config)
        self.config = config

        if config.do_stable_layer_norm:
            self.encoder = MMEncoderStableLayerNorm(config)
        else:
            self.encoder = MMEncoder(config)

        self.adapter = MMAdapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, MMPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_kv: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MMBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            hidden_states,
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            attention_mask_kv=attention_mask_kv,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        # if self.adapter is not None:
        #     hidden_states = self.adapter(hidden_states)

        # if not return_dict:
        #     return (hidden_states,) + encoder_outputs[1:]

        # return MMBaseModelOutput(
        #     last_hidden_state=hidden_states,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )

        return hidden_states

class LSTMLayers(nn.Module):
    def __init__(self, cfg, config):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(cfg.model.embed_dim, cfg.model.embed_dim//2, 2, bidirectional=True, batch_first=True)

    def forward(self, hidden_states, **kwargs):
        out, (_, _) = self.lstm(hidden_states)
        return out



def _get_activation_fn(activation: str):
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

def create_PositionalEncoding(input_dim, max_seq_len=2000): 
    position_encoding = np.array([ 
        [pos / np.power(10000, 2.0 * (j // 2) / input_dim) for j in range(input_dim)] 
        for pos in range(max_seq_len)]) 
    
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
    position_encoding = nn.Parameter(position_encoding, requires_grad=False) 
    
    return position_encoding

class Multihead_attention(nn.Module):
    '''
    k and v inputs are always the same here. if q == k, it becomes self attention, otherwise is cross attention.
    Input dimension order is (batch_size, seq_len, input_dim).
    If only embed_dim is provided, it performs self attention.
    If qdim, kdim, vdim are provided, it performs cross attention.
    All the q, k, v inputs' feature dimensions are first projected to embed_dim, and then perform attention operation.
    '''
    def __init__(self, self_attn, embed_dim, num_heads, qdim=None, kdim=None, vdim=None, dropout=0., bias=True):
        super().__init__()
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        if self_attn:
            self.project_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        else:
            self.project_q = nn.Linear(self.qdim, embed_dim, bias=bias)
            self.project_k = nn.Linear(self.kdim, embed_dim, bias=bias)
            self.project_v = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.scaling = float(self.head_dim) ** -0.5
        self.self_attn = self_attn
        
    def forward(self, query, key=None, value=None, key_padding_mask=None, attn_mask=None):
        '''
        Args:
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask: mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer).
        Shape:
            Inputs:
            - query: :math:`(B, T, E)` where T is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - key: :math:`(B, S, E)`, where S is the source sequence length, B is the batch size, E is
              the embedding dimension.
            - value: :math:`(B, S, E)` where S is the source sequence length, B is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(B, S)`, ByteTensor, where B is the batch size, S is the source sequence length.
              0(False) represents the ture data, 1(True) represents the paded data.
              3-D key_padding_mask with math:`(B, T, S)` is supported now, where T is the target sequence length.
            - attn_mask: :math:`(T, S)` where T is the target sequence length, S is the source sequence length.
        '''
        bsz, tgt_len, _ = query.size()
        if self.self_attn:
            Q, K, V = self.project_qkv(query).chunk(3, dim=-1)
        else:
            Q = self.project_q(query)
            K = self.project_k(key)
            V = self.project_v(value)
        Q = Q * self.scaling
        Q = Q.transpose(0, 1).contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        K = K.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        V = V.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = K.size(1)
        attn_output_weights = torch.bmm(Q, K.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1) if key_padding_mask.dim() == 3 else key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask, float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, V)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        attn_output = self.project_out(attn_output)

        return attn_output, attn_output_weights

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.embed_dim * 3 * self.embed_dim

        flops += self.num_heads * N * (self.embed_dim // self.num_heads) * N   

        flops += self.num_heads * N * (self.embed_dim // self.num_heads) * N   
   
        # x = self.proj(x)
        flops += N * self.embed_dim * self.embed_dim
        return flops

class Key_sparse_Multihead_attention(nn.Module):
    '''
    k and v inputs are always the same here. if q == k, it becomes self attention, otherwise is cross attention.
    Input dimension order is (batch_size, seq_len, input_dim).
    If only embed_dim is provided, it performs self attention.
    If qdim, kdim, vdim are provided, it performs cross attention.
    All the q, k, v inputs' feature dimensions are first projected to embed_dim, and then perform attention operation.
    '''
    def __init__(self, self_attn, embed_dim, num_heads, qdim=None, kdim=None, vdim=None, dropout=0., bias=True, keep_rate=0.50):
        super().__init__()
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.keep_rate = keep_rate

        if self_attn:
            self.project_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        else:
            self.project_q = nn.Linear(self.qdim, embed_dim, bias=bias)
            self.project_k = nn.Linear(self.kdim, embed_dim, bias=bias)
            self.project_v = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.scaling = float(self.head_dim) ** -0.5
        self.self_attn = self_attn
        
    def forward(self, query, key=None, value=None, key_padding_mask=None, attn_mask=None):
        '''
        Args:
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask: mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer).
        Shape:
            Inputs:
            - query: :math:`(B, T, E)` where T is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - key: :math:`(B, S, E)`, where S is the source sequence length, B is the batch size, E is
              the embedding dimension.
            - value: :math:`(B, S, E)` where S is the source sequence length, B is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(B, S)`, ByteTensor, where B is the batch size, S is the source sequence length.
              3-D key_padding_mask with math:`(B, T, S)` is supported now, where T is the target sequence length.
            - attn_mask: :math:`(T, S)` where T is the target sequence length, S is the source sequence length.
        '''
        bsz, tgt_len, _ = query.size()
        if self.self_attn:
            Q, K, V = self.project_qkv(query).chunk(3, dim=-1)
        else:
            Q = self.project_q(query)
            K = self.project_k(key)
            V = self.project_v(value)
        Q = Q * self.scaling
        Q = Q.transpose(0, 1).contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        K = K.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        V = V.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = K.size(1)
        attn_output_weights = torch.bmm(Q, K.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1) if key_padding_mask.dim() == 3 else key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask, float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        
        attn_output_weights = attn_output_weights.transpose(1, 2)  # BH, KT, QT
        K_sum = attn_output_weights.sum(dim=-1)
        zero = torch.tensor(0, device=attn_output_weights.device, dtype=attn_output_weights.dtype)
        K_mask = torch.ones((bsz * self.num_heads, src_len), device=attn_output_weights.device, dtype=attn_output_weights.dtype)
        top_k_K = int(src_len*self.keep_rate)
        values, indices = K_sum.topk(top_k_K, dim=-1, largest=True, sorted=True)
        threshold = values[:,top_k_K-1].unsqueeze(dim=-1)
        K_mask = torch.where(K_sum<threshold, zero, K_mask).unsqueeze(dim=-1)
        attn_output_weights = attn_output_weights * K_mask
        attn_output_weights = attn_output_weights.transpose(1, 2)  # BH, QT, KT

        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, V)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        attn_output = self.project_out(attn_output)

        return attn_output

class TransformerEncoder(nn.Module):
    def __init__(self, self_attn, embed_dim, qdim=None, kdim=None, ffn_embed_dim=2304, num_heads=8, dropout=0.1, attention_dropout=0.1, activation='relu', mode=None):
        super().__init__()
        self.self_attn = self_attn
        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation)

        MHA = Key_sparse_Multihead_attention if mode=='ks' else Multihead_attention
        self.attention = MHA(self_attn, embed_dim, num_heads, qdim, kdim, kdim, attention_dropout)
        self.attention_layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
    
    def add_position(self, x, position=None, mask=None):
        '''
        add position information to the input x
        x: B, T, C
        position: T, C
        mask: B, T
        '''
        if position is None:
            return x
        else:
            B, T = x.shape[:2]
            position = position[:T].unsqueeze(dim=0).repeat(B, 1, 1)  # -> B, T, C
            position = position*((1 - mask.unsqueeze(-1).type_as(x))) if mask is not None else position
            return x + position

    def forward(self, query, key=None, value=None, query_position=None, key_position=None, key_padding_mask=None, attn_mask=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = query
        query = self.add_position(query, query_position)
        key = self.add_position(key, key_position) if not self.self_attn else key
            
        x, _ = self.attention(query, key, value, key_padding_mask, attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.attention_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x

class Transformer(nn.Module):

    def __init__(self, self_attn, num_layers, embed_dim, qdim=None, kdim=None, ffn_embed_dim=2304, num_heads=8, dropout=0.1, attention_dropout=0.1, activation='relu', mode=None):
        super().__init__()
        self.self_attn = self_attn
        self.query_position = create_PositionalEncoding(embed_dim)
        self.query_input_norm = nn.LayerNorm(embed_dim)
        self.key_position = create_PositionalEncoding(kdim) if not self_attn else None
        self.key_input_norm = nn.LayerNorm(kdim) if not self_attn else None

        self.layers = nn.ModuleList([TransformerEncoder(self_attn, embed_dim, qdim, kdim, ffn_embed_dim, num_heads, dropout, attention_dropout, activation, mode) for _ in range(num_layers)])

        self._reset_parameters()     

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.kaiming_uniform_(p)

    def forward(self, query, key=None, key_padding_mask=None, attn_mask=None):
        output = self.query_input_norm(query)
        value = None
        if self.self_attn:
            if key is not None:
                print("you don't need to provide key input in forward function when doing self attention")
        else:
            assert key is not None, 'key input should be provided for doing cross attention.'
            key = self.key_input_norm(key)
            value = key

        for layer in self.layers:
            output = layer(output, key, value, self.query_position, self.key_position, key_padding_mask, attn_mask)

        return output, output


class TransformerLayers2(nn.Module):

    def __init__(self, cfg, qdim=None, kdim=None, dropout=0.1, attention_dropout=0.1, activation='relu', mode=None, **kwargs):
        super().__init__()
        num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        embed_dim = cfg.hidden_size
        kdim = cfg.hidden_size
        ffn_embed_dim = embed_dim // 2
        self_attn = cfg.self_attn
        self.self_attn = self_attn
        self.query_position = create_PositionalEncoding(embed_dim)
        self.query_input_norm = nn.LayerNorm(embed_dim)
        self.key_position = create_PositionalEncoding(kdim) if not self_attn else None
        self.key_input_norm = nn.LayerNorm(kdim) if not self_attn else None

        self.layers = nn.ModuleList([TransformerEncoder(self_attn, embed_dim, qdim, kdim, ffn_embed_dim, num_heads, dropout, attention_dropout, activation, mode) for _ in range(num_layers)])

        self._reset_parameters()     

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.kaiming_uniform_(p)

    def forward(self, query, key=None, attention_mask=None, attention_mask_kv=None, attn_mask=None, **kwargs):
        output = self.query_input_norm(query)
        value = None
        if self.self_attn:
            if key is not None:
                print("you don't need to provide key input in forward function when doing self attention")
        else:
            assert key is not None, 'key input should be provided for doing cross attention.'
            key = self.key_input_norm(key)
            value = key

        for layer in self.layers:
            output = layer(output, key, value, self.query_position, self.key_position, attention_mask_kv, attn_mask)

        return output

def build_transformer(self_attn, num_layers, embed_dim, qdim=None, kdim=None, ffn_embed_dim=2304, num_heads=8, dropout=0.1, attention_dropout=0.1, activation='relu', mode=None):
    '''
    forward arguments in Transformer: 
    query, key=None, key_padding_mask=None, attn_mask=None

    Usage example:
        Input feature: (B, T, C) where T is the sequence length.
        For self attention:
            model = build_transformer(self_attn=True, num_layers=2, embed_dim=64)
            out = model(query=x, key_padding_mask, attn_mask)
        For cross attention:
            model = build_transformer(self_attn=False, num_layers=2, embed_dim=64, kdim=128)
            out = model(query=x_q, key=x_k, key_padding_mask, attn_mask)
    '''
    if qdim is not None:
        assert embed_dim == qdim
    if self_attn:
        if kdim is not None:
            print("you don't need to provide kdim in build_transformer when doing self attention")
    else:
        assert kdim is not None, 'kdim should be provided for cross attention.'

    return Transformer(self_attn, num_layers, embed_dim, qdim, kdim, ffn_embed_dim, num_heads, dropout, attention_dropout, activation, mode)

class single_CCAB(nn.Module):
    def __init__(self, embed_dim, kdim, ffn_embed_dim, num_heads):
        super().__init__()
        self.ks_trans_1 = build_transformer(self_attn=False, num_layers=1, embed_dim=embed_dim, kdim=kdim, ffn_embed_dim=ffn_embed_dim, num_heads=num_heads, mode='ks')
        self.ks_trans_2 = build_transformer(self_attn=True, num_layers=1, embed_dim=embed_dim, ffn_embed_dim=ffn_embed_dim, num_heads=num_heads, mode='ks')
    
    def forward(self, x, k, key_padding_mask):
        x = self.ks_trans_1(query=x, key=k, key_padding_mask=key_padding_mask)
        x = self.ks_trans_2(query=x, key_padding_mask=None)
        return x
    
class CCABs(nn.Module):
    def __init__(self, embed_dim, kdim, ffn_embed_dim, num_layers, num_heads):
        super().__init__()
        self.CCABs_model = nn.ModuleList([single_CCAB(embed_dim, kdim, ffn_embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, k, key_padding_mask):
        residual = x
        for layer in self.CCABs_model:
            x = layer(x, k, key_padding_mask)
        x = x + residual
        return x

def build_CCAB(embed_dim, kdim, ffn_embed_dim, num_layers, num_heads):
    '''
    forward: x, k, key_padding_mask
    '''
    return CCABs(embed_dim, kdim, ffn_embed_dim, num_layers, num_heads)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    # B, L, C = x.shape
    # pad_l = 0
    # pad_r = (window_size - L % window_size) % window_size
    # x = F.pad(x, (0, 0, pad_l, pad_r))

    B, L, C = x.shape
    windows = x.view(-1, window_size, C).contiguous()
    return windows

def window_partition_noreshape(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows, window_size, C)
    """
    # B, L, C = x.shape
    # pad_l = 0
    # pad_r = (window_size - L % window_size) % window_size
    # x = F.pad(x, (0, 0, pad_l, pad_r))
    B, L, C = x.shape
    windows = x.view(B, L // window_size, window_size, C).contiguous()
    return windows

def window_reverse(windows, B):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        B (int): Batch size

    Returns:
        x: (B, L, C)
    """
    x = windows.contiguous().view(B, -1, windows.shape[-1])
    return x

def get_topk_closest_indice(q_windows, k_windows, topk=1):
    # get pair-wise relative position index for each token inside the window
    coords_q = torch.arange(q_windows)
    
    if q_windows != k_windows:
        factor = k_windows // q_windows
        coords_q = coords_q * factor + factor // 2
    else:
        factor = 1 
    
    coords_k = torch.arange(k_windows)

    relative_coords = coords_q[:, None] - coords_k[None, :]  # Wh_q, Wh_k
    
    relative_position_dists = abs(relative_coords)  # Wh_q, Wh_k
    # print(relative_position_dists)
    topk = min(topk, relative_position_dists.shape[1])
    # topk_score_k, topk_index_k = torch.topk(relative_position_dists, topk, dim=1, largest=False)  # Wh_q, Wh_k
    topk_score_k, topk_index_k = torch.topk(-relative_position_dists, topk, dim=1)  # Wh_q, Wh_k
    relative_coord_topk = torch.gather(relative_coords, 1, topk_index_k)  # Wh_q, Wh_k
    # print(q_windows, k_windows, topk_index_k)
    return topk_index_k, relative_coord_topk.contiguous().float(), topk

class WindowAttentionBias(nn.Module):
    def __init__(self, nW, window_size_key, window_size_query):
        super().__init__()
        self.nW = nW
        self.window_size_key = window_size_key
        self.window_size_query = window_size_query

        # self.attn2bias = nn.Sequential(
        #     nn.Conv2d(nW, nW, 1, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(nW, nW, (window_size_key, window_size_query), stride=(window_size_key, window_size_query)))
        
        self.attn2bias = nn.Sequential(
            # nn.Conv3d(1, 1, 1, stride=1),
            nn.Conv3d(1, 1, (1, window_size_key, window_size_query), stride=(1, window_size_key, window_size_query), padding=(0,0,0)),
            nn.ReLU(inplace=True),
            # nn.Conv3d(1, 1, (3, window_size_key, window_size_query), stride=(1, window_size_key, window_size_query), padding=(1,0,0)))
            nn.Conv3d(1, 1, (3, 1, 1), stride=(1, 1, 1), padding=(1,0,0)))
        
        # self.attn2bias_q = nn.Sequential(
        #     nn.Linear(self.window_size_query, self.window_size_query), 
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.window_size_query, 1))
        # self.attn2bias_k = nn.Sequential(
        #     nn.Linear(self.window_size_key, self.window_size_key), 
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.window_size_key, 1))
        # self.attn2bias_nW = nn.Sequential(
        #     nn.Linear(self.nW, self.nW), 
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.nW, self.nW))

    def forward(self, attn_qk):
        # attn_qk = self.attn2bias(attn_qk.view(-1, self.nW, self.window_size_key, self.window_size_query)).view(-1, self.nW) # conv2d
        attn_qk = self.attn2bias(attn_qk.view(-1, 1, self.nW, self.window_size_key, self.window_size_query)).view(-1, self.nW) # conv3d
        # # attn_qk = attn_qk.view(-1, self.nW, self.window_size_key, self.window_size_query).contiguous().mean(-1) # B x nW x wsk
        # attn_qk = self.attn2bias_q(attn_qk.view(-1, self.nW, self.window_size_key, self.window_size_query).contiguous()).squeeze(-1) # B x nW x wsk
        # attn_qk = self.attn2bias_k(attn_qk).squeeze(-1) # B x nW
        # attn_qk = self.attn2bias_nW(attn_qk) # B x nW
        
        # key_bias = torch.trunc(attn_qk).long().unsqueeze(-1)
        # key_bias = torch.clamp(key_bias, -self.window_size_key, self.window_size_key)
        return attn_qk

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, window_size_query, len_q, len_k, focal_factors, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pool_method="none"):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.window_size_query = window_size_query
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads        
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_factors = focal_factors
        
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.topK = self.window_size
 
        # coords_window = torch.arange(self.window_size) - self.window_size // 2
        # self.register_buffer("window_coords", coords_window)
        self.topks = []

        self.nW = len_q // self.window_size_query
        
        # self.attn2bias_all = nn.ModuleList()
        # for focal_factor in focal_factors:
        #     self.attn2bias_all.append(WindowAttentionBias(self.nW, self.window_size//focal_factor, self.window_size_query))

    def forward(self, query, key, key_padding_mask=[None]):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        x = key[0] # B, L, C, [16, 56, 96], len(key)==focal_level

        B, L, C = x.shape
        # nW = L // self.window_size
        # kv = self.kv(x).reshape(B, L, 2, C).permute(2, 0, 1, 3).contiguous()
        # k, v = kv[0], kv[1]  # B, L, C

        q = self.q(query)
        
        # partition q map, B x L x C -> (B*nW) x wsq x C
        q_windows = window_partition(q, self.window_size_query)
        

        k_all = []; v_all = []; topKs = []; topk_rpbs = []
        masks = []
        for l_k in range(len(key)):
            # if l_k == 0:
            #     k_k = k.view(B, -1, self.num_heads, C // self.num_heads)
            #     v_k = v.view(B, -1, self.num_heads, C // self.num_heads)
            # else:
            x_k = key[l_k]
            kv_k = self.kv(x_k).view(B, -1, 2, self.num_heads, C // self.num_heads)
            k_k, v_k = kv_k[:,:,0], kv_k[:,:,1]
            # print(k_k.shape, L, self.window_size, self.nW)

            # k_windows = k_k.view(B*nW, -1, C).contiguous()
            # attn_qk = k_windows @ q_windows.transpose(-2, -1)  # (B*nW) x wsk x wsq
            # key_bias = self.attn2bias_all[l_k](attn_qk)

            # window_coords = getattr(self, "window_coords")
            window_coords = (torch.arange(self.window_size_query) - self.window_size_query // 2).to(k_k.device)
            # window_size_glo = math.floor(self.window_size_glo / (2 ** l_k))
            # rangeL = window_size_glo * self.window_size
            rangeL = k_k.size(1)
            # rangeL = L if l_k == 0 else nW
            # build relative position rangeL            
            topk_closest_indice, topk_closest_coord, topK_updated = get_topk_closest_indice(self.nW, rangeL, self.topK)   # nW x ws
            topk_closest_indice, topk_closest_coord = topk_closest_indice.to(window_coords.device), topk_closest_coord.to(window_coords.device)
            self.topks.append(topK_updated)
            topk_indice_k = topk_closest_indice.repeat(B, 1, 1)    # nW x ws -> B x nW x ws
            # topk_indice_k = topk_indice_k + key_bias
            # topk_indice_k = torch.clamp(topk_indice_k, 0, rangeL-1)
            # print(topk_closest_indice)

            # if l_k > 0:
            #     # scaling the coordinates for pooled windows
            #     topk_closest_coord = topk_closest_coord * self.window_size
            # topk_coords_k = topk_closest_coord.unsqueeze(1) + window_coords[None, :, None]  # nW x ws -> nW x wsq x ws
            
            # topk_rpb_k = self.coord2rpb_all[l_k](topk_coords_k.unsqueeze(-1))   # nW x wsq x ws x nHead
            # topk_rpbs.append(topk_rpb_k)
            
            k_k_selected = torch.gather(k_k, 1, topk_indice_k.view(B, -1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_heads, C // self.num_heads))
            v_k_selected = torch.gather(v_k, 1, topk_indice_k.view(B, -1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_heads, C // self.num_heads))
            # B x (nW*rangeL) x nHead x (C//nHead) -> B x nW x nHead x ws x (C//nHead)
            k_k_selected = k_k_selected.view((B,) + topk_closest_indice.shape + (self.num_heads, C // self.num_heads,)).transpose(2, 3)
            v_k_selected = v_k_selected.view((B,) + topk_closest_indice.shape + (self.num_heads, C // self.num_heads,)).transpose(2, 3)
            # k_k_selected = k_k_selected + key_bias.view(B, -1, 1, 1, 1)
            # v_k_selected = v_k_selected + key_bias.view(B, -1, 1, 1, 1)
            if key_padding_mask[0] != None:
                mask_selected = torch.gather(key_padding_mask[l_k], 1, topk_indice_k.view(B, -1)).view(B, self.nW, self.window_size) # B x nW x ws
                masks.append(mask_selected)
                # print(key_padding_mask[l_k], mask_selected, topk_indice_k.view(B, -1))
            # (B*nW) x nHead x ws x (C//nHead)
            k_all.append(k_k_selected.view(-1, self.num_heads, topk_closest_indice.shape[1], C // self.num_heads))
            v_all.append(v_k_selected.view(-1, self.num_heads, topk_closest_indice.shape[1], C // self.num_heads))
            topKs.append(topk_closest_indice.shape[1])
            # print('x_k:', key[l_k].shape, 'k_k:',k_k.shape, 'k_k_selected:',k_k_selected.shape, topKs[-1])

        k_all = torch.cat(k_all, 2) # (B*nW) x nHead x wscat x (C//nHead)
        v_all = torch.cat(v_all, 2) # (B*nW) x nHead x wscat x (C//nHead)
        
        q_windows = q_windows.view(-1, self.window_size_query, self.num_heads, C // self.num_heads).transpose(1, 2) # (B*nW) x nHead x wsq x (C/nHead)
        q_windows = q_windows * self.scale
        # print(q_windows.shape, k_all.shape)
        attn = (q_windows @ k_all.transpose(-2, -1))  # (B*nW) x nHead x wsq x wscat

        if key_padding_mask[0] is not None:
            masks = torch.cat(masks, 2).view(B*self.nW, 1, 1, -1) # B x nW x 1 x 1 x wscat
            attn = attn.masked_fill(masks, float('-inf'))
            mask_qkv = masks.all(-1, True) # (B*nW) x 1 x 1 x 1
            attn = attn.masked_fill(mask_qkv, 0.)
            # print(masks)
            # print(attn)

        # topk_rpb_cat = torch.cat(topk_rpbs, 2).permute(0, 3, 1, 2).contiguous().unsqueeze(0).repeat(B, 1, 1, 1, 1).view(attn.shape)
        # attn = attn + topk_rpb_cat  # (B*nW) x nHead x wsq x wscat

        attn = self.softmax(attn)        
        # print(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v_all).transpose(1, 2).flatten(2)  # (B*nW) x nHead x wsq x (C//nHead) -> (B*nW) x wsq x C
        # if key_padding_mask[0] is not None:
        #     mask_qkv = masks.all(-1).view(-1, 1, 1) # (B*nW) x 1 x 1
        #     x = x.masked_fill(mask_qkv, 0.)
        x = self.proj(x)
        # print(x)
        x = self.proj_drop(x)   # (B*nW) x wsq x C
        return x, attn

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        for k in range(len(self.focal_factors)):
            flops += self.num_heads * N * (self.dim // self.num_heads) * self.topK  

        #  x = (attn @ v)
        for k in range(len(self.focal_factors)):
            flops += self.num_heads * N * (self.dim // self.num_heads) * self.topK  
   
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class FocalTransformerBlock(nn.Module):
    r""" Focal Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7,  window_size_query=7, len_q=40, len_k=40, self_attn=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="none",  
                 focal_factors=[1], use_layerscale=False, layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_size_query = window_size_query
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_factors = focal_factors
        self.use_layerscale = use_layerscale
        self.len_q = len_q
        self.len_k = len_k

        # self.pool_layers = nn.ModuleList()
        # if self.pool_method != "none":
        #     for focal_factor in self.focal_factors:
        #         if focal_factor==1: continue
        #         if self.pool_method == "fc":
        #             self.pool_layers.append(nn.Linear(focal_factor, 1))
        #             self.pool_layers[-1].weight.data.fill_(1./(focal_factor))
        #             self.pool_layers[-1].bias.data.fill_(0)
        #         elif self.pool_method == "conv":
        #             self.pool_layers.append(nn.Conv1d(dim, dim, kernel_size=focal_factor, stride=focal_factor, groups=dim))

        if self.pool_method != "none":
            self.attn = WindowAttention(
                dim, window_size=self.window_size, 
                window_size_query=self.window_size_query, 
                len_q=len_q, len_k=len_k,
                focal_factors=self.focal_factors, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
                pool_method=pool_method)
        else:
            self.attn = Multihead_attention(
                self_attn=self_attn,
                embed_dim=dim,
                num_heads=num_heads,
                dropout=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        # self.norm1_key = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        # self.norm2_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        # self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        # self.dropout = drop
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # attn_mask = None
        # self.register_buffer("attn_mask", attn_mask)

        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, query, key, attention_mask_kv):
        shortcut = query

        B, L, C = key.shape
        query = self.norm1(query)
        key = self.norm1(key)
        
        if self.pool_method != "none": 
            # if we add coarser granularity and the pool method is not none
            # for k in range(self.focal_level-1):     
            x_windows_all = [key]
            x_window_masks_all = [attention_mask_kv]
            for focal_factor in self.focal_factors:
                if focal_factor==1: continue
                # window_size_glo = math.floor(self.window_size / (2 ** k))   # ws for focal_level k
                # pooled_l = math.ceil(L / self.window_size) * (2 ** k)   # nW for focal_level k
                # L_pool = pooled_l * window_size_glo
                L_pool = math.ceil(L / focal_factor) * focal_factor
                # print(window_size_glo, pooled_l, L_pool, L)
                x_level_k = key
                x_level_mask_k = attention_mask_kv
                # trim or pad x depending on the required size
                if L > L_pool:
                    trim_l = (L - L_pool) // 2
                    trim_r = L - L_pool - trim_l
                    x_level_k = x_level_k[:, trim_l:-trim_r]
                    if attention_mask_kv != None:
                        x_level_mask_k = x_level_mask_k[:, trim_l:-trim_r]
                elif L < L_pool:
                    pad_l = (L_pool - L) // 2
                    pad_r = L_pool - L - pad_l
                    x_level_k = F.pad(x_level_k, (0,0,pad_l,pad_r))
                    if attention_mask_kv != None:
                        x_level_mask_k = F.pad(x_level_mask_k, (pad_l,pad_r))
                
                x_windows_noreshape = window_partition_noreshape(x_level_k.contiguous(), focal_factor) # B, nW, ws, C    
                if self.pool_method == "mean":
                    x_windows_pooled = x_windows_noreshape.mean(-2) # B, nW, C
                elif self.pool_method == "max":
                    x_windows_pooled = x_windows_noreshape.max(-2)[0] # B, nW, C                    
                # elif self.pool_method == "fc":
                #     x_windows_noreshape = x_windows_noreshape.transpose(-1, -2) # B, nW, C, ws
                #     x_windows_pooled = self.pool_layers[k](x_windows_noreshape).flatten(-2) # B, nW, C                      
                # elif self.pool_method == "conv":
                #     x_windows_noreshape = x_windows_noreshape.view(-1, focal_factor, C).transpose(-1, -2).contiguous() # B * nw, C, ws
                #     x_windows_pooled = self.pool_layers[k](x_windows_noreshape).view(B, -1, C) # B, nW, C           
                x_windows_all += [x_windows_pooled]
                
                if attention_mask_kv != None:
                    x_windows_noreshape_mask = x_level_mask_k.view(B, -1, focal_factor) # B, nW, ws    
                    x_windows_mask_pooled = x_windows_noreshape_mask.float().mean(2).round().bool() # B, nW
                    x_window_masks_all += [x_windows_mask_pooled]
        
            attn_windows, attn = self.attn(query, key=x_windows_all, key_padding_mask=x_window_masks_all)  # (B*nW) x ws x C

            attn_windows = attn_windows[:, :self.window_size_query]
            
            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size_query, C)
            x = window_reverse(attn_windows, B)  # B x L x C
        else:
            x, attn = self.attn(query, key=key, value=key, key_padding_mask=attention_mask_kv)  # B x L x C

        # FFN
        # print(shortcut.shape, x.shape, attn_windows.shape, key.shape, query.shape)
        x = shortcut + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        x = x + self.drop_path(self.mlp(self.norm2(x)) if (not self.use_layerscale) else (self.gamma_2 * self.mlp(self.norm2(x))))
        
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = shortcut + x
        # x = self.norm2(x)

        # shortcut = x
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.fc2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = shortcut + x
        # x = self.norm2_2(x)

        return x, attn

    def flops(self):
        flops = 0
        # norm1
        flops += self.dim * (self.len_q + self.len_k)
        
        if self.pool_method != "none":
            nW = self.len_q / self.window_size_query
            flops += nW * self.attn.flops(self.window_size)

            for window_size_glo in self.focal_factors:
                nW_glo = self.len_k // window_size_glo
                # (sub)-window pooling
                flops += nW_glo * self.dim * window_size_glo
        
        else:
            flops += self.attn.flops(self.len_k)

        # mlp
        flops += 2 * self.len_q * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * self.len_q
        return flops


class BasicFocalLayer(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, depth, num_heads, window_size, window_size_query, len_q, len_k, self_attn,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, pool_method="none", 
                 focal_factors=[1], use_layerscale=False, layerscale_value=1e-4):

        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # build blocks
        self.blocks = nn.ModuleList([
            FocalTransformerBlock(dim=dim,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 window_size_query=window_size_query,
                                 len_q=len_q, len_k=len_k,
                                 self_attn=self_attn,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, 
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pool_method=pool_method, 
                                 focal_factors=focal_factors, 
                                 use_layerscale=use_layerscale, 
                                 layerscale_value=layerscale_value)
            for i in range(depth)])

    def forward(self, query, key=None, attention_mask_kv=None):
        for blk in self.blocks:
            query, attn = blk(query, key=key, attention_mask_kv=attention_mask_kv)
        return query, attn

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class FocalTransformerLayers(nn.Module):
    r""" Focal Transformer: Focal Self-attention for Local-Global Interactions in Vision Transformer

    Args:
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        focal_stages (list): Which stages to perform focal attention. Default: [0, 1, 2, 3], means all stages 
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1] 
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value for layer scale. Default: 1e-4 
    """
    def __init__(self, 
                cfg=None,
                len_q=None,
                len_k=None,
                depths=1, 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0.5, 
                attn_drop_rate=0.2,
                drop_path_rate=0.5,
                norm_layer=nn.LayerNorm, 
                ape=False, 
                focal_pool="mean", 
                use_layerscale=False, 
                layerscale_value=1e-4, 
                **kwargs):
        super().__init__()

        # window_size = 2**(focal_levels-1)
        num_layers = cfg.num_hidden_layers if cfg!=None else 5
        num_heads = cfg.num_attention_heads if cfg!=None else 8
        embed_dim = cfg.hidden_size if cfg!=None else 1024
        mlp_ratio = cfg.mlp_ratio if cfg!=None else 1.
        window_size = cfg.window_size if cfg!=None else 4
        focal_factors = cfg.focal_factors if cfg!=None else [1,2,4]
        focal_stages = cfg.focal_stages if cfg!=None else []
        self_attn = cfg.self_attn if cfg!=None else True

        depths=[depths]*num_layers
        num_heads=[num_heads]*num_layers
        self.len_q = len_q
        self.len_k = len_k
        # focal_levels=[focal_levels]*num_layers

        ## padding for key
        self.pad_k = (window_size - len_k % window_size) % window_size
        L_k = len_k + self.pad_k

        ## padding for query
        nw = L_k // window_size
        remainder = len_q % nw
        window_size_query = len_q // nw
        window_size_query = window_size_query+1 if remainder>0 else window_size_query
        self.pad_q = nw * window_size_query - len_q
        L_q = len_q + self.pad_q
        # print(window_size, window_size_query, len_k, len_q, L_k, L_q)

        self.window_size = window_size
        self.window_size_query = window_size_query
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.ape = ape
        self.mlp_ratio = mlp_ratio
        self.focal_stages = focal_stages

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicFocalLayer(dim=self.embed_dim,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               window_size_query=window_size_query,
                               len_q=L_q,
                               len_k=L_k,
                               self_attn=self_attn,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, 
                               qk_scale=qk_scale,
                               drop=drop_rate, 
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer, 
                               pool_method=focal_pool if i_layer in focal_stages else "none",
                               focal_factors=focal_factors,
                            #    focal_level=focal_levels[i_layer], 
                               use_layerscale=use_layerscale, 
                               layerscale_value=layerscale_value)
            self.layers.append(layer)

        self.norm = norm_layer(self.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, query, key=None, attention_mask_kv=None, **kwargs):
        if key==None: key = query
        # print(query.shape, key.shape)

        if not self.focal_stages==[]:
            key = F.pad(key, (0, 0, 0, self.pad_k))
            query = F.pad(query, (0, 0, 0, self.pad_q))
            # print(query.shape, key.shape)

        for layer in self.layers:
            query, attn = layer(query, key=key, attention_mask_kv=attention_mask_kv)
        query = self.norm(query)  # B L C
        if not self.focal_stages==[]:
            L = query.shape[1] - self.pad_q
            query = query[:, :L]
        return query, attn

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.embed_dim * self.len_q
        return flops


if __name__ == '__main__':
    key = torch.rand(1, 40, 1024).cuda()
    query = torch.rand(1, 20, 1024).cuda()
    mask = torch.zeros(key.shape[:-1]).cuda()
    mask[:,-10:] = 1
    mask = mask.bool()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load('../configs/mm-pretraining.yaml')
    model = FocalTransformerLayers(cfg.model.config_cross.rewrite_param_1, 
                            # window_size=8,
                            # focal_factors=[1,2,4], 
                            depths=1,
                            # num_heads=8,
                            drop_path_rate=0.2, 
                            len_q=query.shape[1],
                            len_k=key.shape[1],
                            ).cuda()
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            out = model(query, key, attention_mask_kv=mask)
    print(out.shape)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
