
""" PyTorch GPT Simple model."""


import os
import math
from typing import Optional, Tuple, Union
import time

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


# export GC_KERNEL_PATH=/home/chengming.zhang/exps/fc_nn/custom_mm1/Habana_Custom_Kernel/build/src/libcustom_tpc_perf_lib.so:/usr/lib/habanalabs/libtpc_kernels.so

# custom_outer_op_lib_path = "/home/chengming.zhang/exps/fc_nn/custom_outer/build/lib.linux-x86_64-3.8/hpu_custom_outer.cpython-38-x86_64-linux-gnu.so"
# torch.ops.load_library(custom_outer_op_lib_path)

class GPTConfig():
    def __init__(
        self,
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_layers=24,
        attention_types=[[["global", "local"], 12]],
        num_heads=16,
        intermediate_size=None,
        window_size=256,
        activation_function="gelu_new",
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.window_size = window_size
        self.activation_function = activation_function
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.attention_types = attention_types
        self.attention_layers = self.expand_attention_types_params(attention_types)

    @staticmethod
    def expand_attention_types_params(attention_types):
        attentions = []
        for item in attention_types:
            for _ in range(item[1]):
                attentions.extend(item[0])
        return attentions


class GPTSelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()

        self.layer_id = layer_id
        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        query = query * self.scale
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, )
        if output_attentions:
            outputs += (attn_weights,)
        return outputs 


class GPTWAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()

        self.layer_id = layer_id

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.window_size = 128

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        query = query * self.scale
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights, attn_raw = torch.tensor_split(attn_weights, (self.window_size,), dim=-1)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = torch.cat((attn_weights, attn_raw), dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, )
        if output_attentions:
            outputs += (attn_weights,)
        return outputs 


# class GPTXAttention(nn.Module):
#     """
#     GPT Favor attention module. 
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         self.resid_dropout = nn.Dropout(float(config.resid_dropout))
#         self.is_causal = True

#         self.embed_dim = config.hidden_size
#         self.num_heads = config.num_heads
#         self.head_dim = self.embed_dim // self.num_heads
#         if self.head_dim * self.num_heads != self.embed_dim:
#             raise ValueError(
#                 f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
#                 f" {self.num_heads})."
#             )

#         self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
#         self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
#         self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
#         self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

#         self.num_sparse_heads = 2
#         self.num_linear_heads = self.num_heads - self.num_sparse_heads

#         self.dim_features = self.num_linear_heads * self.head_dim #self.embed_dim
#         self.softmax_temp = self.dim_features ** -0.25
#         self.h_scale = math.log(math.sqrt(self.dim_features))
#         self.epsilon = 1e-6
#         self.q_offset = -self.h_scale + self.epsilon
#         self.k_offset = -self.h_scale + self.epsilon
#         self.q_feature_map = self.get_feature_map(self.dim_features, self.dim_features)
#         self.k_feature_map = self.get_feature_map(self.dim_features, self.dim_features)

#     def get_feature_map(self, dim_input: int, dim_features: int):
#         H = torch.randn((math.ceil(dim_input / dim_features), dim_features, dim_features), requires_grad=False)
#         features = H.flatten(0, 1)[:dim_input]
#         return features

#     def _split_heads(self, tensor, num_heads, attn_head_size):
#         """
#         Splits hidden_size dim into attn_head_size and num_heads
#         """
#         new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
#         tensor = tensor.view(new_shape)
#         return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

#     def _merge_heads(self, tensor, num_heads, attn_head_size):
#         """
#         Merges attn_head_size dim and num_attn_heads dim into hidden_size
#         """
#         tensor = tensor.permute(0, 2, 1, 3).contiguous()
#         new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
#         return tensor.view(new_shape)

#     def causal_attention0(
#         self, k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Algorithm 1 in the paper
#         ref_v = torch.ones_like(v.unsqueeze(2))  # BATCH x SEQ x 1 x EMB
#         Gps = k_prime.unsqueeze(3) @ v.unsqueeze(2)
#         Grenorm = k_prime.unsqueeze(3) @ ref_v

#         # Consolidate against the feature dimension
#         att_raw = q_prime.unsqueeze(2) @ Gps
#         att_norm = q_prime.unsqueeze(2) @ Grenorm

#         # Cumulative sum over the sequence
#         att_raw = att_raw.cumsum(1)
#         att_norm = att_norm.cumsum(1)
#         att_raw = att_raw.squeeze(2)
#         att_norm = att_norm.squeeze(2)
#         return att_raw, att_norm

#     def causal_attention1(
#         self, k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Algorithm 1 in the paper
#         ref_v = torch.ones_like(v.unsqueeze(3))  # BATCH x HEAD x SEQ x 1 x EMB

#         # Gps = torch.ops.custom_op.custom_outer(k_prime, v)
#         # Gps = Gps.view(Gps.size()[:-1]+(self.head_dim, self.head_dim))
#         Gps = k_prime.unsqueeze(4) @ v.unsqueeze(3)

#         Grenorm = k_prime.unsqueeze(4) @ ref_v

#         # Consolidate against the feature dimension
#         att_raw = q_prime.unsqueeze(3) @ Gps
#         att_norm = q_prime.unsqueeze(3) @ Grenorm

#         # Cumulative sum over the sequence
#         att_raw = att_raw.cumsum(2)
#         att_norm = att_norm.cumsum(2)

#         att_raw = att_raw.squeeze(3)
#         att_norm = att_norm.squeeze(3)
#         return att_raw, att_norm

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         output_attentions=False,
#     ):
#         query = self.q_proj(hidden_states)
#         key = self.k_proj(hidden_states)
#         value = self.v_proj(hidden_states)

#         q_sparse, q_linear = torch.tensor_split(query, (self.num_sparse_heads*self.head_dim,), dim=-1)
#         k_sparse, k_linear = torch.tensor_split(key, (self.num_sparse_heads*self.head_dim,), dim=-1)
#         v_sparse, v_linear = torch.tensor_split(value, (self.num_sparse_heads*self.head_dim,), dim=-1)

#         # Project key and queries onto the feature map space
#         q_scaled = q_linear * self.softmax_temp
#         norm_q_2 = q_scaled.square().sum(dim=-1, keepdim=True)
#         self.q_offset = -0.5 * norm_q_2 + self.q_offset
#         q_scaled = q_scaled @ self.q_feature_map
#         q_prime = torch.exp(q_scaled + self.q_offset)
        
#         k_scaled = k_linear * self.softmax_temp
#         norm_k_2 = k_scaled.square().sum(dim=-1, keepdim=True)
#         self.k_offset = -0.5 * norm_k_2 + self.k_offset
#         k_scaled = k_scaled @ self.k_feature_map
#         k_prime = torch.exp(k_scaled + self.k_offset)

#         q_prime = self._split_heads(q_prime, self.num_linear_heads, self.head_dim)
#         k_prime = self._split_heads(k_prime, self.num_linear_heads, self.head_dim)
#         v_linear = self._split_heads(v_linear, self.num_linear_heads, self.head_dim)

#         att_raw, att_normalization = self.causal_attention1(k_prime, q_prime, v_linear)

#         q_prime = self._split_heads(q_prime, self.num_linear_heads, self.head_dim)
#         k_prime = self._split_heads(k_prime, self.num_linear_heads, self.head_dim)
#         v_linear = self._split_heads(v_linear, self.num_linear_heads, self.head_dim)


#         q_sparse = self._split_heads(q_sparse, self.num_sparse_heads, self.head_dim)
#         k_sparse = self._split_heads(k_sparse, self.num_sparse_heads, self.head_dim)
#         v_sparse = self._split_heads(v_sparse, self.num_sparse_heads, self.head_dim)

#         # query = query * self.scale
#         attn_weights = torch.matmul(q_sparse, k_sparse.transpose(-1, -2))
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)
#         # attn_weights = self.attn_dropout(attn_weights)
#         attn_sparse = torch.matmul(attn_weights, v_sparse)
#         print(f'attn_sparse {attn_sparse.shape}')

#         # Normalize
#         attn_linear = att_raw / att_normalization
#         print(f'attn_linear {attn_linear.shape}')

#         attn_output = torch.cat((attn_sparse, attn_linear), dim=1)

#         attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

#         attn_output = self.out_proj(attn_output)
#         attn_output = self.resid_dropout(attn_output)

#         outputs = (attn_output,)
#         return outputs 
    

class GPTXAttention(nn.Module):
    """
    GPT Favor attention module. 
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.resid_dropout = nn.Dropout(float(config.resid_dropout))
        self.is_causal = True

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.dim_features = self.embed_dim
        self.softmax_temp = self.dim_features ** -0.25
        self.h_scale = math.log(math.sqrt(self.dim_features))
        self.epsilon = 1e-6
        self.q_offset = -self.h_scale + self.epsilon
        self.k_offset = -self.h_scale + self.epsilon
        self.q_feature_map = self.get_feature_map(self.dim_features, self.dim_features)
        self.k_feature_map = self.get_feature_map(self.dim_features, self.dim_features)

    def get_feature_map(self, dim_input: int, dim_features: int):
        H = torch.randn((math.ceil(dim_input / dim_features), dim_features, dim_features), requires_grad=False)
        features = H.flatten(0, 1)[:dim_input]
        return features

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def causal_attention0(
        self, k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Algorithm 1 in the paper
        ref_v = torch.ones_like(v.unsqueeze(2))  # BATCH x SEQ x 1 x EMB
        Gps = k_prime.unsqueeze(3) @ v.unsqueeze(2)
        Grenorm = k_prime.unsqueeze(3) @ ref_v

        # Consolidate against the feature dimension
        att_raw = q_prime.unsqueeze(2) @ Gps
        att_norm = q_prime.unsqueeze(2) @ Grenorm

        # Cumulative sum over the sequence
        att_raw = att_raw.cumsum(1)
        att_norm = att_norm.cumsum(1)
        att_raw = att_raw.squeeze(2)
        att_norm = att_norm.squeeze(2)
        return att_raw, att_norm

    def causal_attention1(
        self, k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Algorithm 1 in the paper
        # ref_v = torch.ones_like(v.unsqueeze(3))  # BATCH x HEAD x SEQ x 1 x EMB

        # Gps = torch.ops.custom_op.custom_outer(k_prime, v)
        # Gps = Gps.view(Gps.size()[:-1]+(self.head_dim, self.head_dim))

        Gps = k_prime.unsqueeze(4) @ v.unsqueeze(3)
        # Grenorm = k_prime.unsqueeze(4) @ ref_v

        # Consolidate against the feature dimension
        att_raw = q_prime.unsqueeze(3) @ Gps
        att_norm = k_prime
        # att_norm = q_prime.unsqueeze(3) @ Grenorm

        # Cumulative sum over the sequence
        att_raw = att_raw.cumsum(2)
        # att_norm = att_norm.cumsum(2)

        att_raw = att_raw.squeeze(3)
        # att_norm = att_norm.squeeze(3)

        return att_raw, att_norm

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Project key and queries onto the feature map space
        # q_scaled = query * self.softmax_temp
        q_scaled = query
        # norm_q_2 = q_scaled.square().sum(dim=-1, keepdim=True)
        # self.q_offset = -0.5 * norm_q_2 + self.q_offset
        q_scaled = q_scaled @ self.q_feature_map
        q_prime = torch.exp(q_scaled + self.q_offset)
        
        # k_scaled = key * self.softmax_temp
        k_scaled = key
        # norm_k_2 = k_scaled.square().sum(dim=-1, keepdim=True)
        # self.k_offset = -0.5 * norm_k_2 + self.k_offset
        k_scaled = k_scaled @ self.k_feature_map
        k_prime = torch.exp(k_scaled + self.k_offset)

        q_prime = self._split_heads(q_prime, self.num_heads, self.head_dim)
        k_prime = self._split_heads(k_prime, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        att_raw, att_normalization = self.causal_attention1(k_prime, q_prime, value)
        attn_output = att_raw / att_normalization

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        # attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output,)
        return outputs 


class GPTMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = F.gelu
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # self.attn = GPTSelfAttention(config, layer_id)
        self.attn = GPTXAttention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        outputs = (hidden_states,)
        return outputs  # hidden_states


# class GPTNeoPreTrainedModel(nn.Module):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """

#     config_class = GPTNeoConfig
#     base_model_prefix = "transformer"

#     def __init__(self, config, *inputs, **kwargs):
#         super().__init__()
#         self.config = config
#         self.dtype = torch.float

#     def _init_weights(self, module):
#         """Initialize the weights."""
#         if isinstance(module, (nn.Linear,)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def post_init(self):
#         """
#         A method executed at the end of each Transformer model initialization, to execute code that needs the model's
#         modules properly initialized (such as weight initialization).
#         """
#         self.apply(self._init_weights)


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(float(config.embed_dropout))
        self.h = nn.ModuleList([GPTBlock(config, layer_id=i) for i in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        # if attention_mask is not None:
        #     if batch_size <= 0:
        #         raise ValueError("batch_size has to be defined and > 0")
        #     attention_mask = attention_mask.view(batch_size, -1)
        #     # We create a 3D attention mask from a 2D tensor mask.
        #     # Sizes are [batch_size, 1, 1, to_seq_length]
        #     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        #     # this attention mask is more simple than the triangular masking of causal attention
        #     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #     attention_mask = attention_mask[:, None, None, :]

        #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        #     # masked positions, this operation will create a tensor which is 0.0 for
        #     # positions we want to attend and the dtype's smallest value for masked positions.
        #     # Since we are adding it to the raw scores before the softmax, this is
        #     # effectively the same as removing these entirely.
        #     attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        # head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
            )

            hidden_states = outputs[0]

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return hidden_states


# class GPTNeoForCausalLM(GPTNeoPreTrainedModel):
#     _tied_weights_keys = ["lm_head.weight"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.transformer = GPTModel(config)
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
#         token_type_ids = kwargs.get("token_type_ids", None)
#         # only last token for inputs_ids if past is defined in kwargs
#         if past_key_values:
#             input_ids = input_ids[:, -1].unsqueeze(-1)
#             if token_type_ids is not None:
#                 token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

#         attention_mask = kwargs.get("attention_mask", None)
#         position_ids = kwargs.get("position_ids", None)

#         if attention_mask is not None and position_ids is None:
#             # create position_ids on the fly for batch generation
#             position_ids = attention_mask.long().cumsum(-1) - 1
#             position_ids.masked_fill_(attention_mask == 0, 1)
#             if past_key_values:
#                 position_ids = position_ids[:, -1].unsqueeze(-1)

#         return {
#             "input_ids": input_ids,
#             "past_key_values": past_key_values,
#             "use_cache": kwargs.get("use_cache"),
#             "position_ids": position_ids,
#             "attention_mask": attention_mask,
#             "token_type_ids": token_type_ids,
#         }

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
#             `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
#             are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]

#         lm_logits = self.lm_head(hidden_states)

#         loss = None
#         if labels is not None:
#             # move labels to correct device to enable model parallelism
#             labels = labels.to(lm_logits.device)
#             # Compute loss in fp32 to match with mesh-tf version
#             # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
#             lm_logits = lm_logits.to(torch.float32)

#             # Shift so that tokens < n predict n
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#             lm_logits = lm_logits.to(hidden_states.dtype)
#             loss = loss.to(hidden_states.dtype)

#         if not return_dict:
#             output = (lm_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )

#     @staticmethod
#     def _reorder_cache(
#         past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
#     ) -> Tuple[Tuple[torch.Tensor]]:
#         """
#         This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
#         [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
#         beam_idx at every generation step.
#         """
#         return tuple(
#             tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
#             for layer_past in past_key_values
#         )




if __name__ == '__main__':

    device = torch.device("hpu")
    torch.cuda.current_device = lambda: None
    torch.cuda.set_device = lambda x: None

    lazy_mode = True
    if lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"] = "1"
    else:
        os.environ["PT_HPU_LAZY_MODE"] = "2"

    # os.environ['LOG_LEVEL_ALL'] = str(1)
    # os.environ['TPC_RUNNER'] = str(1)
    # os.environ['HABANA_PROFILE'] = str(1)
    
    seq_len = 1024 * 6
    config = GPTConfig(max_position_embeddings=seq_len, hidden_size=1024, num_layers=12, num_heads=16, attention_dropout=0, resid_dropout=0)
    # model = GPTSelfAttention(config)
    # model = GPTWAttention(config)
    # model = GPTXAttention(config)
    # model = GPTBlock(config, 0)
    model = GPTModel(config)
    model = model.to(device)
    model.eval()

    batch_size = 4
    input = torch.randn([batch_size, config.max_position_embeddings, config.hidden_size], dtype=torch.float, requires_grad=False).to(device)
    input_ids = torch.randint(low=0, high=20000, size=(batch_size, config.max_position_embeddings), dtype=torch.long, requires_grad=False).to(device)
    position_ids = torch.stack([torch.arange(0, config.max_position_embeddings, dtype=torch.long, requires_grad=False) for _ in range(batch_size)])
    attention_mask = torch.tril(torch.ones(1, config.max_position_embeddings, config.max_position_embeddings, dtype=torch.float, requires_grad=False), diagonal=1).to(device)
    print(f'input shape: {input_ids.shape}')

    # in0 = torch.randn(batch_size, config.num_heads, config.max_position_embeddings, 64, dtype=torch.float32, requires_grad=False).to(device)
    # in1 = torch.randn(batch_size, config.num_heads, config.max_position_embeddings, 64, dtype=torch.float32, requires_grad=False).to(device)
    # print(f'input shape: {in0.shape}; num: {batch_size * config.num_heads * config.max_position_embeddings}')

    ites = 2
    # warm up
    with torch.no_grad():
        for i in range(ites):
            # output = model(input, attention_mask=attention_mask)
            # out1 = in0.unsqueeze(4) @ in1.unsqueeze(3)
            output = model(input_ids=input_ids, past_key_values=None, attention_mask=attention_mask, token_type_ids=None, position_ids=None)
            htcore.mark_step()

    ht.hpu.synchronize()
    start=time.time()
    with torch.no_grad():
        for i in range(ites):
            # output = model(input, attention_mask=attention_mask)
            # out1 = in0.unsqueeze(4) @ in1.unsqueeze(3)
            output = model(input_ids=input_ids, past_key_values=None, attention_mask=attention_mask, token_type_ids=None, position_ids=None)
            # print(output)
            htcore.mark_step()

    ht.hpu.synchronize()
    end=time.time()
    run_time = (end-start)/ites * 1000.0
    print(f'run time {run_time} ms')
    print('finish! ')
