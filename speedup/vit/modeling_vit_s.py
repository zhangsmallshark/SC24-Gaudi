
""" PyTorch ViT Simple model."""

import argparse
import os
import datetime
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import time

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class ViTConfig():
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride


class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings


class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        attention_output = self.dense(context_layer)
        attention_output = self.o_dropout(attention_output)
        outputs = (attention_output,)
        return outputs


class ViTXAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        # self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        # self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        # self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dim_features = self.all_head_size
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

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Project key and queries onto the feature map space
        q_scaled = query_layer * self.softmax_temp
        # norm_q_2 = q_scaled.square().sum(dim=-1, keepdim=True)
        # self.q_offset = -0.5 * norm_q_2 + self.q_offset
        q_scaled = q_scaled @ self.q_feature_map
        q_prime = torch.exp(q_scaled + self.q_offset)
        
        k_scaled = key_layer * self.softmax_temp
        # norm_k_2 = k_scaled.square().sum(dim=-1, keepdim=True)
        # self.k_offset = -0.5 * norm_k_2 + self.k_offset
        k_scaled = k_scaled @ self.k_feature_map
        k_prime = torch.exp(k_scaled + self.k_offset)

        att_normalization = q_prime @ (
            k_prime.transpose(-2, -1) @ torch.ones_like(value_layer)
        )
        att_raw = q_prime @ (k_prime.transpose(-2, -1) @ value_layer)

        # Normalize
        context_layer = att_raw / att_normalization

        attention_output = self.dense(context_layer)
        # attention_output = self.o_dropout(attention_output)
        outputs = (attention_output,)
        return outputs


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig, layer_id=0) -> None:
        super().__init__()
        self.layer_id = layer_id
        # self.attention = ViTAttention(config)
        self.attention = ViTXAttention(config)

        self.dense0 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

        self.dense1 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.o_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.dense0(layer_output)
        layer_output = self.intermediate_act_fn(layer_output)

        # second residual connection is done here
        layer_output = self.dense1(layer_output)
        layer_output = self.o_dropout(layer_output)
        layer_output = layer_output + hidden_states

        outputs = (layer_output,) + outputs
        return outputs


# class ViTPreTrainedModel(PreTrainedModel):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """

#     config_class = ViTConfig
#     base_model_prefix = "vit"
#     main_input_name = "pixel_values"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["ViTEmbeddings", "ViTLayer"]

#     def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
#         """Initialize the weights"""
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
#             # `trunc_normal_cpu` not implemented in `half` issues
#             module.weight.data = nn.init.trunc_normal_(
#                 module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
#             ).to(module.weight.dtype)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         elif isinstance(module, ViTEmbeddings):
#             module.position_embeddings.data = nn.init.trunc_normal_(
#                 module.position_embeddings.data.to(torch.float32),
#                 mean=0.0,
#                 std=self.config.initializer_range,
#             ).to(module.position_embeddings.dtype)

#             module.cls_token.data = nn.init.trunc_normal_(
#                 module.cls_token.data.to(torch.float32),
#                 mean=0.0,
#                 std=self.config.initializer_range,
#             ).to(module.cls_token.dtype)


class ViTModel(nn.Module):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.layers = nn.ModuleList([ViTLayer(config, i) for i in range(config.num_hidden_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
    ):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        head_mask = None

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        hidden_states = embedding_output
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, layer_head_mask)
            hidden_states = layer_outputs[0]

        sequence_output = self.layernorm(hidden_states)
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        outputs = (sequence_output,)

        return outputs


if __name__ == '__main__':

    device = torch.device("hpu")
    torch.cuda.current_device = lambda: None
    torch.cuda.set_device = lambda x: None

    lazy_mode = True
    if lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"] = "1"
    else:
        os.environ["PT_HPU_LAZY_MODE"] = "2"

    # os.environ['TPC_RUNNER'] = str(1)
    # os.environ['HABANA_PROFILE'] = str(1)

    config = ViTConfig(hidden_size=1024, num_hidden_layers=12, num_attention_heads=16, image_size=512, patch_size=8)
    config.qkv_bias = False
    # model = ViTXAttention(config).to(device)
    # model = ViTLayer(config).to(device)
    model = ViTModel(config).to(device)
    model.eval()

    batch_size = 4
    num_classes = 1024
    images = torch.rand(batch_size, 3, config.image_size, config.image_size, dtype=torch.float32).to(device)
    labels = torch.randint(low=0, high=num_classes, size=(batch_size, ), dtype=torch.long).to(device)  
    print(f'image size: {config.image_size}')

    # seq_len = 1024 * 4
    # input = torch.randn([batch_size, seq_len, config.hidden_size], dtype=torch.float, requires_grad=False).to(device)
    # print(f'input size: {input.shape}')

    ites = 2
    # warm up
    with torch.no_grad():
        for i in range(ites):
            output = model(pixel_values=images)
            htcore.mark_step()

    ht.hpu.synchronize()
    start=time.time()
    with torch.no_grad():
        for i in range(ites):
            # output = model(input)
            output = model(pixel_values=images)
            htcore.mark_step()

    ht.hpu.synchronize()
    end=time.time()
    run_time = (end-start)/ites * 1000.0
    print(f'run time {run_time} ms')
    print('VIT finish! ')
