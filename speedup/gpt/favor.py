# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple
from enum import Enum, auto

import torch
import torch.nn as nn


class NormDistribution(Enum):
    Xi = auto()
    Uniform = auto()


class SoftMaxPositiveEstimators(torch.nn.Module):
    def __init__(
        self,
        dim_features: int,
        iter_before_redraw: Optional[int],
        normalize_inputs: bool = False,
        epsilon: float = 1e-6,
        softmax_temp: float = -1,
    ):
        super().__init__()
        self.dim_features = dim_features
        self.dim_feature_map = dim_features
        self.iter_before_redraw = iter_before_redraw
        self.features: Optional[torch.Tensor] = None
        self.epsilon = epsilon
        self.normalize_inputs = normalize_inputs
        self._iter_counter = 0
        self.softmax_temp = softmax_temp

        # Handle the scaling from all kernels by √m.
        # This normalizes for all the feature maps involved
        self.h_scale = math.log(math.sqrt(self.dim_features))

    def _get_feature_map(self, dim_input: int, dim_features: int, device: torch.device):
        raise NotImplementedError()

    def pre_scale(self, x: torch.Tensor) -> torch.Tensor:
        # Re-draw counting logic
        if (
            (
                self.iter_before_redraw is not None
                and self._iter_counter > self.iter_before_redraw
            )
            or self.features is None
            or self.features.device != x.device
        ):
            # The feature map is actually using half the dimension, we'll concatenate + and - features
            self._iter_counter = 1
            self.features = self._get_feature_map(
                x.shape[-1], self.dim_feature_map, x.device
            )

        features = self.features
        assert features is not None

        if features.dtype != x.dtype:
            self.features = features.to(x.dtype)

        self._iter_counter += 1

        # Normalization / softmax
        if self.softmax_temp < 0:
            # A = exp(QK.t/√d), so each input will be scaled by √√d
            self.softmax_temp = x.shape[-1] ** -0.25

        x_scaled = x * self.softmax_temp

        # Compute the scaling factors in logspace, applied from within the exponential
        # - dimnish possible exponential overflow
        # - remove a multiply across the batch, replace by an addition
        norm_x_2 = torch.einsum("...d,...d->...", x_scaled, x_scaled).unsqueeze(-1)
        self.offset = -0.5 * norm_x_2 - self.h_scale + self.epsilon

        if self.normalize_inputs:
            # L0 normalize the exponential term, can be useful for numerical stability
            # This ensures that features +- offset is below 1
            self.offset -= norm_x_2.max(1, keepdim=True)[0]

        # Return the scaled inputs, the rest depends on the kernel being used
        return x_scaled

    @staticmethod
    @torch.no_grad()
    def _get_random_ortho_matrix(
        blocks: int,
        dim: int,
        device: torch.device,
        norm_distribution: NormDistribution = NormDistribution.Uniform,
    ) -> torch.Tensor:
        r"""
        Generate a random matrix whose rows are exactly orthonormal

        "How to generate random matrices from the classical compact groups", Mezzadri, 2007
        https://arxiv.org/pdf/math-ph/0609050v2.pdf

        .. note: the typical qr decomposition does not give uniform results, qr decomposition is not
        unique and the qr decomposition routines are biased towards numerical stability. See the above
        paper for more information.

        .. note: this does not follow the original implementation from the Performers authors.
        see docs/assets/kde plots to visualize the impact of using the R signs to correct Q
        """

        H = torch.randn((blocks, dim, dim), device=device, requires_grad=False)
        return H

        # # Randomly scale the norms of the features, Xi distributed
        # if norm_distribution == NormDistribution.Xi:
        #     # NOTE: This averages to sqrt(d)
        #     norms = torch.sqrt(torch.einsum("...d,...d->...", H, H))

        # Q, R = torch.linalg.qr(H)
        # Q = torch.diag_embed(torch.sign(torch.diagonal(R, dim1=1, dim2=2))) @ Q

        # # Normalize if need be. Uniform NormDistribution does nothing, Q is already orthonormal
        # if norm_distribution == NormDistribution.Xi:
        #     return torch.diag_embed(norms) @ Q

        # return Q


class SMOrf(SoftMaxPositiveEstimators):
    """
    "Positive random orthogonal features" softmax estimator,
    SM_ort^m+, as proposed in the Performers_ paper, Lemma 1.

    _Performers: "Rethinking attention with performers." K. Choromanski et al. (2020).
    https://arxiv.org/pdf/2009.14794v1.pdf
    """

    @torch.no_grad()
    def _get_feature_map(self, dim_input: int, dim_features: int, device: torch.device):
        """
        Generate the projection matrix onto the random features

        .. note: The heads dimension needs to be taken into account, hence the per-block random matrix
        and not uniformally random.
        """

        # Get per block random unitary matrices.
        # We need enough of them to project the whole input dimension, regardless of the
        # requested dimension of the features
        features = self._get_random_ortho_matrix(
            math.ceil(dim_input / dim_features),
            dim_features,
            norm_distribution=NormDistribution.Xi,
            device=device,
        )

        return features.flatten(0, 1)[:dim_input]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax-dimension related scaling, shared for all kernels
        x_scaled = super().pre_scale(x)
        assert self.features is not None

        # Project onto the random feature map.
        x_scaled = x_scaled @ self.features
        return x_scaled
        # return torch.exp(x_scaled + self.offset)


class FavorAttention(nn.Module):
    def __init__(
        self,
        causal: bool = False,
        dropout: float = 0.0,
        dim_features: Optional[int] = None,
        dim_head: Optional[int] = None,
        iter_before_redraw: Optional[int] = None,
        # feature_map_type: FeatureMapType = FeatureMapType.SMReg,
        normalize_inputs: bool = False,
        *_,
        **__,
    ):
        r"""
        Kernelized attention, as proposed in Performers_
        ("Rethinking attention with performers." K. Choromanski et al. (2020).).

        FAVOR stands for "Fast Attention Via positive Orthogonal Random features"

        Args:
            dropout (float): the probability of an output to be randomly dropped at training time
            dim_features (int): the dimension of the random features space
            iter_before_redraw (int): the number of steps (forward calls) before a redraw of the features
            feature_map_type (FeatureMapType): the type of feature map being used,
            for instance orthogonal random features.

        .. _Performers: https://arxiv.org/pdf/2009.14794v1.pdf
        """
        super().__init__()

        self.causal = causal
        self.iter_before_redraw = (
            (2 * iter_before_redraw)
            if iter_before_redraw is not None
            else iter_before_redraw
        )  # This will be used for both key and query
        self.normalize_inputs = normalize_inputs
        # self.feature_map_type = feature_map_type
        self.attn_drop = nn.Dropout(dropout, inplace=True)

        # Setup dimension-dependent variables
        # Reasonable dimension default
        if dim_features is None:
            assert dim_head is not None, "dim_features or dim_head needs to be passed"
            self.dim_features = math.ceil(dim_head * (1 + math.log2(dim_head)))
            self.dim_features = 2 * (
                self.dim_features // 2
            )  # needs to be even for some variants
            print(
                f"FAVOR: Automatically setting the random mapping dimension to {self.dim_features} from {dim_head}"
            )
        else:
            self.dim_features = dim_features

        # feature_map_constructor = {
        #     FeatureMapType.SMHyp: SMHyperbolic,
        #     FeatureMapType.SMReg: SMReg,
        #     FeatureMapType.SMOrf: SMOrf,
        # }[self.feature_map_type]
        feature_map_constructor = SMOrf

        feature_settings = {
            "dim_features": self.dim_features,
            "iter_before_redraw": self.iter_before_redraw,
            "normalize_inputs": self.normalize_inputs,
        }

        self.feature_map = feature_map_constructor(**feature_settings)  # type: ignore

        # Properties specific to this attention mechanism
        self.supports_attention_mask = False
        self.supports_key_padding_mask = False

    @staticmethod
    def _maybe_promote(x: torch.Tensor) -> torch.Tensor:
        # Only promote fp16 buffers, bfloat16 would be fine for instance
        return x.float() if x.dtype == torch.float16 else x

    # @staticmethod
    # def _causal_attention(
    #     k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # Algorithm 1 in the paper
    #     ref_v = torch.ones_like(v.unsqueeze(2))  # BATCH x SEQ x 1 x EMB
    #     Gps = k_prime.unsqueeze(3) * v.unsqueeze(2)
    #     Grenorm = k_prime.unsqueeze(3) * ref_v

    #     # Consolidate against the feature dimension
    #     att_raw = torch.einsum("bcfe,bcf->bce", Gps, q_prime)
    #     att_norm = torch.einsum("bcfe,bcf->bce", Grenorm, q_prime)

    #     # Cumulative sum over the sequence
    #     att_raw = att_raw.cumsum(2)
    #     att_norm = att_norm.cumsum(2)

    #     return att_raw, att_norm

    @staticmethod
    def _causal_attention(
        k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor
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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *_,
        **__,
    ):
        # Project key and queries onto the feature map space
        k_prime = self.feature_map(k)
        q_prime = self.feature_map(q)

        # The softmax kernel approximation for Favor will easily overflow
        # Force the computations here to stay in fp32 for numerical stability
        # Note that the dimensions are vastly reduced when compared to scaled_dot_product
        k_prime = self._maybe_promote(k_prime)
        q_prime = self._maybe_promote(q_prime)
        v = self._maybe_promote(v)

        if not self.causal:
            att_normalization = q_prime @ (
                k_prime.transpose(-2, -1) @ torch.ones_like(v)
            )
            att_raw = q_prime @ (k_prime.transpose(-2, -1) @ v)
        else:
            # Actually compute attention
            att_raw, att_normalization = self._causal_attention(k_prime, q_prime, v)

        # Normalize
        att = att_raw / att_normalization

        if self.attn_drop is not None:
            att = self.attn_drop(att)

        return att
