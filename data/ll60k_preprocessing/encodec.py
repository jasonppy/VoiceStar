# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Compression models or wrapper around existing models.
Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import typing as tp

import numpy as np
import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

import logging
import warnings
from einops import rearrange, repeat
import omegaconf
# import flashy

CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_group_norm'])

def dict_from_config(cfg: omegaconf.DictConfig) -> dict:
    """Convenience function to map an omegaconf configuration to a dictionary.

    Args:
        cfg (omegaconf.DictConfig): Original configuration to map to dict.
    Returns:
        dict: Config as dictionary object.
    """
    dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(dct, dict)
    return dct

@dataclass
class QuantizedResult:
    x: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)

class BaseQuantizer(nn.Module):
    """Base class for quantizers.
    """

    def forward(self, x: torch.Tensor, frame_rate: int) -> QuantizedResult:
        """
        Given input tensor x, returns first the quantized (or approximately quantized)
        representation along with quantized codes, bandwidth, and any penalty term for the loss.
        Finally, this returns a dict of metrics to update logging etc.
        Frame rate must be passed so that the bandwidth is properly computed.
        """
        raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth."""
        raise NotImplementedError()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        raise NotImplementedError()

    @property
    def total_codebooks(self):
        """Total number of codebooks."""
        raise NotImplementedError()

    @property
    def num_codebooks(self):
        """Number of active codebooks."""
        raise NotImplementedError()

    def set_num_codebooks(self, n: int):
        """Set the number of active codebooks."""
        raise NotImplementedError()

class CompressionModel(ABC, nn.Module):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> QuantizedResult:
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """See `EncodecModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """See `EncodecModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        ...

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def cardinality(self) -> int:
        ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int:
        ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...

def apply_parametrization_norm(module: nn.Module, norm: str = 'none'):
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs):
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class NormConvTranspose2d(nn.Module):
    """Wrapper around ConvTranspose2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal=False, norm=norm, **norm_kwargs)

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class StreamableConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect'):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn("StreamableConv1d has been initialized with stride > 1 and dilation > 1"
                          f" (kernel_size={kernel_size} stride={stride}, dilation={dilation}).")
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, groups=groups, bias=bias, causal=causal,
                               norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class StreamableConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, causal: bool = False,
                 norm: str = 'none', trim_right_ratio: float = 1.,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                          causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class StreamableLSTM(nn.Module):
    """LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamableConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamableConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                             causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """SEANet encoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid." \
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."

        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            StreamableConv1d(channels, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= i + 2 else norm
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      norm=block_norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,
                                      causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamableConv1d(mult * n_filters, mult * n_filters * 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=block_norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            StreamableConv1d(mult * n_filters, dimension, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0, trim_right_ratio: float = 1.0):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid." \
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamableConv1d(dimension, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            # Add upsampling layers
            model += [
                act(**activation_params),
                StreamableConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                          kernel_size=ratio * 2, stride=ratio,
                                          norm=block_norm, norm_kwargs=norm_params,
                                          causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=block_norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StreamableConv1d(n_filters, channels, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y


def exists(val: tp.Optional[tp.Any]) -> bool:
    return val is not None


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if exists(val) else d


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = einsum("i d, j d -> i j", normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.

    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.8,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        flashy.distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        flashy.distrib.broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        raise NotImplementedError()
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)
        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.

    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int):
        channels_last (bool): Channels are the last dimension in the input tensors.
        commitment_weight (float): Weight for commitment loss.
        orthogonal_reg_weight (float): Orthogonal regularization weights.
        orthogonal_reg_active_codes_only (bool): Apply orthogonal regularization only on active codes.
        orthogonal_reg_max_codes (optional int): Maximum number of codes to consider
            for orthogonal regularization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.8,
        epsilon: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        channels_last: bool = False,
        commitment_weight: float = 1.,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = False,
        orthogonal_reg_max_codes: tp.Optional[int] = None,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

        self.channels_last = channels_last

    @property
    def codebook(self):
        return self._codebook.embed

    @property
    def inited(self):
        return self._codebook.inited

    def _preprocess(self, x):
        if not self.channels_last:
            x = rearrange(x, "b d n -> b n d")
        return x

    def _postprocess(self, quantize):
        if not self.channels_last:
            quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def encode(self, x):
        x = self._preprocess(x)
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = self._postprocess(quantize)
        return quantize

    def forward(self, x):
        device = x.device
        x = self._preprocess(x)

        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        quantize = self.project_out(quantize)
        quantize = self._postprocess(quantize)

        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        codebook_size = kwargs.pop('codebook_size', None)
        if codebook_size is None:
            raise ValueError("codebook_size must be provided in kwargs")
        if type(codebook_size) != list:
            codebook_size = [codebook_size] * num_quantizers
        self.layers = nn.ModuleList(
            [VectorQuantization(codebook_size=cur_codebook_size, **kwargs) for _,cur_codebook_size in zip(range(num_quantizers), codebook_size)]
        )
            

        # self.layers = nn.ModuleList(
        #     [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        # )

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for i, layer in enumerate(self.layers[:n_q]):
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            # the original code is below
            # since quantize has the gradient of residual, according to line 321
            #  quantize = x + (quantize - x).detach()
            # the code below will make commitment loss to be 0 for all codebooks except for codebook1
            # https://github.com/facebookresearch/encodec/issues/25
            # therefore we change it

            residual = residual - quantized
            # residual = residual - quantized.detach()
            # since commitment loss is averaged, the scale of the loss won't get change (not as said in the issue above)
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class ResidualVectorQuantizer(BaseQuantizer):
    """Residual Vector Quantizer.

    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        q_dropout (bool): Random quantizer drop out at train time.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        orthogonal_reg_weight (float): Orthogonal regularization weights.
        orthogonal_reg_active_codes_only (bool): Apply orthogonal regularization only on active codes.
        orthogonal_reg_max_codes (optional int): Maximum number of codes to consider.
            for orthogonal regularization.
    """
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        q_dropout: bool = False,
        bins: tp.Union[int, tp.List[int]] = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = False,
        orthogonal_reg_max_codes: tp.Optional[int] = None,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            orthogonal_reg_weight=self.orthogonal_reg_weight,
            orthogonal_reg_active_codes_only=self.orthogonal_reg_active_codes_only,
            orthogonal_reg_max_codes=self.orthogonal_reg_max_codes,
            channels_last=False
        )

    def forward(self, x: torch.Tensor, frame_rate: int):
        n_q = self.n_q
        if self.training and self.q_dropout:
            n_q = int(torch.randint(1, self.n_q + 1, (1,)).item())
        if type(self.bins) == list:
            bins = self.bins
        else:
            bins = [self.bins] * self.n_q
        bw_per_q = [math.log2(bin) * frame_rate / 1000 for bin in bins]
        bw = torch.tensor(sum(bw_per_q)).to(x)
        quantized, codes, commit_loss = self.vq(x, n_q=n_q)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        n_q = self.n_q
        codes = self.vq.encode(x, n_q=n_q)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks, vq.decode expects [K, B, T].
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        return quantized

    @property
    def total_codebooks(self):
        return self.max_n_q

    @property
    def num_codebooks(self):
        return self.n_q

    def set_num_codebooks(self, n: int):
        assert n > 0 and n <= self.max_n_q
        self.n_q = n

class DummyQuantizer(BaseQuantizer):
    """Fake quantizer that actually does not perform any quantization.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, frame_rate: int):
        q = x.unsqueeze(1)
        return QuantizedResult(x, q, torch.tensor(q.numel() * 32 * frame_rate / 1000 / len(x)).to(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        """
        return x.unsqueeze(1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        """
        return codes.squeeze(1)

    @property
    def total_codebooks(self):
        """Total number of codebooks."""
        return 1

    @property
    def num_codebooks(self):
        """Total number of codebooks."""
        return self.total_codebooks

    def set_num_codebooks(self, n: int):
        """Set the number of active codebooks."""
        raise AttributeError("Cannot override the number of codebooks for the dummy quantizer")


class EncodecModel(CompressionModel):
    """Encodec model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (BaseQuantizer): Quantizer network.
        frame_rate (int): Frame rate for the latent representation.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        renormalize (bool): Whether to renormalize the audio before running the model.
    """
    # we need assignment to override the property in the abstract class,
    # I couldn't find a better way...
    frame_rate: float = 0
    sample_rate: int = 0
    channels: int = 0

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 quantizer: BaseQuantizer,
                 frame_rate: int,
                 sample_rate: int,
                 channels: int,
                 causal: bool = False,
                 renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, 'Causal model does not support renormalize'

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.bins

    def preprocess(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(self,
                    x: torch.Tensor,
                    scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x

    def forward(self, x: torch.Tensor, encode=False) -> QuantizedResult:
        if encode:
            return self.encode(x)
        else:
            raise NotImplementedError("model forward and training is not supported.")
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)

        emb = self.encoder(x)
        q_res = self.quantizer(emb, self.frame_rate)
        out = self.decoder(q_res.x)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = self.postprocess(out, scale)

        return q_res

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale a float tensor containing the scale for audio renormalizealization.
        """
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb)
        return codes, scale

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)

class EncodecModel_encode_only(CompressionModel):
    """Encodec model operating on the raw waveform. Encode only, so no decoder

    Args:
        encoder (nn.Module): Encoder network.
        quantizer (BaseQuantizer): Quantizer network.
        frame_rate (int): Frame rate for the latent representation.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        renormalize (bool): Whether to renormalize the audio before running the model.
    """
    # we need assignment to override the property in the abstract class,
    # I couldn't find a better way...
    frame_rate: float = 0
    sample_rate: int = 0
    channels: int = 0

    def __init__(self,
                 encoder: nn.Module,
                 quantizer: BaseQuantizer,
                 frame_rate: int,
                 sample_rate: int,
                 channels: int,
                 causal: bool = False,
                 renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, 'Causal model does not support renormalize'

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.bins

    def preprocess(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(self,
                    x: torch.Tensor,
                    scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x

    def forward(self, x: torch.Tensor, encode=False) -> QuantizedResult:
        if encode:
            return self.encode(x)
        else:
            raise NotImplementedError("model forward and training is not supported.")
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)

        emb = self.encoder(x)
        q_res = self.quantizer(emb, self.frame_rate)
        out = self.decoder(q_res.x)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = self.postprocess(out, scale)

        return q_res

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale a float tensor containing the scale for audio renormalizealization.
        """
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb)
        return codes, scale

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        raise NotImplementedError("Decode is not supported for encode only model")
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        raise NotImplementedError("Decode is not supported for encode only model")
        return self.quantizer.decode(codes)

def get_quantizer(quantizer: str, cfg: omegaconf.DictConfig, dimension: int) -> BaseQuantizer:
    klass = {
        'no_quant': DummyQuantizer,
        'rvq': ResidualVectorQuantizer
    }[quantizer]
    kwargs = dict_from_config(getattr(cfg, quantizer))
    if quantizer != 'no_quant':
        kwargs['dimension'] = dimension
    return klass(**kwargs)

def get_encodec_autoencoder(encoder_name: str, cfg: omegaconf.DictConfig):
    if encoder_name == 'seanet':
        kwargs = dict_from_config(getattr(cfg, 'seanet'))
        encoder_override_kwargs = kwargs.pop('encoder')
        decoder_override_kwargs = kwargs.pop('decoder')
        encoder_kwargs = {**kwargs, **encoder_override_kwargs}
        decoder_kwargs = {**kwargs, **decoder_override_kwargs}
        encoder = SEANetEncoder(**encoder_kwargs)
        decoder = SEANetDecoder(**decoder_kwargs)
        return encoder, decoder
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")


def get_compression_model(ckpt_fn, encode_only=False, device="cpu") -> CompressionModel:
    """Instantiate a compression model."""
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_fn, map_location='cpu')
    cfg = state['xp.cfg']
    cfg.device = str(device)
    weights = state['best_state']['model']
    assert cfg.compression_model == 'encodec', "Only Encodec model is supported for now."
    if encode_only:
        all_keys = list(weights.keys())
        for key in all_keys:
            if key.startswith('decoder'):
                del weights[key]
        kwargs = dict_from_config(getattr(cfg, 'encodec'))
        encoder_name = kwargs.pop('autoencoder')
        quantizer_name = kwargs.pop('quantizer')
        encoder, _ = get_encodec_autoencoder(encoder_name, cfg)
        quantizer = get_quantizer(quantizer_name, cfg, encoder.dimension)
        frame_rate = kwargs['sample_rate'] // encoder.hop_length
        renormalize = kwargs.pop('renormalize', False)
        # deprecated params
        kwargs.pop('renorm', None)
        compression_model = EncodecModel_encode_only(encoder, quantizer,
                            frame_rate=frame_rate, renormalize=renormalize, **kwargs).to(cfg.device)
        assert compression_model.sample_rate == cfg.sample_rate, "Compression model sample rate should match"
        compression_model.load_state_dict(weights)
        compression_model.eval()
        return compression_model

    else:
        kwargs = dict_from_config(getattr(cfg, 'encodec'))
        encoder_name = kwargs.pop('autoencoder')
        quantizer_name = kwargs.pop('quantizer')
        encoder, decoder = get_encodec_autoencoder(encoder_name, cfg)
        quantizer = get_quantizer(quantizer_name, cfg, encoder.dimension)
        frame_rate = kwargs['sample_rate'] // encoder.hop_length
        renormalize = kwargs.pop('renormalize', False)
        # deprecated params
        kwargs.pop('renorm', None)
        compression_model = EncodecModel(encoder, decoder, quantizer,
                            frame_rate=frame_rate, renormalize=renormalize, **kwargs).to(cfg.device)
        assert compression_model.sample_rate == cfg.sample_rate, "Compression model sample rate should match"
        compression_model.load_state_dict(weights)
        compression_model.eval()
        return compression_model

if __name__ == "__main__":
    import torchaudio
    ckpt_fn = "/home/pyp/BoostedVoiceEditor/pretrained/encodec_6f79c6a8.th"
    audio_in_fns = ["/home/pyp/BoostedVoiceEditor/demo/pam.wav", "/home/pyp/BoostedVoiceEditor/demo/ray.wav", "/home/pyp/BoostedVoiceEditor/demo/84_121550_000074_000000.wav", "/home/pyp/BoostedVoiceEditor/demo/caribbean.wav", "/home/pyp/BoostedVoiceEditor/demo/bible.wav", "/home/pyp/BoostedVoiceEditor/demo/miley.wav"]
    audio_out_fns = ["/home/pyp/BoostedVoiceEditor/demo/pam_encodecTest.wav", "/home/pyp/BoostedVoiceEditor/demo/ray_encodecTest.wav", "/home/pyp/BoostedVoiceEditor/demo/84_121550_000074_000000_encodecTest.wav", "/home/pyp/BoostedVoiceEditor/demo/caribbean_encodecTest.wav", "/home/pyp/BoostedVoiceEditor/demo/bible_encodecTest.wav", "/home/pyp/BoostedVoiceEditor/demo/miley_encodecTest.wav"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_compression_model(ckpt_fn, device=device)

    for audio_in_fn, audio_out_fn in zip(audio_in_fns, audio_out_fns):
        audio_in, sr = torchaudio.load(audio_in_fn)
        if sr != model.sample_rate:
            audio_in = torchaudio.transforms.Resample(sr, model.sample_rate)(audio_in)
        if audio_in.shape[0] == 2:
            audio_in = audio_in.mean(dim=0, keepdim=True)
        audio_in = audio_in.unsqueeze(0)
        audio_in = audio_in.to(torch.float32).to(device)
        codes = model.encode(audio_in)[0]
        audio_out = model.decode(codes)[0].cpu()
        torchaudio.save(audio_out_fn, audio_out, model.sample_rate)