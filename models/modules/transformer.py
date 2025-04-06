import copy, logging
import numbers
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .activation import MultiheadAttention
from .scaling import ActivationBalancer, BalancedDoubleSwish
from .scaling import BasicNorm as _BasicNorm

_shape_t = Union[int, List[int], torch.Size]


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            self.bias = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return (
                F.layer_norm(
                    input,
                    self.normalized_shape,
                    self.weight,
                    self.bias,
                    self.eps,
                ),
                embedding,
            )

        assert embedding is None
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            weight, bias = torch.split(
                self.project_layer(embedding),
                split_size_or_sections=self.d_model,
                dim=-1,
            )
            return (weight * self.norm(input) + bias, embedding)

        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


class BasicNorm(_BasicNorm):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super(BasicNorm, self).__init__(d_model, eps=eps)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return (
                super(BasicNorm, self).forward(input),
                embedding,
            )

        assert embedding is None
        return super(BasicNorm, self).forward(input)


class BalancedBasicNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super(BalancedBasicNorm, self).__init__()
        self.balancer = ActivationBalancer(
            d_model,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            max_abs=6.0,
        )
        self.norm = BasicNorm(d_model, eps, device=device, dtype=dtype)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return self.norm((self.balancer(input), embedding))

        assert embedding is None
        return self.norm(self.balancer(input))


class IdentityNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super(IdentityNorm, self).__init__()

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            return input

        assert embedding is None
        return input


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        linear1_self_attention_cls: nn.Module = nn.Linear,
        linear2_self_attention_cls: nn.Module = nn.Linear,
        linear1_feedforward_cls: nn.Module = nn.Linear,
        linear2_feedforward_cls: nn.Module = nn.Linear,
        layer_norm_cls: nn.Module = LayerNorm,
        layer_norm_eps: float = 1e-5,
        adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )

        # Implementation of Feedforward model
        self.linear1 = linear1_feedforward_cls(
            d_model, dim_feedforward, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(
            dim_feedforward, d_model, **factory_kwargs
        )

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            activation = BalancedDoubleSwish(d_model)

        # # We can't test self.activation in forward() in TorchScript,
        # # so stash some information about it instead.
        # if activation is F.relu or isinstance(activation, torch.nn.ReLU):
        #     self.activation_relu_or_gelu = 1
        # elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
        #     self.activation_relu_or_gelu = 2
        # else:
        #     self.activation_relu_or_gelu = 0
        self.activation = activation

        norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        if layer_norm_cls == IdentityNorm:
            norm2 = BalancedBasicNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
        else:
            norm2 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )

        if adaptive_layer_norm:
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
        else:
            self.norm1 = norm1
            self.norm2 = norm2

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        need_weights: Optional[bool] = False,
        past: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if isinstance(src, dict):
            sinu = src["sinu"]
            pm_sinu = src["pm_sinu"]
            src = src["input"]
        else:
            sinu = None
            pm_sinu = None
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True
        
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        if need_weights:
            raise NotImplementedError
            if self.norm_first:
                out, attn = self._sa_block_attn(
                    self.norm1(x, stage_embedding),
                    src_mask,
                    src_key_padding_mask,
                    past, sinu = sinu
                )
                out, present = out # present is the kvcache of the present timestep
                x = x + out
                x = x + self._ff_block(self.norm2(x, stage_embedding))
            else:
                out, attn = self._sa_block_attn(x, src_mask, src_key_padding_mask, past, sinu = sinu)
                out, present = out # present is the kvcache of the present timestep
                x = self.norm1(
                    x + out,
                    stage_embedding,
                )
                x = self.norm2(x + self._ff_block(x), stage_embedding)
            assert not is_src_tuple
                # return (x, stage_embedding)
            return (x, attn)
        else:
            if self.norm_first:
                out = self._sa_block(
                    self.norm1(x, stage_embedding),
                    src_mask,
                    src_key_padding_mask, past, sinu = sinu, q_sinu=pm_sinu['q'], k_sinu=pm_sinu['q']
                )
                out, present = out # present is the kvcache of the present timestep
                x = x + out
                x = x + self._ff_block(self.norm2(x, stage_embedding))
            else:
                out = self._sa_block(x, src_mask, src_key_padding_mask, sinu = sinu, q_sinu=pm_sinu['q'], k_sinu=pm_sinu['q'])
                out, present = out # present is the kvcache of the present timestep
                x = self.norm1(
                    x + out,
                    stage_embedding, past
                )
                x = self.norm2(x + self._ff_block(x), stage_embedding)
            
            if is_src_tuple:
                x = (x, stage_embedding)
            if present != None:
                x = [x, present]
            return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        past: Optional[Tensor] = None,
        sinu = None,
        q_sinu = None,
        k_sinu = None
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            past=past,
            sinu = sinu,
            q_sinu = q_sinu,
            k_sinu = k_sinu
        )
        x, present = x
        return self.dropout1(x), present

    # self-attention block, also return attention weights
    def _sa_block_attn(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        past: Optional[Tensor] = None,
    ) -> Tensor:
        x, attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            past=past
        )
        x, present = x
        return (self.dropout1(x), present), attn

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def pre_compute_sinusoidal(dim, base, max_len = 10000): # 4000 max length equivalent of mimi code is 320s, as mimi is 12.5hz
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    position_ids_expanded = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # [x_len_max, 1]
    inv_freq_expanded = inv_freq.unsqueeze(0).float() # [1, d//2]
    freqs = position_ids_expanded @ inv_freq_expanded # [x_len_max, d//2]
    freqs = torch.cat((freqs, freqs), dim=-1).unsqueeze(0) # [1, x_len_max, d]
    return {"sin": freqs.sin(), "cos": freqs.cos()}

def pre_compute_freqs(dim, base, max_len = 10000): # 4000 max length equivalent of mimi code is 320s, as mimi is 12.5hz
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    position_ids_expanded = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # [x_len_max, 1]
    inv_freq_expanded = inv_freq.unsqueeze(0).float() # [1, d//2]
    freqs = position_ids_expanded @ inv_freq_expanded # [x_len_max, d//2]
    freqs = torch.cat((freqs, freqs), dim=-1).unsqueeze(0) # [1, x_len_max, d]
    return freqs

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None, rope_base=None, d_model=None, nhead=None, args=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        if args != None:
            self.progress_no_multiple = args.progress_no_multiple
            self.progress_scale = args.progress_scale
        else:
            self.progress_no_multiple = False
            self.progress_scale = 1

        if rope_base is not None:
            if self.progress_no_multiple:
                self.pm_freqs = pre_compute_freqs(d_model//nhead, rope_base)
                self.sinu = None
            else:
                self.sinu = pre_compute_sinusoidal(d_model/nhead, rope_base)
                self.pm_freqs = None
            # logging.info(f"get precomputed sinusoidal for {rope_base=}: {self.sinu=}")
        else:
            self.sinu = None
            self.pm_freqs = None

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_layer_states: bool = False,
        need_weights:Optional[bool] = False,
        past: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            return_layer_states: return layers' state (optional).

        Shape:
            see the docs in Transformer class.
        """
        if return_layer_states:
            raise NotImplementedError
            assert not need_weights
            layer_states = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    past=past
                )
                layer_states.append(output[0])

            if self.norm is not None:
                output = self.norm(output)

            return layer_states, output
        if need_weights:
            raise NotImplementedError
            assert not return_layer_states
            layer_attn = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                    past=past
                )
                layer_attn.append(output[1])

            if self.norm is not None:
                output = self.norm(output)

            return layer_attn, output
        
        output = src
        all_present = []
        if self.sinu is not None:
            # use rope
            assert self.pm_freqs is None
            for k, v in self.sinu.items():
                self.sinu[k] = v.to(output.device)
        if self.pm_freqs is not None:
            assert self.sinu is None
            self.pm_freqs = self.pm_freqs.to(output.device)
            if src_key_padding_mask != None:
                query_lens = (~src_key_padding_mask).int().sum(-1).to(output.device)
            else:
                query_lens = torch.tensor([output.shape[1]]*output.shape[0]).to(output.device)
            assert query_lens.ndim==1, query_lens
            q_lens_expanded = query_lens.unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]
            query_ids_multiple = q_lens_expanded / (q_lens_expanded - 1)
            q_emb = self.pm_freqs * query_ids_multiple # [B, q_len_max, d]
            q_emb = q_emb / q_lens_expanded * self.progress_scale
            q_cos = q_emb.cos().unsqueeze(1) # [B, 1, q_len_max, d] # 1 is for nhead
            q_sin = q_emb.sin().unsqueeze(1)
            self.pm_sinu = {"q": {"cos": q_cos, "sin": q_sin}}
        else:
            self.pm_sinu = {"q": None}

        output = {"input": output, "sinu": self.sinu, "pm_sinu": self.pm_sinu}
        for n_layer, mod in enumerate(self.layers):
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, past=None if past is None else past[n_layer]
            )
            if isinstance(output, list):
                output, present = output
                all_present.append(present)
            if self.sinu is not None or self.pm_sinu is not None:
                output = {"input": output, "sinu": self.sinu, "pm_sinu": self.pm_sinu}
        if self.sinu is not None or self.pm_sinu is not None:
            output = output["input"]
        if self.norm is not None:
            output = self.norm(output)
        if all_present != []:
            all_present = torch.stack(all_present, dim=0) # (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
            output = [output, all_present]
        return output


class TransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        linear1_self_attention_cls: nn.Module = nn.Linear,
        linear2_self_attention_cls: nn.Module = nn.Linear,
        linear1_feedforward_cls: nn.Module = nn.Linear,
        linear2_feedforward_cls: nn.Module = nn.Linear,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        layer_norm_cls: nn.Module = LayerNorm,
        layer_norm_eps: float = 1e-5,
        adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = linear1_feedforward_cls(
            d_model, dim_feedforward, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(
            dim_feedforward, d_model, **factory_kwargs
        )

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            self.activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            self.activation = BalancedDoubleSwish(d_model)
        else:
            self.activation = activation

        if adaptive_layer_norm:
            norm1 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
            norm2 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
            norm3 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )

            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
            self.norm3 = AdaptiveLayerNorm(d_model, norm3)
        else:
            self.norm1 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm2 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
            if layer_norm_cls == IdentityNorm:
                self.norm3 = BalancedBasicNorm(
                    d_model, eps=layer_norm_eps, **factory_kwargs
                )
            else:
                self.norm3 = layer_norm_cls(
                    d_model, eps=layer_norm_eps, **factory_kwargs
                )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = False, # for compatibility with the nn.TransformerDecoder, not used
        memory_is_causal: Optional[bool] = False, # for compatibility with the nn.TransformerDecoder, not used
        past: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            past: the previous kvcache of the decoder (optional). shape: (2, batch_size, num_heads, seq_len, head_dim)

        Shape:
            see the docs in Transformer class.
        """
        if isinstance(tgt, dict):
            pm_sinu = tgt["pm_sinu"]
            sinu = tgt["sinu"]
            args = tgt["args"]
            tgt = tgt["input"]
        else:
            pm_sinu = None
            sinu = None
            args = None
        tgt_is_tuple = False
        if isinstance(tgt, tuple):
            x, stage_embedding = tgt
            tgt_is_tuple = True
        else:
            x, stage_embedding = tgt, None
        # logging.info(f"{tgt_key_padding_mask=}, {memory_key_padding_mask=}")
        # logging.info(f"{tgt_key_padding_mask.shape=}, {memory_key_padding_mask.shape=}")
        # logging.info(f"{query_lens=}, {key_lens=}")

        # past stores the kvcache for self-attention, and it can also be used to infer q_offset
        if past is not None and past.ndim > 2:
            q_offset = past[0].shape[-2] # past is (2, batch_size, num_heads, seq_len, head_dim), 2 contains [k, v], these are for self-attn, therefore also reflect the length of q
        else:
            q_offset = 0


        if self.norm_first:
            temp = self._sa_block(
                self.norm1(x, stage_embedding), tgt_mask, tgt_key_padding_mask, q_sinu=pm_sinu['q'], k_sinu=pm_sinu['q'], sinu=sinu, args = args, past=past, q_offset=q_offset
            )
            present = temp[1]
            x = x + temp[0]
            cross_out = self._mha_block(
                self.norm2(x, stage_embedding),
                memory,
                memory_mask,
                memory_key_padding_mask, q_sinu=pm_sinu['q'], k_sinu=pm_sinu['k'], sinu=sinu, args = args, q_offset=q_offset
            )
            if isinstance(cross_out, dict):
                attention_weights = cross_out["attention_weights"]
                cross_out = cross_out["x"]
            else:
                attention_weights = None
            x = x + cross_out
            x = x + self._ff_block(self.norm3(x, stage_embedding))
        else:
            temp = self._sa_block(x, tgt_mask, tgt_key_padding_mask, q_sinu=pm_sinu['q'], k_sinu=pm_sinu['q'], sinu=sinu, args = args, past=past, q_offset=q_offset)
            present = temp[1]
            x = self.norm1(
                x + temp[0],
                stage_embedding,
            )
            cross_out = self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, q_sinu=pm_sinu['q'], k_sinu=pm_sinu['k'], sinu=sinu, args=args, q_offset=q_offset
                )
            if isinstance(cross_out, dict):
                attention_weights = cross_out["attention_weights"]
                cross_out = cross_out["x"]
            else:
                attention_weights = None 
            x = self.norm2(
                x
                + cross_out,
                stage_embedding,
            )
            x = self.norm3(x + self._ff_block(x), stage_embedding)
        
        if attention_weights is not None:
            x = {"x": x, "attention_weights": attention_weights}
        if tgt_is_tuple:
            x = (x, stage_embedding)
        if present != None:
            x = [x, present]
        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        q_sinu=None,
        k_sinu=None,
        sinu = None, 
        args = None,
        past = None,
        q_offset = 0
    ) -> Tensor:
        # if past is not None and past.ndim > 2:
        #     print(f"self-attn, k len: {past[0].shape[-2] + x.shape[-2]}, q len: {x.shape[-2]} q_offset: {q_offset}")
        # else:
        #     print(f"self-attn, k len: {x.shape[-2]}, q len: {x.shape[-2]} q_offset: {q_offset}")
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            q_sinu = q_sinu,
            k_sinu = k_sinu,
            sinu = sinu,
            past = past,
            q_offset = q_offset
        )
        x, present = x
        return self.dropout1(x), present

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        q_sinu = None,
        k_sinu = None,
        sinu = None, 
        args = None,
        q_offset = 0
    ) -> Tensor:
        # print(f"cross-attn, k len: {mem.shape[-2]}, q len: {x.shape[-2]} q_offset: {q_offset}")
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            q_sinu = q_sinu,
            k_sinu = k_sinu,
            sinu = sinu,
            args = args,
            q_offset = q_offset
        )
        if len(x) == 2 and isinstance(x[0], dict) and "attention_weights" in x[0]:
            x, present = x
            attention_weights = x['attention_weights']
            x = x['attn_output']
            return {"x": self.dropout2(x), "attention_weights": attention_weights}
        elif len(x) == 2:
            x = x[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation)
    )
def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )
def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    __constants__ = ['norm']

    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
        rope_base=None, 
        d_model=None, 
        nhead=None,
        args=None
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.args = args
        if getattr(self.args, 'decoder_regular_rope', False):
            self.sinu = pre_compute_sinusoidal(d_model/nhead, rope_base)
            self.pm_freqs = None
        else:
            self.sinu = None
            if rope_base is not None:
                self.pm_freqs = pre_compute_freqs(d_model/nhead, rope_base)
                # logging.info(f"get precomputed freqs for {rope_base=}: {self.freqs=}")
            else:
                self.pm_freqs = None
        self.progress_scale = getattr(self.args, "progress_scale", 1.0)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False, query_lens: Optional[Tensor] = None, key_lens: Optional[Tensor] = None, past: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        # seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        # tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)
        if self.sinu is not None:
            assert self.pm_freqs is None
            for key in self.sinu:
                self.sinu[key] = self.sinu[key].to(output.device)
        if self.pm_freqs is not None:
            assert self.sinu is None
            if not self.training and hasattr(self, "pm_sinu") and past is not None and past[0].ndim > 2: # inference mode, will use cached sinu for the same example
                assert self.pm_sinu['q'] is not None and self.pm_sinu['k'] is not None
                # check batch size, need to modify the batch size if we use multi_trial during inference
                if self.pm_sinu['q']['cos'].shape[0] != tgt.shape[0]:
                    if self.pm_sinu['q']['cos'].shape[0] > tgt.shape[0]:
                        self.pm_sinu['q']['cos'] = self.pm_sinu['q']['cos'][:tgt.shape[0]]
                        self.pm_sinu['q']['sin'] = self.pm_sinu['q']['sin'][:tgt.shape[0]]
                        self.pm_sinu['k']['cos'] = self.pm_sinu['k']['cos'][:tgt.shape[0]]
                        self.pm_sinu['k']['sin'] = self.pm_sinu['k']['sin'][:tgt.shape[0]]
                    else:
                        assert self.pm_sinu['q']['cos'].shape[0] == 1
                        self.pm_sinu['q']['cos'] = self.pm_sinu['q']['cos'].repeat(tgt.shape[0], 1, 1, 1)
                        self.pm_sinu['q']['sin'] = self.pm_sinu['q']['sin'].repeat(tgt.shape[0], 1, 1, 1)
                        self.pm_sinu['k']['cos'] = self.pm_sinu['k']['cos'].repeat(tgt.shape[0], 1, 1, 1)
                        self.pm_sinu['k']['sin'] = self.pm_sinu['k']['sin'].repeat(tgt.shape[0], 1, 1, 1)
                pass
            else:
                self.pm_freqs = self.pm_freqs.to(output.device)
                if query_lens is None:
                    query_lens = (~tgt_key_padding_mask).int().sum(-1).to(tgt.device)
                if key_lens is None:
                    key_lens = (~memory_key_padding_mask).int().sum(-1).to(tgt.device)
                assert key_lens.ndim==1, key_lens
                assert query_lens.ndim==1, query_lens
                q_lens_expanded = query_lens.unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]
                k_lens_expanded = key_lens.unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]
                query_ids_multiple = q_lens_expanded / (q_lens_expanded - 1)
                key_ids_multiple = k_lens_expanded / (k_lens_expanded - 1)
                q_emb = self.pm_freqs * query_ids_multiple # [B, q_len_max, d]
                k_emb = self.pm_freqs * key_ids_multiple # [B, k_len_max, d]
                q_emb = q_emb / q_lens_expanded * self.progress_scale
                k_emb = k_emb / k_lens_expanded * self.progress_scale
                q_cos = q_emb.cos().unsqueeze(1) # [B, 1, q_len_max, d] # 1 is for nhead
                q_sin = q_emb.sin().unsqueeze(1)
                k_cos = k_emb.cos().unsqueeze(1)
                k_sin = k_emb.sin().unsqueeze(1)
                self.pm_sinu = {"q": {"cos": q_cos, "sin": q_sin}, "k": {"cos": k_cos, "sin": k_sin}}
        else:
            self.pm_sinu = {"q": None, "k": None}

        output = {"input": output, "pm_sinu": self.pm_sinu, "sinu": self.sinu, "args": self.args}
        if past != None:
            all_present = []
        if self.training and getattr(self.args, "attention_alignment_loss", 0):
            all_attn_weights = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory, tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        past=past[i] if past != None else None
                        #  tgt_is_causal=tgt_is_causal,
                        #  memory_is_causal=memory_is_causal
                        )
            if past != None:
                output, cur_present = output
                all_present.append(cur_present)
            if isinstance(output, dict):
                current_attn_weights = output["attention_weights"]
                all_attn_weights.append(current_attn_weights)
                output = output["x"]
            if self.sinu is not None or self.pm_sinu is not None:
                output = {"input": output, "pm_sinu": self.pm_sinu, "sinu": self.sinu, "args": self.args}
        if self.pm_sinu is not None or self.sinu is not None:
            output = output["input"]
        if self.norm is not None:
            output = self.norm(output)
        if self.training and getattr(self.args, "attention_alignment_loss", 0):
            assert len(all_attn_weights) == self.num_layers, f"{len(all_attn_weights)=}, {self.num_layers=}"
            output = {"output": output, "attention_weights": all_attn_weights}
        if past != None:
            all_present = torch.stack(all_present, dim=0)
            output = [output, all_present]
        else:
            output = [output, None]
        return output