"""
VoiceStar: Robust, Duration-Controllable TTS that can Extrapolate

GitHub: https://github.com/jasonppy/VoiceStar
License: MIT

Copyright (c) 2025 Puyuan Peng
"""
import random, os, copy
from typing import Dict, Iterator, List, Tuple, Union
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
import torch.distributed as dist

from voicestar.modules.utils import make_pad_mask, generate_partial_autoregressive_mask

from voicestar.modules.embedding import (
    SinePositionalEmbedding,
    TokenEmbedding,
    SinePositionalEmbedding_progress,
)
from voicestar.modules.transformer import (
    AdaptiveLayerNorm,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerDecoder,
    TransformerEncoder,
    TransformerEncoderLayer,
)


def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=1.0,
    min_p=1.0,
    filter_value=-float("Inf"),
    min_tokens_to_keep=1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if min_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        indices_to_remove = probs < min_p
        if not torch.any(indices_to_remove.sum(-1) == logits.size(-1)):
            logits[indices_to_remove] = filter_value
            top_k = 0
            top_p = 1.0
        # else will use other types of sampling, or no filtering

    # If top_k is a single integer
    if isinstance(top_k, int) and top_k > 0:
        # Safety check to ensure we don't ask for more than available
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))

        # Remove all tokens with a probability less than the last token of the top-k
        threshold = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

    # If top_k is a list, assume it has the same length as M
    elif isinstance(top_k, list):
        # Ensure the length matches the first dimension
        assert len(top_k) == logits.size(
            0
        ), f"top_k list length ({len(top_k)}) must match logits.size(0) ({logits.size(0)})"

        for i in range(logits.size(0)):
            k_i = top_k[i]
            if k_i > 0:
                # Safety check
                k_i = min(max(k_i, min_tokens_to_keep), logits.size(-1))
                row_threshold = torch.topk(logits[i], k_i, dim=-1)[0][-1]
                indices_to_remove_i = logits[i] < row_threshold
                logits[i, indices_to_remove_i] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, min_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, min_p=min_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


class VoiceStarModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert (
            self.args.enc_dec ^ self.args.dec
        ), f"self.args.enc_dec: {self.args.enc_dec}, self.args.dec: {self.args.dec}"
        if not getattr(self.args, "special_first", False):
            self.args.special_first = 0
        if not getattr(self.args, "n_special", False):
            self.args.n_special = 3
        self.args.eos = getattr(self.args, "eos", -1)
        self.eog = nn.Parameter(
            torch.full((self.args.n_codebooks, 1), self.args.eog, dtype=torch.long),
            requires_grad=False,
        )  # [K 1]
        if self.args.eos > 0:
            assert (
                self.args.eos != self.args.audio_pad_token
                and self.args.eos != self.args.empty_token
            ), self.args.eos
            self.eos = nn.Parameter(
                torch.full((self.args.n_codebooks, 1), self.args.eos, dtype=torch.long),
                requires_grad=False,
            )  # [K 1]
        if type(self.args.audio_vocab_size) == str:
            self.args.audio_vocab_size = eval(self.args.audio_vocab_size)
        if type(self.args.audio_vocab_size) == list:  # otherwise they are all lists
            assert self.args.special_first

        self.n_text_tokens = self.args.text_vocab_size + 1
        assert (
            self.args.text_pad_token == self.args.text_vocab_size
        ), f"self.args.text_vocab_size: {self.args.text_vocab_size}, self.args.text_pad_token: {self.args.text_pad_token}"

        if self.args.special_first and type(self.args.audio_vocab_size) == list:
            self.n_audio_tokens = [
                tok + self.args.n_special for tok in self.args.audio_vocab_size
            ]  # special tokens: empty token, EOG token, audio pad token
            assert self.args.empty_token == 0, self.args.empty_token
            assert self.args.eog == 1, self.args.eog
            assert self.args.audio_pad_token == 2, self.args.audio_pad_token
        else:
            self.n_audio_tokens = [
                self.args.audio_vocab_size + self.args.n_special
            ] * self.args.n_codebooks  # special tokens: empty token, EOG token, audio pad token
            assert (
                self.args.audio_vocab_size == self.args.empty_token
            ), self.args.empty_token
            assert self.args.eog == self.args.audio_vocab_size + 1, self.args.eog
            assert (
                self.args.audio_pad_token == self.args.audio_vocab_size + 2
            ), self.args.audio_pad_token

        self.text_embedding = TokenEmbedding(
            dim_model=self.args.d_model,
            vocab_size=self.n_text_tokens,
            dropout=self.args.text_embedding_dropout,
        )

        self.audio_embedding = nn.ModuleList(
            [
                TokenEmbedding(
                    dim_model=self.args.audio_embedding_dim,
                    vocab_size=self.n_audio_tokens[k],
                    dropout=self.args.audio_embedding_dropout,
                )
                for k in range(self.args.n_codebooks)
            ]
        )

        rope_base = getattr(self.args, "rope_base", None)
        use_sinusoidal = getattr(self.args, "use_sinusoidal", False)
        use_sinusoidal_progress = getattr(self.args, "use_sinusoidal_progress", False)
        logging.info(f"rope_base: {rope_base}, use_sinusoidal: {use_sinusoidal}")
        if use_sinusoidal:
            self.text_positional_embedding = SinePositionalEmbedding(
                self.args.d_model,
                dropout=self.args.text_positional_embedding_dropout,
                scale=False,
                alpha=True,  # learnable scaler, scale the volume of positional embedding
            )
            self.audio_positional_embedding = SinePositionalEmbedding(
                self.args.d_model,
                dropout=self.args.audio_positional_embedding_dropout,
                scale=False,
                alpha=True,  # learnable scaler, scale the volume of positional embedding
            )
        elif use_sinusoidal_progress:
            self.text_positional_embedding = SinePositionalEmbedding_progress(
                self.args.d_model,
                dropout=self.args.text_positional_embedding_dropout,
                scale=False,
                alpha=True,  # learnable scaler, scale the volume of positional embedding
                args=self.args,
            )
            self.audio_positional_embedding = SinePositionalEmbedding_progress(
                self.args.d_model,
                dropout=self.args.audio_positional_embedding_dropout,
                scale=False,
                alpha=True,  # learnable scaler, scale the volume of positional embedding
                args=self.args,
            )

        else:

            class NoOp:
                def __init__(self):
                    pass

                def __call__(self, *args, **kwargs):
                    return args[0]

            self.text_positional_embedding = NoOp()
            self.audio_positional_embedding = NoOp()

        if self.args.enc_dec:
            enc_layer = TransformerEncoderLayer(
                d_model=self.args.d_model,
                nhead=self.args.nhead,
                dim_feedforward=self.args.d_model * 4,
                dropout=self.args.trm_dropout,
                batch_first=True,
                norm_first=True,
                layer_norm_cls=LayerNorm,
            )  # use the pre-norm arch

            self.encoder = TransformerEncoder(
                encoder_layer=enc_layer,
                num_layers=self.args.num_encoder_layers,
                norm=LayerNorm(self.args.d_model),
                rope_base=self.args.rope_base,
                d_model=self.args.d_model,
                nhead=self.args.nhead,
                args=self.args,
            )  # use the pre-norm arch

            dec_layer = TransformerDecoderLayer(
                d_model=self.args.d_model,
                nhead=self.args.nhead,
                dim_feedforward=self.args.d_model * 4,
                dropout=self.args.trm_dropout,
                batch_first=True,
                norm_first=True,
                layer_norm_cls=LayerNorm,
            )

            self.decoder = TransformerDecoder(
                decoder_layer=dec_layer,
                num_layers=self.args.num_decoder_layers,
                norm=LayerNorm(self.args.d_model),
                rope_base=self.args.rope_base,
                d_model=self.args.d_model,
                nhead=self.args.nhead,
                args=self.args,
            )  # NOTE: this one I use torch.nn native implementation, as it's not implemented in .modules

        else:
            dec_layer = TransformerEncoderLayer(
                self.args.d_model,
                self.args.nhead,
                dim_feedforward=self.args.d_model * 4,
                dropout=self.args.trm_dropout,
                batch_first=True,
                norm_first=True,
                layer_norm_cls=LayerNorm,
            )
            self.decoder = TransformerEncoder(
                dec_layer,
                num_layers=self.args.num_decoder_layers,
                norm=LayerNorm(self.args.d_model),
            )

        if type(self.args.audio_vocab_size) == int:
            self.predict_layer = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.args.d_model, self.args.audio_vocab_size // 2),
                        nn.GELU(),
                        nn.Linear(
                            self.args.audio_vocab_size // 2, self.n_audio_tokens[k]
                        ),
                    )
                    for k in range(self.args.n_codebooks)
                ]
            )
        else:
            self.predict_layer = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.args.d_model, self.args.d_model // 2),
                        nn.GELU(),
                        nn.Linear(self.args.d_model // 2, self.n_audio_tokens[k]),
                    )
                    for k in range(self.args.n_codebooks)
                ]
            )

        self.accuracy_metrics = nn.ModuleList(
            [
                MulticlassAccuracy(
                    self.n_audio_tokens[k],
                    top_k=10,
                    average="micro",
                    multidim_average="global",
                    ignore_index=None,
                )
                for k in range(self.args.n_codebooks)
            ]
        )

        if self.args.eog_weight != 1:
            raise NotImplementedError(
                "now have different vocab_size for different codebooks, therefore currently don't support eog_weight"
            )
            self.class_weight = nn.Parameter(
                torch.ones(self.n_audio_tokens), requires_grad=False
            )
            self.class_weight.data[self.args.eog] = self.args.eog_weight

    def dec_forward(
        self,
        x_input,
        x_lens,
        x_attention_mask,
        x_padding_mask,
        y_input,
        new_y_lens,
        y_attention_mask,
        y_padding_mask,
        need_weights=False,
        past=None,
        last_3_tokens=False,
    ):
        x_attn_mask = F.pad(
            x_attention_mask,
            (0, new_y_lens.max()),
            value=True,
        )  # x attn to all x, doesn't attn to any y, this follow figure 3 of the valle paper
        y_attn_mask = F.pad(
            y_attention_mask,
            (x_lens.max(), 0),  # y is padded at the front
            value=False,
        )  # y attn to all x, for y itself use lower triangle mask to ensure autoregressive
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)

        # merge key padding and attention masks
        bsz, src_len = x_input.shape[0], x_lens.max() + new_y_lens.max()
        xy_padding_mask = torch.concat([x_padding_mask, y_padding_mask], dim=1)
        _xy_padding_mask = (
            xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.args.nhead, -1, -1)
            .reshape(bsz * self.args.nhead, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)

        new_attn_mask = torch.zeros_like(xy_attn_mask)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask

        xy_input = torch.cat([x_input, y_input], dim=1)
        if need_weights:
            raise NotImplementedError("not implemented yet")
            out, layer_attn_weights = self.decoder(
                (xy_input, None), mask=xy_attn_mask, need_weights=True
            )
            return layer_attn_weights

        if past == None:  # do not use kvcache
            out, _ = self.decoder((xy_input, None), mask=xy_attn_mask)
            return out[:, x_lens.max() :], None
        else:  # use kvcache
            if (
                past.ndim > 3
            ):  # uses kvcache, only need to pass the last tokens, this doesn't work with multi-span speech editing yet
                if last_3_tokens:
                    xy_input = xy_input[:, -3:]
                    xy_attn_mask = xy_attn_mask[:, -3:]
                else:
                    xy_input = xy_input[:, -1:]
                    xy_attn_mask = xy_attn_mask[:, -1:]

            out, present = self.decoder((xy_input, None), mask=xy_attn_mask, past=past)
            if isinstance(out, tuple):  # get rid of stage_embedding
                out = out[0]

            if out.shape[1] > x_lens.max():  # the first pass, not kvcache yet
                return out[:, x_lens.max() :], present
            else:  # used kvcache
                return out, present

    def enc_dec_forward(
        self,
        xa,
        x_attention_mask,
        x_padding_mask,
        y_input,
        new_y_lens,
        y_attention_mask,
        y_padding_mask,
        tgt_y_lens=None,
        need_weights=False,
        past=None,
        last_3_tokens=False,
    ):
        assert not need_weights
        if past != None and past.ndim > 3:
            y_input = y_input[:, -1:]
            y_attention_mask = y_attention_mask[-1:]
        yhat, present = self.decoder(
            tgt=y_input,
            memory=xa,
            tgt_mask=y_attention_mask,
            tgt_key_padding_mask=y_padding_mask,
            memory_key_padding_mask=x_padding_mask,
            query_lens=tgt_y_lens,
            past=past,
        )
        return yhat, present

    def forward(self, batch, calc_loss=False):
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, K, T).
            where K is the number of codebooks
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
        """
        x, x_lens, y, y_lens = batch["x"], batch["x_lens"], batch["y"], batch["y_lens"]
        if len(x) == 0:
            return None
        x = x[
            :, : x_lens.max()
        ]  # this deal with gradient accumulation, where x_lens.max() might not be longer than the length of the current slice of x
        y = y[..., : y_lens.max()]
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3 and y.shape[1] == self.args.n_codebooks, y.shape
        assert y_lens.ndim == 1, y_lens.shape
        x_padding_mask = make_pad_mask(x_lens).to(x.device)
        x_attention_mask = (
            torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1)
            .bool()
            .to(x_padding_mask.device)
        )
        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input, x_lens)
        y_with_eos = [
            torch.cat([item[:, : y_lens[i]], self.eos], dim=-1)
            for i, item in enumerate(y)
        ]
        targets = y_with_eos
        # apply delayed stacking on y
        shifted_y = []
        patterns = []
        new_y_lens = []
        if getattr(self, "empty_tokens", None) == None:
            self.empty_tokens = torch.full(
                (self.args.n_codebooks, self.args.n_codebooks),
                self.args.empty_token,
                dtype=torch.long,
            ).to(
                y.device
            )  # [K, K]
        for i in range(len(y)):
            tmp = torch.cat(
                [y_with_eos[i], self.empty_tokens], dim=-1
            )  # [K, T+n_codebooks]
            for ii in range(self.args.n_codebooks):
                tmp[ii] = torch.roll(tmp[ii], shifts=ii + 1, dims=0)
            shifted_y.append(
                tmp.transpose(1, 0)
            )  # [K, T+n_codebooks] -> [T+n_codebooks, K]
            new_y_lens.append(y_with_eos[i].shape[1] + self.empty_tokens.shape[1])

        new_y_lens = torch.LongTensor(new_y_lens).to(y.device)

        cated_y = torch.nn.utils.rnn.pad_sequence(
            shifted_y, batch_first=False, padding_value=self.args.audio_pad_token
        )
        assert cated_y.shape == torch.Size(
            [max(new_y_lens), len(y), self.args.n_codebooks]
        ), cated_y.shape
        cated_y = cated_y.permute(2, 0, 1)  # [T,B,K]->[K,T,B]
        stacked_embedded_y = torch.stack(
            [self.audio_embedding[k](cated_y[k]) for k in range(self.args.n_codebooks)],
            dim=0,
        )  # [K, T, B, D]
        assert (
            stacked_embedded_y.shape[0] == self.args.n_codebooks
            and stacked_embedded_y.shape[2] == len(y)
            and stacked_embedded_y.shape[-1] == self.args.d_model
        ), stacked_embedded_y.shape
        embedded_y = stacked_embedded_y.sum(dim=0)  # [K,T,B,D]->[T,B,D]
        embedded_y = embedded_y.transpose(1, 0)  # [T,B,D]->[B,T,D]
        assert embedded_y.shape[1:] == torch.Size(
            [max(new_y_lens), self.args.d_model]
        ), embedded_y.shape
        y_input = self.audio_positional_embedding(embedded_y, new_y_lens)
        y_padding_mask = make_pad_mask(new_y_lens).to(y.device)
        y_attention_mask = (
            torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1)
            .bool()
            .to(y_padding_mask.device)
        )
        if self.args.dec:
            y_out = self.dec_forward(
                x_input,
                x_lens,
                x_attention_mask,
                x_padding_mask,
                y_input,
                new_y_lens,
                y_attention_mask,
                y_padding_mask,
            )
        else:
            xa = self.encoder(src=x_input, src_key_padding_mask=x_padding_mask)
            y_out = self.enc_dec_forward(
                xa,
                x_attention_mask,
                x_padding_mask,
                y_input,
                new_y_lens,
                y_attention_mask,
                y_padding_mask,
            )
        y_out = y_out[0]  # no kv-caching during training
        assert (
            y_out.shape == y_input.shape
        ), f"y_out.shape: {y_out.shape}, y_input.shape: {y_input.shape}"  # [B S D]
        logits = torch.stack(
            [self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)], dim=1
        )  # [B K S card]
        assert (
            logits.shape[1] == self.args.n_codebooks
            and logits.shape[3] == self.n_audio_tokens[0]
        ), logits.shape
        logits_use = [
            logit[:, : new_y_lens[i]] for i, logit in enumerate(logits)
        ]  # each of shape [K, T, card]
        logits_final = []
        for i, logit in enumerate(logits_use):
            logit_copy = logit.clone()
            for ii in range(self.args.n_codebooks):
                logit_copy[ii] = torch.roll(logit_copy[ii], shifts=-ii, dims=0)
            logit = logit_copy[
                :, : -self.args.n_codebooks
            ]  # [K, T, card] -> [K, T-n_codebooks, card]
            logits_final.append(logit)
        if self.args.no_loss_on_prefix:
            assert (
                "y_sep_token_position" in batch
            ), f"y_sep_token_position should be in batch, but it's not"
            logit_temp = []
            target_temp = []
            for jj, (logit, target) in enumerate(zip(logits_final, targets)):
                # TODO already taken into consideration in depth transformer
                logit_temp.append(logit[:, batch["y_sep_token_position"][jj] :])
                target_temp.append(target[:, batch["y_sep_token_position"][jj] :])
            logits_final = logit_temp
            targets = target_temp
        logits = torch.cat(logits_final, dim=1)  # [K, T1+T2+T3+..., card]
        targets = torch.cat(targets, dim=1)  # [K, T1+T2+T3+...]

        assert targets.shape[:2] == logits.shape[:2], f"{targets.shape}, {logits.shape}"
        loss = []
        ntokens = []
        top10acc = []
        for k, (logit, target) in enumerate(
            zip(logits, targets)
        ):  # even though the loss and top10acc is calculated in a loop (loop through n_codebooks), validation is still taking a lot of mem, need to optimize this a little more
            loss.append(
                F.cross_entropy(
                    logit,
                    target,
                    reduction="mean",
                    weight=(
                        self.class_weight.data if self.args.eog_weight != 1 else None
                    ),
                    ignore_index=(
                        self.args.y_sep_token if self.args.y_sep_token != None else -100
                    ),
                )
            )  # ignore audio sep token as it's unpredictable (like the random early stop bug happened in 2023)
            # NOTE have to ignore the sep token in the loss calculation
            top10acc.append(self.accuracy_metrics[k](logit.detach(), target))
            ntokens.append(len(logit))

        all_ntokens = sum(ntokens)
        if self.args.codebook_weight != None:
            codebook_weight = (
                eval(self.args.codebook_weight)
                if isinstance(self.args.codebook_weight, str)
                else self.args.codebook_weight
            )
        else:
            codebook_weight = [1.0] * self.args.n_codebooks
        perplexity_by_codebook = [torch.exp(l) for l in loss]
        loss = sum([l * nt * cw for l, nt, cw in zip(loss, ntokens, codebook_weight)])

        top10acc_by_codebook = [t10a * nt for t10a, nt in zip(top10acc, ntokens)]
        top10acc = sum(top10acc_by_codebook)

        ntokens = torch.tensor(all_ntokens).to(logits.device)

        ret = {
            "loss": loss,
            "perplexity_by_codebook": perplexity_by_codebook,
            "top10acc": top10acc,
            "top10acc_by_codebook": top10acc_by_codebook,
            "effective_ntoken": ntokens,
        }

        return ret

    def inference_tts(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        tgt_y_lens: torch.Tensor,  #
        top_k: Union[int, list[int]] = -100,
        top_p: float = 1.0,
        min_p: float = 1.0,
        temperature: float = 1.0,
        stop_repetition: int = 3,
        kvcache: int = 1,
        silence_tokens: list[int] = [],
        multi_trial: list[int] = [],
        *kargs,
    ) -> torch.Tensor:
        """
        This implementation uses kvcache, which should have significant speed up
        Args:
          x:
            A 2-D tensor of shape (1, L).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, K).
          tgt_y_lens:
            *new arg* this specify the target length of y
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          top_p: (`optional`) float
            For Neucleus sampling
          min_p: (`optional`) float
            For min_p filtered sampling
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
          multi_trial: (`optional`) list[int]
            If not empty, it will be [n_trials, beam_size, trial_interval]
            from the start and begining trial_interval, we duplicate the current sample by beam_size,
            at the end of every trial_interval, we choose the sample with the highest log likelihood to keep and throw away the rest
        """
        eog_inference = self.args.eos if self.args.eos > 0 else self.args.eog
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        if self.args.special_first:
            y = y + int(self.args.n_special)
        y = y.transpose(2, 1)  # [1,T,K] -> [1,K,T]
        assert (
            y.shape[0] == 1 and y.shape[1] == self.args.n_codebooks
        ), y.shape  # there is no padding

        # make x attention mask and x_input
        x_attention_mask = (
            torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1)
            .bool()
            .to(x.device)
        )
        # x_attention_mask = torch.zeros(x.shape[1], x.shape[1]).bool().to(x.device)
        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input, x_lens)

        y_len = y.shape[2]
        y_lens = torch.LongTensor([y_len]).to(y.device)

        # rearrange y, we don't add eog to the end, this doesn't actually do anything in the tts scenario
        rearranged_y = [[y[0]]]
        assert rearranged_y[0][0].shape[0] == self.args.n_codebooks, rearranged_y[0][
            0
        ].shape

        # # shift y to create the delayed pattern
        if getattr(self, "empty_tokens", None) == None:
            self.empty_tokens = torch.full(
                (self.args.n_codebooks, self.args.n_codebooks),
                self.args.empty_token,
                dtype=torch.long,
            ).to(
                y.device
            )  # [K, K]
        temp = rearranged_y[0][0]
        assert temp.ndim == 2 and temp.shape[0] == self.args.n_codebooks, temp.shape
        temp = torch.cat([temp, self.empty_tokens], dim=-1)  # [K, T+n_codebooks]
        for ii in range(self.args.n_codebooks):
            temp[ii] = torch.roll(temp[ii], shifts=ii + 1, dims=0)
        shifted_y = [[temp]]

        # below is different from forward or inference
        # where we cut this shifted part
        shifted_y[0][0] = shifted_y[0][0][:, : -(self.args.n_codebooks - 1)]
        assert (
            not (
                shifted_y[0][0][self.args.n_codebooks :] == self.args.empty_token
            ).any()
            and not (shifted_y[0][0][self.args.n_codebooks :] == self.args.eog).any()
        ), shifted_y[0][0]
        # next section in inference is insert mask at the intersection of each tensor in a sample, but we don't need to do that
        # next section is concate tensors of each sample to one tensor, which we also don't need
        cated_y = shifted_y[0][0].unsqueeze(-1)  # [K,S]->[K,S,B]
        new_y_lens = torch.LongTensor([cated_y.shape[1]]).to(cated_y.device)
        assert cated_y.shape == torch.Size((self.args.n_codebooks, cated_y.shape[1], 1))
        assert not (cated_y == self.args.audio_pad_token).any(), cated_y

        # replace tokens in y with the embeddings, add sum codebooks up
        embedded_y = torch.stack(
            [self.audio_embedding[k](cated_y[k]) for k in range(self.args.n_codebooks)],
            dim=0,
        )  # [K, S, B, D]
        assert embedded_y.shape[0] == self.args.n_codebooks, embedded_y.shape
        assert embedded_y.shape[-1] == self.args.d_model, embedded_y.shape
        embedded_y = embedded_y.sum(dim=0)  # [K,S,B,D]->[S,B,D]
        embedded_y = embedded_y.transpose(1, 0)  # [S,B,D]->[B,S,D]

        # positional embedding
        y_input = self.audio_positional_embedding(embedded_y, tgt_y_lens)

        # make attention mask and padding mask
        y_attention_mask = (
            torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1)
            .bool()
            .to(y.device)
        )

        x_padding_mask = torch.full((1, x_lens[0]), False).to(x.device)
        y_padding_mask = torch.full((1, new_y_lens[0]), False).to(y.device)

        # entering the generation stage
        # starting from line 708
        codebook_eog = [False] * self.args.n_codebooks
        generated = []  # doesn't contain any empty token, contain eog
        cur_generated = []
        # say 0 is empty, 4 is eog
        # tensor([[ 1,  2,  3,  4,  0,  0],
        #         [ 0,  1,  2,  3,  4,  0],
        #         [ 0,  0,  1,  2,  3,  4]])
        num_gen = []
        cur_num_gen = 0
        ##################### silence repetition handling #####################
        ##################### silence repetition handling #####################
        # silence_tokens = [1388,1898,131] # [1388, 2045, 2041, 1996]
        # silence_tokens = []
        consec_silence_count = 0
        prev_token = None
        ##################### silence repetition handling #####################
        ##################### silence repetition handling #####################

        def sample_helper(
            n_eog,
            logits,
            codebook_eog,
            top_k,
            top_p,
            min_p,
            temperature,
            prev_token,
            consec_silence_count,
            stop_repetition,
            silence_tokens,
            cur_num_gen,
        ):
            if n_eog == 0:
                logits_adjust = logits
                for jj in range(1, self.args.n_codebooks):
                    logits_adjust[jj][eog_inference] = -10000
                    logits_adjust[jj][self.args.empty_token] = -10000
                if (
                    cur_num_gen <= self.args.encodec_sr // 5
                ):  # this shouldn't happen, but just in case the model stopped too early
                    logits_adjust[0][eog_inference] = -10000
                ##################### silence repetition handling #####################
                if (
                    stop_repetition > 0
                    and prev_token in silence_tokens
                    and consec_silence_count > stop_repetition
                ):
                    if logits_adjust[0, prev_token] < 0:
                        logits_adjust[0, prev_token] = logits_adjust[0, prev_token] * (
                            consec_silence_count - (stop_repetition - 1)
                        )
                    else:
                        logits_adjust[0, prev_token] = logits_adjust[0, prev_token] / (
                            consec_silence_count - (stop_repetition - 1)
                        )
                ##################### silence repetition handling #####################
                samples = topk_sampling(
                    logits_adjust,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    temperature=temperature,
                )  # [K, 1]
                assert samples.shape == torch.Size(
                    (self.args.n_codebooks, 1)
                ), f"samples.shape: {samples.shape}"
                if cur_num_gen < self.args.n_codebooks - 1:
                    for jj in range(1, self.args.n_codebooks - cur_num_gen):
                        samples[-jj, 0] = self.args.empty_token

                if (
                    (
                        samples[0, 0] == eog_inference
                        or torch.argmax(logits[0], dim=-1) == eog_inference
                        or y_input.shape[1] > x_lens[0] * (self.args.encodec_sr // 4)
                    )
                    or self.args.rope_base is not None
                    and not self.args.decoder_regular_rope
                    and self.args.progress_no_multiple
                    and cur_num_gen
                    > (
                        tgt_y_lens[0]
                        + self.args.encodec_sr * getattr(self.args, "extra_cutoff", 5)
                    )
                ):
                    # last one condition in the first bracket means y is already too long, shouldn't happen, but put it here
                    # the second bracket means we are using progress-monitoring RoPE, but the model is generating excessively long sequence (5 seconds more than specified), in which case we terminate the generation
                    samples[0, 0] = eog_inference
                    codebook_eog[0] = True
                ##################### silence repetition handling #####################
                if samples[0, 0] in silence_tokens and samples[0, 0] == prev_token:
                    consec_silence_count += 1
                else:
                    consec_silence_count = 0
                prev_token = samples[0, 0]
                ##################### silence repetition handling #####################
                return samples, codebook_eog, prev_token, consec_silence_count
            else:
                assert (
                    sum(codebook_eog[i] for i in range(n_eog)) == n_eog
                ), f"codebook_eog: {codebook_eog}, but n_eog: {n_eog}"
                logits_adjust = logits
                for jj in range(n_eog + 1, self.args.n_codebooks):
                    logits_adjust[jj][eog_inference] = -10000
                    logits_adjust[jj][self.args.empty_token] = -10000
                samples = topk_sampling(
                    logits_adjust,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    temperature=temperature,
                )  # [K, 1]
                for jj in range(n_eog):
                    samples[jj, 0] = self.args.empty_token
                samples[n_eog, 0] = eog_inference
                codebook_eog[n_eog] = True
                return samples, codebook_eog, prev_token, consec_silence_count

        # prepare the cache placeholder
        # n_layers, 2, bsz, num_heads, src_len, head_dim, 2 means [key, value]
        past = (
            torch.ones(
                [self.args.num_decoder_layers, 2, x.shape[0]],
                device=x.device,
                dtype=torch.float32,
            )
            if kvcache
            else None
        )
        if self.args.enc_dec:
            xa = self.encoder(src=x_input, src_key_padding_mask=x_padding_mask)
        while True:
            if self.args.dec:
                y_out, present = self.dec_forward(
                    x_input,
                    x_lens,
                    x_attention_mask,
                    x_padding_mask,
                    y_input,
                    new_y_lens,
                    y_attention_mask,
                    y_padding_mask,
                    past=past,
                )
            else:
                y_out, present = self.enc_dec_forward(
                    xa,
                    x_attention_mask,
                    x_padding_mask,
                    y_input,
                    new_y_lens,
                    y_attention_mask,
                    y_padding_mask,
                    tgt_y_lens=tgt_y_lens,
                    past=past,
                )
            if past != None:
                past = (
                    torch.cat([past, present.to(past.dtype)], dim=-2)
                    if past.ndim > 3
                    else present.to(past.dtype)
                )

            y_out = y_out[:, -1:]  # only take the last token

            logits = torch.stack(
                [self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)],
                dim=1,
            )  # [B K S card], B==S==1, so [1 K 1 card]
            logits = logits.squeeze(0).squeeze(1)  # [K card]
            assert logits.shape == torch.Size(
                (self.args.n_codebooks, self.n_audio_tokens[0])
            ), f"{logits.shape}"

            n_eog = sum(codebook_eog)
            assert n_eog < self.args.n_codebooks
            if (
                self.args.eos > 0
            ):  # if we are using end-of-sentence token (which is used by default), eog shouldn't be used here, as there is no masked spans
                for jj in range(self.args.n_codebooks):
                    logits[jj][self.args.eog] = -10000.0

            samples, codebook_eog, prev_token, consec_silence_count = sample_helper(
                n_eog,
                logits,
                codebook_eog,
                top_k,
                top_p,
                min_p,
                temperature,
                prev_token,
                consec_silence_count,
                stop_repetition,
                silence_tokens,
                cur_num_gen,
            )
            # samples.shape is [K,1]
            # ge samples_emb
            samples_emb = torch.stack(
                [
                    self.audio_embedding[k](samples[k])
                    for k in range(self.args.n_codebooks)
                ],
                dim=0,
            )  # [K,1,D]
            samples_emb = samples_emb.sum(dim=0, keepdim=True)  # [1,1,D]

            cur_num_gen += 1
            cur_generated.append(samples.squeeze(-1))  # [K,1] -> [K]

            if (
                sum(codebook_eog) == self.args.n_codebooks
            ):  # generation for the current span is done
                codebook_eog = [False] * self.args.n_codebooks
                num_gen.append(cur_num_gen)
                cur_num_gen = 0
                generated.append(cur_generated)
                cur_generated = []
                break
            else:
                assert samples_emb.shape == torch.Size(
                    (1, 1, self.args.d_model)
                ), f"samples_emb.shape: {samples_emb.shape}"

            embedded_y = torch.cat([embedded_y, samples_emb], dim=1)
            new_y_lens = torch.LongTensor([embedded_y.shape[1]]).to(y.device)
            y_input = self.audio_positional_embedding(embedded_y, tgt_y_lens)  # [B T D]
            # make attention mask and padding mask
            y_attention_mask = (
                torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1)
                .bool()
                .to(y.device)
            )
            y_padding_mask = torch.full((1, new_y_lens[0]), False).to(y.device)

        assert len(generated) == 1, f"len(generated): {len(generated)}"

        # revert the pattern
        flatten_gen = []
        for l, orig_span in enumerate(generated):
            span = torch.stack(orig_span, dim=0)  # [T, K]
            span = span.transpose(1, 0)  # [K, T]
            assert span.shape[0] == self.args.n_codebooks, span.shape
            unshifted_span = []
            for j, s in enumerate(span):
                start_from = j
                end_at = -(self.args.n_codebooks - start_from)
                unshifted_span.append(s[start_from:end_at])
            unshifted_span = torch.stack(unshifted_span, dim=0)

            assert (
                unshifted_span.shape[1] == num_gen[l] - self.args.n_codebooks
            ), f"len(unshifted_spans[0]): {len(unshifted_span[0])}, num_gen[l]: {num_gen[l]}"

            flatten_gen.append(unshifted_span)
        assert len(flatten_gen) == 1, len(flatten_gen)

        # combine
        res = [y[0], flatten_gen[0]]
        res = torch.cat(res, dim=1).unsqueeze(0)  # [K, new_t] -> [1, K, new_T]
        expected_y_len = y_len + sum([item - self.args.n_codebooks for item in num_gen])
        assert res.shape == torch.Size(
            (1, self.args.n_codebooks, expected_y_len)
        ), f"res.shape: {res.shape}, expected_y_len: {expected_y_len}. y_len + sum([item - self.args.n_codebooks for item in num_gen]): {y_len} + {sum([item - self.args.n_codebooks for item in num_gen])}"

        if self.args.special_first:
            res = res - int(self.args.n_special)
            flatten_gen = flatten_gen - int(self.args.n_special)
        return res, flatten_gen[0].unsqueeze(0)
