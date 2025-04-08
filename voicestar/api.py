"""
VoiceStar: Robust, Duration-Controllable TTS that can Extrapolate

GitHub: https://github.com/jasonppy/VoiceStar
License: MIT

Copyright (c) 2025 Puyuan Peng
"""

import argparse, pickle
import logging
import os, random
import numpy as np
import torch
import torchaudio

from data.tokenizer import AudioTokenizer, TextTokenizer, tokenize_audio, tokenize_text
import argparse, time, tqdm


class VoiceStarAPI:
    def __init__(
        self,
        model,
        model_args,
        phn2num,
        text_tokenizer,
        audio_tokenizer,
        device="cuda",
        codec_sr=50,
        top_k=0,
        top_p=0.8,
        min_p=0.0,
        temperature=1.0,
        stop_repetition=-1,
        kvcache=1,
        silence_tokens="[1388,1898,131]",
        quiet=False,
    ):
        """
        Initialize the VoiceStar API with model and configuration.

        Args:
            model: The VoiceStar model
            model_args: Model arguments
            phn2num: Phoneme to number mapping
            text_tokenizer: Text tokenizer
            audio_tokenizer: Audio tokenizer
            device: Device to run inference on
            codec_sr: Codec sample rate
            top_k, top_p, min_p, temperature: Sampling parameters
            stop_repetition: Stop generation when token repeats this many times
            kvcache: Whether to use KV cache for faster inference
            silence_tokens: Tokens representing silence
            quiet: Whether to suppress logging
        """
        self.model = model
        self.model_args = model_args
        self.phn2num = phn2num
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.device = device
        self.quiet = quiet

        # Default decode config
        self.decode_config = {
            "codec_sr": codec_sr,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "temperature": temperature,
            "stop_repetition": stop_repetition,
            "kvcache": kvcache,
            "silence_tokens": silence_tokens,
            "sample_batch_size": 1,
        }

    @torch.no_grad()
    def generate(
        self,
        audio_fn,
        target_text,
        prompt_end_frame,
        target_generation_length,
        delay_pattern_increment,
        prefix_transcript=None,
        repeat_prompt=0,
        decode_config=None,
        multi_trial=[],
    ):
        """
        Generate speech for the given text using the reference audio.

        Args:
            audio_fn: Path to reference audio file
            target_text: Text to synthesize
            prompt_end_frame: Number of frames to use from reference audio
            target_generation_length: Target length of generated audio in seconds
            delay_pattern_increment: Delay pattern increment
            prefix_transcript: Optional transcript of the reference audio
            repeat_prompt: Number of times to repeat the prompt (or "max")
            decode_config: Optional custom decoding configuration
            multi_trial: List for multi-trial inference (usually empty)

        Returns:
            tuple: (concatenated_sample, generated_sample)
        """
        # Use custom decode config if provided, otherwise use default
        if decode_config is None:
            decode_config = self.decode_config

        # encode audio
        encoded_frames = tokenize_audio(
            self.audio_tokenizer, audio_fn, offset=0, num_frames=prompt_end_frame
        )
        single_encoded_frames = encoded_frames

        if isinstance(repeat_prompt, int) and repeat_prompt > 0:
            cur_repeat_prompt = repeat_prompt
            while cur_repeat_prompt > 0:
                encoded_frames = torch.cat(
                    [encoded_frames, single_encoded_frames], dim=2
                )
                cur_repeat_prompt -= 1
        elif isinstance(repeat_prompt, str) and repeat_prompt.lower() == "max":
            repeat_prompt = 0
            while (
                encoded_frames.shape[2]
                + decode_config["codec_sr"] * target_generation_length
                + delay_pattern_increment
                + single_encoded_frames.shape[2]
                < self.model_args.audio_max_length * decode_config["codec_sr"]
            ):
                encoded_frames = torch.cat(
                    [encoded_frames, single_encoded_frames], dim=2
                )
                repeat_prompt += 1
        if getattr(self.model_args, "y_sep_token", None) != None:
            encoded_frames = torch.cat(
                [
                    encoded_frames,
                    torch.LongTensor(
                        [self.model_args.y_sep_token] * self.model_args.n_codebooks
                    )
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .to(encoded_frames.device),
                ],
                dim=2,
            )
        original_audio = encoded_frames.transpose(2, 1)  # [1,T,K]
        assert (
            original_audio.ndim == 3
            and original_audio.shape[0] == 1
            and original_audio.shape[2] == self.model_args.n_codebooks
        ), original_audio.shape

        # phonemize
        if isinstance(target_text, list):
            text_tokens = [
                self.phn2num[phn] for phn in target_text if phn in self.phn2num
            ]
        else:
            text_tokens = [
                self.phn2num[phn]
                for phn in tokenize_text(self.text_tokenizer, text=target_text.strip())
                if phn in self.phn2num
            ]
        if getattr(self.model_args, "x_sep_token", None) != None:
            assert (
                prefix_transcript != None
            ), "prefix_transcript must be provided if x_sep_token is not None"
        if prefix_transcript is not None:
            if isinstance(prefix_transcript, list):
                prefix_tokens = [
                    self.phn2num[phn]
                    for phn in prefix_transcript
                    if phn in self.phn2num
                ]
            else:
                prefix_tokens = [
                    self.phn2num[phn]
                    for phn in tokenize_text(
                        self.text_tokenizer, text=prefix_transcript.strip()
                    )
                    if phn in self.phn2num
                ]
            single_prefix_tokens = prefix_tokens
            repeat_prompt_count = repeat_prompt
            while repeat_prompt_count > 0:
                prefix_tokens = prefix_tokens + single_prefix_tokens
                repeat_prompt_count -= 1
            if getattr(self.model_args, "x_sep_token", None) != None:
                text_tokens = (
                    prefix_tokens
                    + [getattr(self.model_args, "x_sep_token", None)]
                    + text_tokens
                )
            else:
                text_tokens = prefix_tokens + text_tokens
        if getattr(self.model_args, "add_eos_to_text", 0) != 0:
            text_tokens.append(self.model_args.add_eos_to_text)
        if getattr(self.model_args, "add_bos_to_text", 0) != 0:
            text_tokens = [self.model_args.add_bos_to_text] + text_tokens
        text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
        text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

        if not self.quiet:
            logging.info(
                f"original audio length: {original_audio.shape[1]} codec frames, which is {original_audio.shape[1]/decode_config['codec_sr']:.2f} sec."
            )

        if getattr(self.model_args, "parallel_pattern", 0) != 0:
            tgt_y_lens = torch.LongTensor(
                [
                    int(
                        original_audio.shape[1]
                        + decode_config["codec_sr"] * target_generation_length
                        + 2
                    )
                ]
            )  # parallel pattern, therefore only add the empty_token (i.e. the sos token) and eos (i.e. 2 more tokens). Note that the delayed pattern between, both sos and eos is counted (sos is counted in the n_codebooks, eos is counted in the 1)
        else:
            tgt_y_lens = torch.LongTensor(
                [
                    int(
                        original_audio.shape[1]
                        + decode_config["codec_sr"] * target_generation_length
                        + delay_pattern_increment
                    )
                ]
            )  # delay pattern increment has accounted for the added eos

        # forward
        assert decode_config["sample_batch_size"] <= 1
        stime = time.time()
        assert multi_trial == []
        if not self.quiet:
            logging.info(f"running inference with batch size 1")
        concat_frames, gen_frames = self.model.inference_tts(
            text_tokens.to(self.device),
            text_tokens_lens.to(self.device),
            original_audio[..., : self.model_args.n_codebooks].to(
                self.device
            ),  # [1,T,8]
            tgt_y_lens=tgt_y_lens.to(self.device),
            top_k=decode_config["top_k"],
            top_p=decode_config["top_p"],
            min_p=decode_config["min_p"],
            temperature=decode_config["temperature"],
            stop_repetition=decode_config["stop_repetition"],
            kvcache=decode_config["kvcache"],
            silence_tokens=(
                eval(decode_config["silence_tokens"])
                if type(decode_config["silence_tokens"]) == str
                else decode_config["silence_tokens"]
            ),
        )  # output is [1,K,T]
        if not self.quiet:
            logging.info(
                f"inference on one sample take: {time.time() - stime:.4f} sec."
            )
            logging.info(
                f"generated encoded_frames.shape: {gen_frames.shape}, which is {gen_frames.shape[-1]/decode_config['codec_sr']} sec."
            )

        if getattr(self.model_args, "y_sep_token", None) != None:
            concat_frames = torch.cat(
                [
                    concat_frames[:, :, : original_audio.shape[1] - 1],
                    concat_frames[:, :, original_audio.shape[1] :],
                ],
                dim=2,
            )
        # Handle MPS device compatibility
        if self.device == "mps":
            # Move tensors to CPU before decoding to avoid MPS placeholder storage error
            concat_frames = concat_frames.cpu()
            gen_frames = gen_frames.cpu()
        concat_sample = self.audio_tokenizer.decode(concat_frames)  # [1,8,T]
        gen_sample = self.audio_tokenizer.decode(gen_frames)

        # Empty cuda cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return concat_sample, gen_sample


# this script only works for the musicgen architecture
def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--manifest_fn", type=str, default="path/to/eval_metadata_file")
    parser.add_argument("--audio_root", type=str, default="path/to/audio_folder")
    parser.add_argument("--exp_dir", type=str, default="path/to/model_folder")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--codec_audio_sr",
        type=int,
        default=16000,
        help="the sample rate of audio that the codec is trained for",
    )
    parser.add_argument(
        "--codec_sr", type=int, default=50, help="the sample rate of the codec codes"
    )
    parser.add_argument("--top_k", type=int, default=0, help="sampling param")
    parser.add_argument("--top_p", type=float, default=0.8, help="sampling param")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling param")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--signature", type=str, default=None, help="path to the encodec model"
    )
    parser.add_argument("--crop_concat", type=int, default=0)
    parser.add_argument(
        "--stop_repetition",
        type=int,
        default=-1,
        help="used for inference, when the number of consecutive repetition of a token is bigger than this, stop it",
    )
    parser.add_argument(
        "--kvcache",
        type=int,
        default=1,
        help="if true, use kv cache, which is 4-8x faster than without",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="batch size for sampling, NOTE that it's not running inference for several samples, but duplicate one input sample batch_size times, and during inference, we only return the shortest generation",
    )
    parser.add_argument(
        "--silence_tokens",
        type=str,
        default="[1388,1898,131]",
        help="note that if you are not using the pretrained encodec 6f79c6a8, make sure you specified it yourself, rather than using the default",
    )
    return parser.parse_args()


@torch.no_grad()
def inference_one_sample(
    model,
    model_args,
    phn2num,
    text_tokenizer,
    audio_tokenizer,
    audio_fn,
    target_text,
    device,
    decode_config,
    prompt_end_frame,
    target_generation_length,
    delay_pattern_increment,
    prefix_transcript=None,
    quiet=False,
    repeat_prompt=0,
    multi_trial=[],
):
    """
    Backward compatibility function that uses the VoiceStarAPI class.

    This function has the same signature as the original inference_one_sample
    but internally uses the new VoiceStarAPI class.
    """
    # Create API instance
    api = VoiceStarAPI(
        model=model,
        model_args=model_args,
        phn2num=phn2num,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        device=device,
        codec_sr=decode_config.get("codec_sr", 50),
        top_k=decode_config.get("top_k", 0),
        top_p=decode_config.get("top_p", 0.8),
        min_p=decode_config.get("min_p", 0.0),
        temperature=decode_config.get("temperature", 1.0),
        stop_repetition=decode_config.get("stop_repetition", -1),
        kvcache=decode_config.get("kvcache", 1),
        silence_tokens=decode_config.get("silence_tokens", "[1388,1898,131]"),
        quiet=quiet,
    )

    # Call the generate method
    return api.generate(
        audio_fn=audio_fn,
        target_text=target_text,
        prompt_end_frame=prompt_end_frame,
        target_generation_length=target_generation_length,
        delay_pattern_increment=delay_pattern_increment,
        prefix_transcript=prefix_transcript,
        repeat_prompt=repeat_prompt,
        decode_config=decode_config,
        multi_trial=multi_trial,
    )
