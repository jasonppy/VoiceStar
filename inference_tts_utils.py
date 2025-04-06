import argparse, pickle
import logging
import os, random
import numpy as np
import torch
import torchaudio

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text
)
import argparse, time, tqdm


# this script only works for the musicgen architecture
def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest_fn", type=str, default="path/to/eval_metadata_file")
    parser.add_argument("--audio_root", type=str, default="path/to/audio_folder")
    parser.add_argument("--exp_dir", type=str, default="path/to/model_folder")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--codec_audio_sr", type=int, default=16000, help='the sample rate of audio that the codec is trained for')
    parser.add_argument("--codec_sr", type=int, default=50, help='the sample rate of the codec codes')
    parser.add_argument("--top_k", type=int, default=0, help="sampling param")
    parser.add_argument("--top_p", type=float, default=0.8, help="sampling param")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling param")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--signature", type=str, default=None, help="path to the encodec model")
    parser.add_argument("--crop_concat", type=int, default=0)
    parser.add_argument("--stop_repetition", type=int, default=-1, help="used for inference, when the number of consecutive repetition of a token is bigger than this, stop it")
    parser.add_argument("--kvcache", type=int, default=1, help='if true, use kv cache, which is 4-8x faster than without')
    parser.add_argument("--sample_batch_size", type=int, default=1, help="batch size for sampling, NOTE that it's not running inference for several samples, but duplicate one input sample batch_size times, and during inference, we only return the shortest generation")
    parser.add_argument("--silence_tokens", type=str, default="[1388,1898,131]", help="note that if you are not using the pretrained encodec 6f79c6a8, make sure you specified it yourself, rather than using the default")
    return parser.parse_args()


@torch.no_grad()
def inference_one_sample(model, model_args, phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_text, device, decode_config, prompt_end_frame, target_generation_length, delay_pattern_increment, prefix_transcript=None, quiet=False, repeat_prompt=0, multi_trial=[]):
    # seq_len_thres = 500 # 10s, 26% of the data in seed tts
    # encode audio
    encoded_frames = tokenize_audio(audio_tokenizer, audio_fn, offset=0, num_frames=prompt_end_frame)
    # if sequence length is shorter than seq_len_thres, repeat the audio
    # if encoded_frames.shape[2] < seq_len_thres:
    #     encoded_frames = torch.cat([encoded_frames, encoded_frames, encoded_frames], dim=2)
    #     doubled = True
    single_encoded_frames = encoded_frames

    if isinstance(repeat_prompt, int) and repeat_prompt > 0:
        cur_repeat_prompt = repeat_prompt
        while cur_repeat_prompt > 0:
            encoded_frames = torch.cat([encoded_frames, single_encoded_frames], dim=2)
            cur_repeat_prompt -= 1
    elif isinstance(repeat_prompt, str) and repeat_prompt.lower() == "max":
        repeat_prompt = 0
        while encoded_frames.shape[2] + decode_config['codec_sr'] * target_generation_length + delay_pattern_increment + single_encoded_frames.shape[2] < model_args.audio_max_length * decode_config['codec_sr']:
            encoded_frames = torch.cat([encoded_frames, single_encoded_frames], dim=2)
            repeat_prompt += 1
    if getattr(model_args, "y_sep_token", None) != None:
        encoded_frames = torch.cat([encoded_frames, torch.LongTensor([model_args.y_sep_token]*model_args.n_codebooks).unsqueeze(0).unsqueeze(2).to(encoded_frames.device)], dim=2)
    # print(encoded_frames.shape)
    original_audio = encoded_frames.transpose(2,1) # [1,T,K]
    assert original_audio.ndim==3 and original_audio.shape[0] == 1 and original_audio.shape[2] == model_args.n_codebooks, original_audio.shape

    # phonemize
    if isinstance(target_text, list):
        text_tokens = [phn2num[phn] for phn in target_text if phn in phn2num]
    else:
        text_tokens = [phn2num[phn] for phn in
                tokenize_text(
                    text_tokenizer, text=target_text.strip()
                ) if phn in phn2num
            ]
    if getattr(model_args, "x_sep_token", None) != None:
        assert prefix_transcript != None, "prefix_transcript must be provided if x_sep_token is not None"
    if prefix_transcript is not None:
        if isinstance(prefix_transcript, list):
            prefix_tokens = [phn2num[phn] for phn in prefix_transcript if phn in phn2num]
        else:
            prefix_tokens = [phn2num[phn] for phn in
                tokenize_text(
                    text_tokenizer, text=prefix_transcript.strip()
                ) if phn in phn2num
            ]
        # if doubled:
        #     prefix_tokens = prefix_tokens + prefix_tokens + prefix_tokens
        single_prefix_tokens = prefix_tokens
        while repeat_prompt > 0:
            prefix_tokens = prefix_tokens + single_prefix_tokens
            repeat_prompt -= 1
        if getattr(model_args, "x_sep_token", None) != None:
            text_tokens = prefix_tokens + [getattr(model_args, "x_sep_token", None)] + text_tokens
        else:
            text_tokens = prefix_tokens + text_tokens
    if getattr(model_args, "add_eos_to_text", 0) != 0:
        text_tokens.append(model_args.add_eos_to_text)
    if getattr(model_args, "add_bos_to_text", 0) != 0:
        text_tokens = [model_args.add_bos_to_text] + text_tokens
    text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

    if not quiet:
        logging.info(f"original audio length: {original_audio.shape[1]} codec frames, which is {original_audio.shape[1]/decode_config['codec_sr']:.2f} sec.")


    if getattr(model_args, "parallel_pattern", 0) != 0:
        tgt_y_lens = torch.LongTensor([int(original_audio.shape[1] + decode_config['codec_sr'] * target_generation_length + 2)]) # parallel pattern, therefore only add the empty_token (i.e. the sos token) and eos (i.e. 2 more tokens). Note that the delayed pattern between, both sos and eos is counted (sos is counted in the n_codebooks, eos is counted in the 1)
    else:
        tgt_y_lens = torch.LongTensor([int(original_audio.shape[1] + decode_config['codec_sr'] * target_generation_length + delay_pattern_increment)]) # delay pattern increment has accounted for the added eos

    # forward
    assert decode_config['sample_batch_size'] <= 1
    stime = time.time()
    assert multi_trial == [] 
    if not quiet:
        logging.info(f"running inference with batch size 1")
    concat_frames, gen_frames = model.inference_tts(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        original_audio[...,:model_args.n_codebooks].to(device), # [1,T,8]
        tgt_y_lens = tgt_y_lens.to(device),
        top_k=decode_config['top_k'],
        top_p=decode_config['top_p'],
        min_p=decode_config['min_p'],
        temperature=decode_config['temperature'],
        stop_repetition=decode_config['stop_repetition'],
        kvcache=decode_config['kvcache'],
        silence_tokens=eval(decode_config['silence_tokens']) if type(decode_config['silence_tokens'])==str else decode_config['silence_tokens']
    ) # output is [1,K,T]
    if not quiet:
        logging.info(f"inference on one sample take: {time.time() - stime:.4f} sec.")

        logging.info(f"generated encoded_frames.shape: {gen_frames.shape}, which is {gen_frames.shape[-1]/decode_config['codec_sr']} sec.")
    
    # for timestamp, codes in enumerate(gen_frames[0].transpose(1,0)):
    #     logging.info(f"{timestamp}: {codes.tolist()}")
    # decode (both original and generated)
    # concat_sample = audio_tokenizer.decode(
    #     [(concat_frames, None)] # [1,T,8] -> [1,8,T]
    # )
    if getattr(model_args, "y_sep_token", None) != None:
        concat_frames = torch.cat([concat_frames[:, :, :original_audio.shape[1]-1], concat_frames[:, :, original_audio.shape[1]:]], dim=2)
    concat_sample = audio_tokenizer.decode(
        concat_frames # [1,8,T]
    )
    gen_sample = audio_tokenizer.decode(
        gen_frames
    )
    #Empty cuda cache between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # return
    return concat_sample, gen_sample