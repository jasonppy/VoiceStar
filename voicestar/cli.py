import os
import torch
import torchaudio
import numpy as np
import random
import whisper
import click
from argparse import Namespace

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)

import voicestar.voicestar as voice_star
from voicestar.api import inference_one_sample
from huggingface_hub import hf_hub_download
from transformers import pipeline

from voicestar.utils import seed_everything, estimate_duration

############################################################
# Main Inference Function
############################################################


@click.command()
@click.option(
    "--reference-speech",
    default="./demo/5895_34622_000026_000002.wav",
    help="Path to reference speech audio file",
)
@click.option(
    "--target-text",
    default="I cannot believe that the same model can also do text to speech synthesis too! And you know what? this audio is 8 seconds long.",
    help="Text to synthesize",
)
@click.option(
    "--model-name",
    default="VoiceStar_840M_30s",
    help="Model name (VoiceStar_840M_30s or VoiceStar_840M_40s)",
)
@click.option(
    "--reference-text",
    default=None,
    help="Reference text (if None, will use Whisper to transcribe)",
)
@click.option(
    "--target-duration",
    default=None,
    type=float,
    help="Target duration in seconds (if None, will estimate)",
)
@click.option(
    "--codec-audio-sr", default=16000, help="Codec audio sample rate (do not change)"
)
@click.option("--codec-sr", default=50, help="Codec sample rate (do not change)")
@click.option(
    "--top-k", default=10, help="Top-k sampling parameter (try 10, 20, 30, 40)"
)
@click.option("--top-p", default=1.0, help="Top-p sampling parameter (do not change)")
@click.option("--min-p", default=1.0, help="Min-p sampling parameter (do not change)")
@click.option("--temperature", default=1.0, help="Sampling temperature")
@click.option("--kvcache", default=1, help="Use KV cache (set to 0 if OOM)")
@click.option(
    "--repeat-prompt", default=1, help="Repeat prompt to improve speaker similarity"
)
@click.option(
    "--stop-repetition", default=3, help="Stop repetition parameter (will not use it)"
)
@click.option(
    "--sample-batch-size", default=1, help="Sample batch size (do not change)"
)
@click.option("--seed", default=1, help="Random seed")
@click.option("--output-dir", default="./generated_tts", help="Output directory")
@click.option("--cut-off-sec", default=100, help="Cut-off seconds (do not adjust)")
def run_inference(
    reference_speech,
    target_text,
    model_name,
    reference_text,
    target_duration,
    codec_audio_sr,
    codec_sr,
    top_k,
    top_p,
    min_p,
    temperature,
    kvcache,
    repeat_prompt,
    stop_repetition,
    sample_batch_size,
    seed,
    output_dir,
    cut_off_sec,
):
    """
    VoiceStar TTS inference CLI.

    Example:
        voicestar --reference-speech "./demo/5895_34622_000026_000002.wav" \
            --target-text "I cannot believe ... this audio is 10 seconds long." \
            --reference-text "Optional text to use as prefix" \
            --target-duration 10.0
    """
    # Default values for parameters not exposed in click options
    silence_tokens = None
    multi_trial = None

    # Seed everything
    seed_everything(seed)

    # Load model, phn2num, and args
    torch.serialization.add_safe_globals([Namespace])
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )  # MPS support
    ckpt_fn = hf_hub_download(repo_id="pyp1/VoiceStar", filename=f"{model_name}.pth")

    bundle = torch.load(ckpt_fn, map_location=device, weights_only=True)
    args = bundle["args"]
    phn2num = bundle["phn2num"]
    model = voice_star.VoiceStarModel(args)
    model.load_state_dict(bundle["model"])
    model.to(device)
    model.eval()

    # If reference_text not provided, use whisper large-v3-turbo
    if reference_text is None:
        print(
            "[Info] No reference_text provided, transcribing reference_speech with Whisper."
        )
        wh_model = whisper.load_model("large-v3-turbo")
        result = wh_model.transcribe(reference_speech)
        prefix_transcript = result["text"]
        print(f"[Info] Whisper transcribed text: {prefix_transcript}")
    else:
        prefix_transcript = reference_text

    # If target_duration not provided, estimate from reference speech + target_text
    if target_duration is None:
        target_generation_length = estimate_duration(reference_speech, target_text)
        print(
            f"[Info] target_duration not provided, estimated as {target_generation_length:.2f} seconds. If not desired, please provide a target_duration."
        )
    else:
        target_generation_length = float(target_duration)

    # signature from snippet
    if args.n_codebooks == 4:
        # signature = "./pretrained/encodec_6f79c6a8.th"
        signature = hf_hub_download(
            repo_id="pyp1/VoiceCraft", filename="encodec_4cb2048_giga.th"
        )  # not sure if this is the right signature
    elif args.n_codebooks == 8:
        signature = hf_hub_download(
            repo_id="pyp1/VoiceCraft", filename="encodec_8cb1024_giga.th"
        )
    else:
        # fallback, just use the 6-f79c6a8
        raise ValueError(f"Invalid number of codebooks: {args.n_codebooks}")
        # not sure where to download 6-f79c6a8 from
        # signature = "./pretrained/encodec_6f79c6a8.th"

    if silence_tokens is None:
        # default from snippet
        silence_tokens = []

    if multi_trial is None:
        # default from snippet
        multi_trial = []

    delay_pattern_increment = args.n_codebooks + 1  # from snippet

    # We can compute prompt_end_frame if we want, from snippet
    info = torchaudio.info(reference_speech)
    prompt_end_frame = int(cut_off_sec * info.sample_rate)

    # Prepare tokenizers
    audio_tokenizer = AudioTokenizer(signature=signature)
    text_tokenizer = TextTokenizer(backend="espeak")

    # decode_config from snippet
    decode_config = {
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "temperature": temperature,
        "stop_repetition": stop_repetition,
        "kvcache": kvcache,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "silence_tokens": silence_tokens,
        "sample_batch_size": sample_batch_size,
    }

    # Run inference
    print("[Info] Running TTS inference...")
    concated_audio, gen_audio = inference_one_sample(
        model,
        args,
        phn2num,
        text_tokenizer,
        audio_tokenizer,
        reference_speech,
        target_text,
        device,
        decode_config,
        prompt_end_frame=prompt_end_frame,
        target_generation_length=target_generation_length,
        delay_pattern_increment=delay_pattern_increment,
        prefix_transcript=prefix_transcript,
        multi_trial=multi_trial,
        repeat_prompt=repeat_prompt,
    )

    # The model returns a list of waveforms, pick the first
    concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()

    # Save the audio (just the generated portion, as the snippet does)
    os.makedirs(output_dir, exist_ok=True)
    out_filename = "generated.wav"
    out_path = os.path.join(output_dir, out_filename)
    torchaudio.save(out_path, gen_audio, codec_audio_sr)

    print(f"[Success] Generated audio saved to {out_path}")


def main():
    run_inference()


if __name__ == "__main__":
    main()
