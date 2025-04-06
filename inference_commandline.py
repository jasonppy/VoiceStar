import os
import torch
import torchaudio
import numpy as np
import random
import whisper
import fire
from argparse import Namespace

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)

from models import voice_star
from inference_tts_utils import inference_one_sample

############################################################
# Utility Functions
############################################################

def seed_everything(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def estimate_duration(ref_audio_path, text):
    """
    Estimate duration based on seconds per character from the reference audio.
    """
    info = torchaudio.info(ref_audio_path)
    audio_duration = info.num_frames / info.sample_rate
    length_text = max(len(text), 1)
    spc = audio_duration / length_text  # seconds per character
    return len(text) * spc

############################################################
# Main Inference Function
############################################################

def run_inference(
    reference_speech="./demo/5895_34622_000026_000002.wav",
    target_text="I cannot believe that the same model can also do text to speech synthesis too! And you know what? this audio is 8 seconds long.",
    # Model
    model_name="VoiceStar_840M_30s", # or VoiceStar_840M_40s, the later model is trained on maximally 40s long speech
    model_root="./pretrained",
    # Additional optional
    reference_text=None,  # if None => run whisper on reference_speech
    target_duration=None, # if None => estimate from reference_speech and target_text
    # Default hyperparameters from snippet
    codec_audio_sr=16000, # do not change
    codec_sr=50, # do not change
    top_k=20, # try 10, 20, 30, 40
    top_p=1, # do not change
    min_p=1, # do not change
    temperature=1,
    silence_tokens=None, # do not change it
    kvcache=1, # if OOM, set to 0
    multi_trial=None, # do not change it
    repeat_prompt=1, # increase this to improve speaker similarity, but it reference speech duration in total adding target duration is longer than maximal training duration, quality may drop
    stop_repetition=3, # will not use it
    sample_batch_size=1, # do not change
    # Others
    seed=1,
    output_dir="./generated_tts",
    # Some snippet-based defaults
    cut_off_sec=100, # do not adjust this, we always use the entire reference speech. If you wish to change, also make sure to change the reference_transcript, so that it's only the trasnscript of the speech remained
):
    """
    Inference script using Fire.

    Example:
        python inference_commandline.py \
            --reference_speech "./demo/5895_34622_000026_000002.wav" \
            --target_text "I cannot believe ... this audio is 10 seconds long." \
            --reference_text "(optional) text to use as prefix" \
            --target_duration (optional float) 
    """

    # Seed everything
    seed_everything(seed)

    # Load model, phn2num, and args
    torch.serialization.add_safe_globals([Namespace])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_fn = os.path.join(model_root, model_name+".pth")
    if not os.path.exists(ckpt_fn):
        # use wget to download
        print(f"[Info] Downloading {model_name} checkpoint...")
        os.system(f"wget https://huggingface.co/pyp1/VoiceStar/resolve/main/{model_name}.pth?download=true -O {ckpt_fn}")
    bundle = torch.load(ckpt_fn, map_location=device, weights_only=True)
    args = bundle["args"]
    phn2num = bundle["phn2num"]
    model = voice_star.VoiceStar(args)
    model.load_state_dict(bundle["model"])
    model.to(device)
    model.eval()

    # If reference_text not provided, use whisper large-v3-turbo
    if reference_text is None:
        print("[Info] No reference_text provided, transcribing reference_speech with Whisper.")
        wh_model = whisper.load_model("large-v3-turbo")
        result = wh_model.transcribe(reference_speech)
        prefix_transcript = result["text"]
        print(f"[Info] Whisper transcribed text: {prefix_transcript}")
    else:
        prefix_transcript = reference_text

    # If target_duration not provided, estimate from reference speech + target_text
    if target_duration is None:
        target_generation_length = estimate_duration(reference_speech, target_text)
        print(f"[Info] target_duration not provided, estimated as {target_generation_length:.2f} seconds. If not desired, please provide a target_duration.")
    else:
        target_generation_length = float(target_duration)

    # signature from snippet
    if args.n_codebooks == 4:
        signature = "./pretrained/encodec_6f79c6a8.th"
    elif args.n_codebooks == 8:
        signature = "./pretrained/encodec_8cb1024_giga.th"
    else:
        # fallback, just use the 6-f79c6a8
        signature = "./pretrained/encodec_6f79c6a8.th"

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
        'top_k': top_k,
        'top_p': top_p,
        'min_p': min_p,
        'temperature': temperature,
        'stop_repetition': stop_repetition,
        'kvcache': kvcache,
        'codec_audio_sr': codec_audio_sr,
        'codec_sr': codec_sr,
        'silence_tokens': silence_tokens,
        'sample_batch_size': sample_batch_size
    }

    # Run inference
    print("[Info] Running TTS inference...")
    concated_audio, gen_audio = inference_one_sample(
        model, args, phn2num, text_tokenizer, audio_tokenizer,
        reference_speech, target_text,
        device, decode_config,
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
    fire.Fire(run_inference)

if __name__ == "__main__":
    main()
