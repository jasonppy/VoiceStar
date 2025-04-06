#!/usr/bin/env python3
"""
gradio_tts_app.py

Run:
    python gradio_tts_app.py

Then open the printed local or public URL in your browser.
"""

import os
import random
import numpy as np
import torch
import torchaudio
import whisper
import gradio as gr
from argparse import Namespace

# ---------------------------------------------------------------------
# The following imports assume your local project structure:
#    data/tokenizer.py
#    models/voice_star.py
#    inference_tts_utils.py
# Adjust if needed.
# ---------------------------------------------------------------------
from data.tokenizer import AudioTokenizer, TextTokenizer
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
    # User-adjustable parameters (no "# do not change" in snippet)
    reference_speech="./demo/5895_34622_000026_000002.wav",
    target_text="VoiceStar is a very interesting model, it's duration controllable and can extrapolate",
    model_name="VoiceStar_840M_30s",
    model_root="./pretrained",
    reference_text=None,    # optional
    target_duration=None,   # optional
    top_k=10,               # can try 10, 20, 30, 40
    temperature=1,
    kvcache=1,              # if OOM, set to 0
    repeat_prompt=1,        # use higher to improve speaker similarity
    stop_repetition=3,      # snippet says "will not use it" but not "do not change"
    seed=1,
    output_dir="./generated_tts",

    # Non-adjustable parameters (based on snippet instructions)
    codec_audio_sr=16000,   # do not change
    codec_sr=50,            # do not change
    top_p=1,                # do not change
    min_p=1,                # do not change
    silence_tokens=None,    # do not change it
    multi_trial=None,       # do not change it
    sample_batch_size=1,    # do not change
    cut_off_sec=100,        # do not adjust
):
    """
    Inference script for VoiceStar TTS.
    """
    # 1. Set seed
    seed_everything(seed)

    # 2. Load model checkpoint
    torch.serialization.add_safe_globals([Namespace])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_fn = os.path.join(model_root, model_name + ".pth")
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

    # 3. If reference_text not provided, transcribe reference speech with Whisper
    if reference_text is None:
        print("[Info] No reference_text provided. Transcribing reference_speech with Whisper (large-v3-turbo).")
        wh_model = whisper.load_model("large-v3-turbo")
        result = wh_model.transcribe(reference_speech)
        prefix_transcript = result["text"]
        print(f"[Info] Whisper transcribed text: {prefix_transcript}")
    else:
        prefix_transcript = reference_text

    # 4. If target_duration not provided, estimate from reference speech + target_text
    if target_duration is None:
        target_generation_length = estimate_duration(reference_speech, target_text)
        print(f"[Info] target_duration not provided, estimated as {target_generation_length:.2f}s. Provide --target_duration if needed.")
    else:
        target_generation_length = float(target_duration)

    # 5. Prepare signature from snippet
    if args.n_codebooks == 4:
        signature = "./pretrained/encodec_6f79c6a8.th"
    elif args.n_codebooks == 8:
        signature = "./pretrained/encodec_8cb1024_giga.th"
    else:
        signature = "./pretrained/encodec_6f79c6a8.th"

    if silence_tokens is None:
        silence_tokens = []

    if multi_trial is None:
        multi_trial = []

    delay_pattern_increment = args.n_codebooks + 1  # from snippet

    info = torchaudio.info(reference_speech)
    prompt_end_frame = int(cut_off_sec * info.sample_rate)

    # 6. Tokenizers
    audio_tokenizer = AudioTokenizer(signature=signature)
    text_tokenizer = TextTokenizer(backend="espeak")

    # 7. decode_config
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

    # 8. Run inference
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

    # 9. Save generated audio
    os.makedirs(output_dir, exist_ok=True)
    out_filename = "generated.wav"
    out_path = os.path.join(output_dir, out_filename)
    torchaudio.save(out_path, gen_audio, codec_audio_sr)

    print(f"[Success] Generated audio saved to {out_path}")
    return out_path  # Return the path for Gradio to load


############################
# Transcription function
############################

def transcribe_audio(reference_speech):
    """
    Transcribe uploaded reference audio with Whisper, return text.
    If no file, return empty string.
    """
    if reference_speech is None:
        return ""
    audio_path = reference_speech  # Because type="filepath"

    if not os.path.exists(audio_path):
        return "File not found."

    print("[Info] Transcribing with Whisper...")
    model = whisper.load_model("medium")  # or "large-v2" etc.
    result = model.transcribe(audio_path)
    return result["text"]

############################
# Gradio UI
############################

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## VoiceStar TTS with Editable Reference Text")

        with gr.Row():
            reference_speech_input = gr.Audio(
                label="Reference Speech",
                type="filepath",
                elem_id="ref_speech"
            )
            transcribe_button = gr.Button("Transcribe")

        # The transcribed text appears here and can be edited
        reference_text_box = gr.Textbox(
            label="Reference Text (Editable)",
            placeholder="Click 'Transcribe' to auto-fill from reference speech...",
            lines=2
        )

        target_text_box = gr.Textbox(
            label="Target Text",
            value="VoiceStar is a very interesting model, it's duration controllable and can extrapolate to unseen duration.",
            lines=3
        )

        model_name_box = gr.Textbox(
            label="Model Name",
            value="VoiceStar_840M_30s"
        )

        model_root_box = gr.Textbox(
            label="Model Root Directory",
            value="/data1/scratch/pyp/BoostedVoiceEditor/runs"
        )

        reference_duration_box = gr.Textbox(
            label="Target Duration (Optional)",
            placeholder="Leave empty for auto-estimate."
        )

        top_k_box = gr.Number(label="top_k", value=10)
        temperature_box = gr.Number(label="temperature", value=1.0)
        kvcache_box = gr.Number(label="kvcache (1 or 0)", value=1)
        repeat_prompt_box = gr.Number(label="repeat_prompt", value=1)
        stop_repetition_box = gr.Number(label="stop_repetition", value=3)
        seed_box = gr.Number(label="Random Seed", value=1)
        output_dir_box = gr.Textbox(label="Output Directory", value="./generated_tts")

        generate_button = gr.Button("Generate TTS")
        output_audio = gr.Audio(label="Generated Audio", type="filepath")

        # 1) When user clicks "Transcribe", we call `transcribe_audio`
        transcribe_button.click(
            fn=transcribe_audio,
            inputs=[reference_speech_input],
            outputs=[reference_text_box],
        )

        # 2) The actual TTS generation function. 
        def gradio_inference(
            reference_speech,
            reference_text,
            target_text,
            model_name,
            model_root,
            target_duration,
            top_k,
            temperature,
            kvcache,
            repeat_prompt,
            stop_repetition,
            seed,
            output_dir
        ):
            # Convert any empty strings to None for optional fields
            dur = float(target_duration) if target_duration else None

            out_path = run_inference(
                reference_speech=reference_speech,
                reference_text=reference_text if reference_text else None,
                target_text=target_text,
                model_name=model_name,
                model_root=model_root,
                target_duration=dur,
                top_k=int(top_k),
                temperature=float(temperature),
                kvcache=int(kvcache),
                repeat_prompt=int(repeat_prompt),
                stop_repetition=int(stop_repetition),
                seed=int(seed),
                output_dir=output_dir
            )
            return out_path

        # 3) Link the "Generate TTS" button
        generate_button.click(
            fn=gradio_inference,
            inputs=[
                reference_speech_input,
                reference_text_box,
                target_text_box,
                model_name_box,
                model_root_box,
                reference_duration_box,
                top_k_box,
                temperature_box,
                kvcache_box,
                repeat_prompt_box,
                stop_repetition_box,
                seed_box,
                output_dir_box
            ],
            outputs=[output_audio],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)

if __name__ == "__main__":
    main()