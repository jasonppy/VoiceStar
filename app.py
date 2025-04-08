import gradio as gr
import torch
import os
from voicestar import VoiceStar
from voicestar.utils import seed_everything
from txtsplit import txtsplit
import numpy as np

ABOUT = """
# VoiceStar TTS

Gradio demo for [VoiceStar](https://github.com/jasonppy/VoiceStar): robust, duration-controllable TTS that can extrapolate.
"""

# Initialize model once outside the function for better performance
model = VoiceStar()


def generate_audio(
    reference_speech,
    text,
    duration=0.0,
    top_k=10,
    temperature=1.0,
    repeat_prompt=1,
    seed=1,
    progress=gr.Progress(),
):
    # Set seed for reproducibility if provided
    if seed > 0:
        seed_everything(seed)

    # Update model parameters if needed
    model.api.top_k = top_k
    model.api.temperature = temperature
    model.repeat_prompt = repeat_prompt

    # Generate speech
    target_duration = None if duration <= 0 else duration
    texts = txtsplit(text)

    audios = []
    for t in progress.tqdm(texts):
        audio = model.generate(
            reference_speech=reference_speech, text=t, target_duration=target_duration
        )
        audios.append(audio.waveform.squeeze().numpy())

    audio = np.concatenate(audios)

    # Return audio for gradio
    return (16000, audio)


with gr.Blocks() as demo:
    gr.Markdown(ABOUT)
    inp_ref = gr.Audio(label="Reference Audio", type="filepath")
    inp_text = gr.Textbox(label="Text to synthesize")

    with gr.Accordion("Advanced Settings", open=False):
        inp_reference_text = gr.Textbox(
            label="Reference Text",
            info="Enter a transcription of the reference audio. This is optional - if not provided, the model will transcribe the audio automatically.",
        )
        inp_duration = gr.Number(
            label="Duration",
            info="Set to 0 to automatically estimate duration",
            value=0.0,
        )
        inp_top_k = gr.Slider(label="Top-k", minimum=1, maximum=100, step=1, value=10)
        inp_temp = gr.Slider(
            label="Temperature", minimum=0.0, maximum=2.0, step=0.01, value=1.0
        )
        inp_repeat_prompt = gr.Slider(
            label="Repeat prompt", minimum=1, maximum=10, step=1, value=1
        )
        inp_seed = gr.Number(label="Seed", info="Set to 0 to use random seed", value=1)

    btn_generate = gr.Button("Generate", variant="primary")
    out_audio = gr.Audio(label="Generated Audio")

    btn_generate.click(
        fn=generate_audio,
        inputs=[
            inp_ref,
            inp_text,
            inp_duration,
            inp_top_k,
            inp_temp,
            inp_repeat_prompt,
            inp_seed,
        ],
        outputs=[out_audio],
    )

demo.queue().launch()
