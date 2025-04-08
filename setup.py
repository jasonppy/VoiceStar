from setuptools import setup, find_packages
from voicestar import __version__

setup(
    name="voicestar",
    version=__version__,
    description="VoiceStar: Robust, Duration-Controllable TTS that can Extrapolate",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jasonppy/VoiceStar",
    author="Puyuan Peng",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "numpy",
        "tqdm",
        "fire",
        "phonemizer",
        "torchmetrics",
        "einops",
        "omegaconf==2.3.0",
        "openai-whisper",
        "transformers[torch]",
        "huggingface_hub",
        "gradio",
        "click"
    ],
    extras_require={
        "train": [
            "huggingface_hub",
            "datasets",
            "tensorboard",
            "wandb",
            "matplotlib",
            "ffmpeg-python",
            "scipy",
            "soundfile"
        ]
    },
    entry_points={
        "console_scripts": [
            "voicestar=voicestar.cli:run_inference"
        ]
    }
)