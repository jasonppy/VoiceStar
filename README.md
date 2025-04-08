# VoiceStar: Robust, Duration-Controllable TTS that can Extrapolate

VoiceStar is a robust, duration-controllable TTS model with support for test-time extrapolation, meaning it can generate speech longer than the duration it was trained on.

## Features

- **Duration control**: Specify the duration of the generated speech.
- **Zero-shot voice cloning**: Clone any voice with a short reference audio clip ([demo video](https://x.com/PuyuanPeng/status/1908822618167300419)).

Coming soon: research paper (ETA: 7 April 2025 - 14 April 2025)

## Quick Start

### Install

```bash
pip install voicestar
```

Make sure you also have `espeak-ng` installed.

**Note:** If you run into issues installing VoiceStar with `uv`, try installing it with `pip` instead.

### Usage

Basic usage:

```bash
voicestar --reference-speech "./demo/5895_34622_000026_000002.wav" --target-text "I cannot believe that the same model can also do text to speech synthesis too! And you know what? this audio is 8 seconds long." --target-duration 8
```

Please refer to the CLI and Python API documentation below for more advanced usage.

## Training

Please refer to the [training docs](docs/training.md) for more information.

## Inference

### CLI

```bash
voicestar --reference-speech "./demo/5895_34622_000026_000002.wav" --target-text "I cannot believe that the same model can also do text to speech synthesis too!"
```

View all available options:

```bash
voicestar --help
```

### Python API

```python
from voicestar import VoiceStar

# Initialize the model
model = VoiceStar()

# Generate speech from text
audio = model.generate("I cannot believe that the same model can also do text to speech synthesis too!")
audio.save("output.wav")
```

## License

The code in this repo is licensed under the MIT license. The pretrained model weights available on Hugging Face are licensed under the CC-BY-4.0 license.

This repository may contain third-party software which may be licensed under different licenses.