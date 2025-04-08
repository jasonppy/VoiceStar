"""
VoiceStar: Robust, Duration-Controllable TTS that can Extrapolate

GitHub: https://github.com/jasonppy/VoiceStar
License: MIT

Copyright (c) 2025 Puyuan Peng
"""
import os
import random
import numpy as np
import torch
import torchaudio

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
