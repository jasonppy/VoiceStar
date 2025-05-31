# VoiceStar: Robust, Duration-controllable TTS that can Extrapolate
[![Paper](https://img.shields.io/badge/arXiv-2505.19462-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2505.19462.pdf) [![YouTube demo](https://img.shields.io/youtube/views/eikybOi8iwU)](https://youtu.be/rTJeabxUxJ4)  [![Demo page](https://img.shields.io/badge/Audio_Samples-blue?logo=Github&style=flat-square)](https://jasonppy.github.io/VoiceCraft_web/)

## 1. Env setup
### Download model
```bash
# under VoiceStar root dir
wget -O ./pretrained/encodec_6f79c6a8.th https://huggingface.co/pyp1/Encodec_VoiceStar/resolve/main/encodec_4cb2048_giga.th?download=true
wget -O ./pretrained/VoiceStar_840M_30s.pth https://huggingface.co/pyp1/VoiceStar/resolve/main/VoiceStar_840M_30s.pth?download=true
wget -O ./pretrained/VoiceStar_840M_40s.pth https://huggingface.co/pyp1/VoiceStar/resolve/main/VoiceStar_840M_40s.pth?download=true
```
### Inference only:
```bash
conda create -n voicestar python=3.10
conda activate voicestar # this seems to lead to much worse results in terms of wer and spksim (comparing e9_rerun and e9_rerun_newba_upgraded)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 
pip install numpy, tqdm, fire
pip install phonemizer==3.2.1
apt-get install espeak-ng # backend for the phonemizer
pip install torchmetrics
pip install einops
pip install omegaconf==2.3.0
pip install openai-whisper
pip install gradio
```

* avoid warnings likes
[WARNING] words_mismatch.py:88 || words count mismatch on 200.0% of the lines (2/1)
```python
# go to ~/miniconda3/envs/voicestar/lib/python3.10/site-packages/phonemizer/backend/espeak/words_mismatch.py
# pass the warning like this
    def _resume(self, nmismatch: int, nlines: int):
        """Logs a high level undetailed warning"""
        pass
        # if nmismatch:
        #     self._logger.warning(
        #         'words count mismatch on %s%% of the lines (%s/%s)',
        #         round(nmismatch / nlines, 2) * 100, nmismatch, nlines)
```

### Training and data processing
*additional packages*:
```bash
pip install huggingface_hub
pip install datasets
pip install tensorboard
pip install wandb
pip install matplotlib
pip install ffmpeg-python
pip install scipy
pip install soundfile
```

## 2. example 
### command line example
check signature of `run_inference` func in `inference_commandline.py` for adjustable hyperparameters
```bash
# under root dir
conda activate voicestar
python inference_commandline.py \
  --reference_speech "./demo/5895_34622_000026_000002.wav" \
  --target_text "I cannot believe that the same model can also do text to speech synthesis too! And you know what? this audio is 8 seconds long." \
  --target_duration 8
```

### Gradio
```bash
conda activate voicestar
python inference_gradio.py
```


## License
Code license: MIT

Model Weights License: CC-BY-4.0 (as Emilia dataset we used is under this license)