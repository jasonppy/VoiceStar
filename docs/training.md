# Training

## Setup Environment

First, setup the environment with the inference requirements:

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


Install additional packages required for training and data processing:

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

## Download Models

If you are training, you may need to download models manually:

```bash
# under VoiceStar root dir
mkdir pretrained
wget -O ./pretrained/encodec_6f79c6a8.th "https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th"
wget -O ./pretrained/VoiceStar_840M_30s.pth "https://huggingface.co/pyp1/VoiceStar/resolve/main/VoiceStar_840M_30s.pth"
wget -O ./pretrained/VoiceStar_840M_40s.pth "https://huggingface.co/pyp1/VoiceStar/resolve/main/VoiceStar_840M_40s.pth"
```

## Training

TODO: Finish training docs

Training scripts can be found in `train` folder. The data processing scripts can be found in `data` folder. An example training script can be found in `scripts/e1_840M_30s.sh`.