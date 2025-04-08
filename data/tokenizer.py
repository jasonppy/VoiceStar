# cp from https://github.com/lifeiteng/vall-e/blob/main/valle/data/tokenizer.py
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union
from voicestar.data.encodec import get_compression_model
import numpy as np
import torch
import torchaudio

# from encodec import EncodecModel
# from encodec.utils import convert_audio
# from lhotse.features import FeatureExtractor
# from lhotse.utils import Seconds, compute_num_frames
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator


try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass


class PypinyinBackend:
    """PypinyinBackend for Chinese. Most codes is referenced from espnet.
    There are two types pinyin or initials_finals, one is
    just like "ni1 hao3", the other is like "n i1 h ao3".
    """

    def __init__(
        self,
        backend="initials_finals",
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
    ) -> None:
        self.backend = backend
        self.punctuation_marks = punctuation_marks

    def phonemize(
        self, text: List[str], separator: Separator, strip=True, njobs=1
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        for _text in text:
            _text = re.sub(" +", " ", _text.strip())
            _text = _text.replace(" ", separator.word)
            phones = []
            if self.backend == "pypinyin":
                for n, py in enumerate(
                    pinyin(_text, style=Style.TONE3, neutral_tone_with_five=True)
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)

                        phones.extend(list(py[0]))
                    else:
                        phones.extend([py[0], separator.syllable])
            elif self.backend == "pypinyin_initials_finals":
                for n, py in enumerate(
                    pinyin(_text, style=Style.TONE3, neutral_tone_with_five=True)
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)
                        phones.extend(list(py[0]))
                    else:
                        if py[0][-1].isalnum():
                            initial = get_initials(py[0], strict=False)
                            if py[0][-1].isdigit():
                                final = get_finals(py[0][:-1], strict=False) + py[0][-1]
                            else:
                                final = get_finals(py[0], strict=False)
                            phones.extend(
                                [
                                    initial,
                                    separator.phone,
                                    final,
                                    separator.syllable,
                                ]
                            )
                        else:
                            assert ValueError
            else:
                raise NotImplementedError
            phonemized.append(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}")
            )
        return phonemized


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        if backend == "espeak":
            phonemizer = EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )
        elif backend in ["pypinyin", "pypinyin_initials_finals"]:
            phonemizer = PypinyinBackend(
                backend=backend,
                punctuation_marks=punctuation_marks + separator.word,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone] + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized]


def tokenize_text(tokenizer: TextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0]  # k2symbols


def remove_encodec_weight_norm(model):
    from encodec.modules import SConv1d
    from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)
    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)


class AudioTokenizer:
    """mimi audio."""

    def __init__(
        self,
        bandwidth: float = 6.0,
        device: Any = None,
        hificodec=False,
        signature=None,
        encode_only=False,
    ) -> None:
        self.signature = signature

        model = get_compression_model(signature, encode_only=encode_only, device=device)
        self.sample_rate = model.sample_rate
        self.channels = model.channels

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda")

        self._device = device

        self.codec = model.to(device)

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        if self.signature != None:
            if self.signature == "lfsc":
                if wav.ndim == 3:
                    assert wav.shape[:2] == torch.Size((1, 1)), wav.shape
                    wav = wav.squeeze(0)
                elif wav.ndim == 2:
                    assert wav.shape[0] == 1, wav.shape
                else:
                    raise ValueError(wav.shape)
                audio_len = torch.tensor([wav.shape[1]]).to(self.device)
                codes, encoded_len = self.codec.encode(
                    audio=wav.to(self.device), audio_len=audio_len
                )
                return codes[:, :, : encoded_len[0]]
            else:
                codes = self.codec.encode(wav.to(self.device))
                return codes[0]
        else:
            assert wav.ndim == 3 and wav.shape[:2] == torch.Size((1, 1)), wav.shape
            return self.codec.encode(wav.to(self.device))

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        if self.signature != None and self.signature == "lfsc":
            encoded_len = torch.tensor([frames.shape[-1]]).to(self.device)
            reconstructed_audio, decoded_len = self.codec.decode(
                tokens=frames, tokens_len=encoded_len
            )
            return reconstructed_audio[:, : decoded_len[0]].unsqueeze(0)
        else:
            return self.codec.decode(frames)


def tokenize_audio(
    tokenizer: AudioTokenizer, audio_path: str, offset=-1, num_frames=-1
):
    # Load and pre-process the audio waveform
    if offset != -1 and num_frames != -1:
        wav, sr = torchaudio.load(
            audio_path, frame_offset=offset, num_frames=num_frames
        )
    else:
        wav, sr = torchaudio.load(audio_path)
    if sr != tokenizer.sample_rate:
        wav = torchaudio.transforms.Resample(sr, tokenizer.sample_rate)(wav)
        sr = tokenizer.sample_rate
    if wav.shape[0] == 2:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0)
    # Extract discrete codes from mimi
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames


if __name__ == "__main__":
    # tok = AudioTokenizer(signature="lfsc", device="cpu")
    tok = AudioTokenizer(
        signature="/home/pyp/BoostedVoiceEditor/pretrained/encodec_6f79c6a8.th",
        device="cpu",
    )
    inaudio = "/home/pyp/BoostedVoiceEditor/demo/pam.wav"
    encoded_frames = tokenize_audio(tok, inaudio)
    print(encoded_frames.shape)
    # decode it back
    decoded_audio = tok.decode(encoded_frames)
    torchaudio.save(
        "/home/pyp/BoostedVoiceEditor/demo/pam_reconstructed_encodec_4cb_2nd.wav",
        decoded_audio[0],
        tok.sample_rate,
    )
