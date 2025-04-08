import os
import ffmpeg
import torch
import random
import copy
import logging
import torch.distributed as dist
import shutil
import csv
import torchaudio
import glob
import numpy as np
from data.tokenizer import TextTokenizer, tokenize_text, AudioTokenizer


def find_files(root_dir, endswith=".wav"):
    files = []
    # os.walk generates the file names in a directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # os.path.splitext splits the file name into a base and extension
            # base, ext = os.path.splitext(filename)
            if filename.lower().endswith(endswith):
                # os.path.join combines one or more path names into a single path
                full_path = os.path.join(dirpath, filename)
                files.append(full_path)
    return files


class dataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.args.target_time_stretch_prob = getattr(
            self.args, "target_time_stretch_prob", 0
        )
        self.args.target_time_stretch_bound = getattr(
            self.args, "target_time_stretch_bound", 0.1
        )
        self.split = split

        assert self.split in [
            "train",
            "valid",
            "test",
        ], f"split should be one of ['train', 'valid', 'test'], but it's {split}"

        if "[" not in self.args.dataset_dir or "]" not in self.args.dataset_dir:
            self.dataset_dir = f"['{self.args.dataset_dir}']"
        else:
            self.dataset_dir = copy.deepcopy(self.args.dataset_dir)
        self.dataset_dir = eval(self.dataset_dir)
        data = []
        if "[" not in self.args.manifest_name or "]" not in self.args.manifest_name:
            self.args.manifest_name = f"['{self.args.manifest_name}']"
        else:
            self.args.manifest_name = copy.deepcopy(self.args.manifest_name)
        self.manifest_name = eval(self.args.manifest_name)
        if len(self.manifest_name) != len(self.dataset_dir):
            assert (
                len(self.manifest_name) == 1
            ), f"len(self.manifest_name) should be 1 or equal to len(self.dataset_dir), but it's {len(self.manifest_name)}"
            self.manifest_name = self.manifest_name * len(self.dataset_dir)
        for i_data, dataset_dir in enumerate(self.dataset_dir):
            if (
                getattr(self.args, "no_libri_in_training", None) != None
                and ("librilight" in dataset_dir)
                and self.split == "train"
            ):
                if not dist.is_initialized() or dist.get_rank() == 0:
                    logging.info(f"skipping librilight in training split")
                continue
            n_datapoints = 0
            manifest_fn = os.path.join(
                dataset_dir, self.manifest_name[i_data], self.split + ".txt"
            )
            if not os.path.isfile(manifest_fn):
                all_manifest_fn = glob.glob(manifest_fn.replace(".txt", "_*=*.txt"))
                if len(all_manifest_fn) == 0:
                    logging.info(
                        f"no manifest file found for {split} split in {dataset_dir}"
                    )
                    continue
                if self.args.debug:
                    logging.info(
                        f"debugging mode, only using the frist found manifest file: {all_manifest_fn[0]}"
                    )
                    all_manifest_fn = all_manifest_fn[:1]
                else:
                    if dist.is_initialized() and dist.get_rank() == 0:
                        logging.info(
                            f"Combining found manifest files for {split}: {all_manifest_fn}"
                        )
                for cur_manifest_fn in all_manifest_fn:
                    with open(cur_manifest_fn, "r") as rf:
                        tmp = [
                            l.strip().split("\t") + [i_data] for l in rf.readlines()
                        ]  # i_data is the index of the dataset
                        n_datapoints += len(tmp)
                        data += tmp
            else:
                with open(manifest_fn, "r") as rf:
                    tmp = [l.strip().split("\t") + [i_data] for l in rf.readlines()]
                    data += tmp
                    n_datapoints += len(tmp)
            if dist.is_initialized() and dist.get_rank() == 0:
                logging.info(
                    f"number of data points for {split} split in {dataset_dir}: {n_datapoints}"
                )
        assert len(data) > 0, f"no data found for {split} split"
        lengths_list = [
            int(item[1]) for item in data
        ]  # use 1 because there might be more than 1 columns (for gigaspeech we have 3 columns: path, duration, selfsim)
        self.data = []
        self.lengths_list = []
        total_duration = 0
        for d, l in zip(data, lengths_list):
            if l >= self.args.encodec_sr * self.args.audio_min_length:
                if (
                    self.args.drop_long
                    and l > self.args.encodec_sr * self.args.audio_max_length
                ):
                    continue
                self.data.append(d)
                self.lengths_list.append(l)
                total_duration += l / self.args.encodec_sr / 3600
        # logging.info(f"for now cut the dataset to only have 500 examples for debugging")
        # self.data = self.data[:1000]
        # self.lengths_list = self.lengths_list[:1000]
        if dist.is_initialized() and dist.get_rank() == 0:
            logging.info(
                f"TOTAL number of data points for {self.split} split: {len(self.lengths_list)}"
            )
            logging.info(
                f"TOTAL duration for {self.split} split: {total_duration:.1f} hours"
            )
        # phoneme vocabulary
        phn_set = set()
        for dataset_dir in self.dataset_dir:
            vocab_fn = os.path.join(dataset_dir, "vocab.txt")
            with open(vocab_fn, "r") as f:
                temp = [l.strip().split("\t") for l in f.readlines() if len(l) != 0]
                phn_set.update([item[-1] for item in temp])
        self.phn2num = {item: i for i, item in enumerate(phn_set)}
        assert self.args.text_vocab_size > len(
            self.phn2num
        ), f"need self.args.text_vocab_size to be bigger than number of phns in vocab to handle OOD phn, but the former is {self.args.text_vocab_size} while the latter is {len(self.phn2num)}"

        if (
            self.args.neighbor_prompt_prob > 0 and self.args.time_stretch_prob > 0
        ) or self.args.target_time_stretch_prob > 0:
            userdir = os.path.expanduser("~")
            encodec_signature = getattr(
                self.args,
                "encodec_signature",
                os.path.join(userdir, "VoiceStar", "pretrained", "encodec_6f79c6a8.th"),
            )
            self.audio_tokenizer = AudioTokenizer(
                signature=encodec_signature,
                device=torch.device("cpu"),
                encode_only=True,
            )
            assert (
                self.audio_tokenizer.sample_rate == self.args.codec_audio_sr
            ), f"audio_tokenizer.sample_rate: {self.audio_tokenizer.sample_rate}, self.args.encodec_sr: {self.args.encodec_sr}"
            if dist.is_initialized() and dist.get_rank() == 0:
                logging.info(
                    f"rank: {dist.get_rank()}, audio_tokenizer device: {self.audio_tokenizer._device}"
                )

    def __len__(self):
        return len(self.lengths_list)

    def _load_phn_enc(self, index):
        item = self.data[index]
        dataset_dir = self.dataset_dir[item[-1]]
        pf = os.path.join(dataset_dir, self.args.phn_folder_name, item[0] + ".txt")
        ef = os.path.join(dataset_dir, self.args.encodec_folder_name, item[0] + ".txt")
        # with certain probability, we load the audio, and time stretch it, note that we should not hit self.args.audio_max_length
        if "/librilight" in dataset_dir:
            audio_ext = ".flac"
        elif "/emilia" in dataset_dir:
            audio_ext = ".mp3"
        else:
            raise NotImplementedError(f"dataset_dir: {dataset_dir}")

        audio_fn = os.path.join(
            dataset_dir,
            self.args.audio_folder_name,
            item[0].replace(".txt", "") + audio_ext,
        )
        speed_factor = (
            random.uniform(
                -self.args.target_time_stretch_bound,
                self.args.target_time_stretch_bound,
            )
            + 1
        )
        length_ok = (
            float(item[1]) / self.args.encodec_sr
        ) / speed_factor < self.args.audio_max_length  # NOTE to calculate the maximal duration after time stretching, we should be used as orig/(1-bound), rather than orig*(1+bound)
        if (
            self.args.target_time_stretch_prob > 0
            and random.random() < self.args.target_time_stretch_prob
            and os.path.isfile(audio_fn)
            and length_ok
        ):
            try:
                with open(pf, "r") as p:
                    phns = [l.strip() for l in p.readlines()]
                    assert len(phns) == 1, phns
                    all_phns = phns[0].split(" ")
                    x = [
                        self.phn2num[item] for item in all_phns if item in self.phn2num
                    ]
            except:
                logging.info(
                    f"loading failed for {pf}, maybe files don't exist or are corrupted"
                )
                return [], [[]], dataset_dir, audio_ext
            # time stretch
            try:
                process = (
                    ffmpeg.input(
                        audio_fn, ss=0, t=float(item[1]) / self.args.encodec_sr
                    )
                    .output(
                        "pipe:1",
                        format="f32le",
                        ac=1,
                        ar=self.audio_tokenizer.sample_rate,
                        filter="atempo={}".format(speed_factor),
                    )
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
                # Read the processed audio from ffmpeg stdout
                output, _ = process.communicate()

                # Convert the output to a numpy array
                output_np = np.frombuffer(output, dtype=np.float32).copy()

                # Reshape the numpy array back to the expected shape (1, samples for mono)
                waveform = torch.from_numpy(output_np)
                waveform = waveform.unsqueeze(0).unsqueeze(0)
                assert (
                    waveform.ndim == 3
                    and waveform.shape[0] == 1
                    and waveform.shape[1] == 1
                ), waveform.shape
                with torch.no_grad():
                    encos = self.audio_tokenizer.encode(
                        waveform.to(self.audio_tokenizer._device)
                    )
                assert (
                    encos.shape[1] == self.args.n_codebooks
                ), f"encos.shape: {encos.shape}"
                encos = encos.cpu().squeeze(0).numpy().tolist()  # [K, T]
                if self.args.special_first:
                    raise NotImplementedError
                    # y = [[int(n)+self.args.n_special for n in l] for l in encos]
                else:
                    y = [[int(n) for n in l] for l in encos]
                return x, y, dataset_dir, audio_ext
            except Exception as e:
                logging.info(
                    f"failed with time stretch and codec encode for {audio_fn}"
                )
                logging.info(f"error: {e}")
                pass

        try:
            with open(pf, "r") as p, open(ef, "r") as e:
                phns = [l.strip() for l in p.readlines()]
                assert len(phns) == 1, phns
                all_phns = phns[0].split(" ")
                x = [
                    self.phn2num[item] for item in all_phns if item in self.phn2num
                ]  # we assume that OOD will not happen, because phn vocab is small
                encos = [
                    l.strip().split()
                    for k, l in enumerate(e.readlines())
                    if k < self.args.n_codebooks
                ]

                assert len(encos) == self.args.n_codebooks, ef

                if self.args.special_first:
                    raise NotImplementedError
                    # y = [[int(n)+self.args.n_special for n in l] for l in encos]
                else:
                    y = [[int(n) for n in l] for l in encos]
        except:
            logging.info(
                f"loading failed for {pf} and {ef}, maybe files don't exist or are corrupted"
            )
            return [], [[]], dataset_dir, audio_ext

        return x, y, dataset_dir, audio_ext

    # this uses the output of step7_ipa_alignment.py
    def find_neighbor(self, neighbors, y_len, dataset_dir, audio_ext):
        neighbor = random.choice(neighbors)
        neighbor_enc_fn = os.path.join(
            dataset_dir, self.args.encodec_folder_name, neighbor[0]
        )
        if not os.path.isfile(neighbor_enc_fn):
            return None, None
        neighbor_audio_path = os.path.join(
            dataset_dir,
            self.args.audio_folder_name,
            neighbor[0].replace(".txt", audio_ext),
        )
        if getattr(self.args, "time_stretch_prob", 0) > 0 and not os.path.isfile(
            neighbor_audio_path
        ):
            logging.info(f"audio file not found: {neighbor_audio_path}")
            return None, None
        if random.random() < getattr(self.args, "time_stretch_prob", 0):
            time_stretch_flag = True
            speed_factor = (
                random.uniform(
                    -self.args.time_stretch_bound, self.args.time_stretch_bound
                )
                + 1
            )
            duration_factor = 1 / speed_factor
        else:
            time_stretch_flag = False
            duration_factor = 1

        ####################### TODO for now always use the entire neighbor for emilia
        ####################### TODO for now always use the entire neighbor for emilia
        # if it's gigaspeech or emilia, we did not run MFA forced alignment, and therefore no ipa alignment, and will just use the entire neighbor as the prompt

        if "/emilia" in dataset_dir:
            # get neighbor duration
            neighbor_dur = float(neighbor[2])
            if (
                neighbor_dur * duration_factor + y_len / self.args.encodec_sr
                > self.args.audio_max_length
                or neighbor_dur * duration_factor < self.args.min_prompt_len
            ):
                return None, None
            try:
                neighbor_pf = os.path.join(
                    dataset_dir, self.args.phn_folder_name, neighbor[0]
                )
                with open(neighbor_pf, "r") as p:
                    phns = [l.strip() for l in p.readlines()]
                    assert len(phns) == 1, phns
                    all_phns = phns[0].split(" ")
                    phn_token = [
                        self.phn2num[item] for item in all_phns if item in self.phn2num
                    ]
            except:
                logging.info(
                    f"loading failed for {neighbor_pf}, maybe files don't exist"
                )
                return None, None
            # if do not stretch the audio
            if not time_stretch_flag:
                with open(neighbor_enc_fn, "r") as f:
                    neighbor_enc = [l.strip().split() for l in f.readlines()]
                if len(neighbor_enc) != self.args.n_codebooks:
                    return None, None
                # if too long
                else:
                    if self.args.special_first:
                        raise NotImplementedError
                        # neighbor_enc = [[int(n)+self.args.n_special for n in l] for l in neighbor_enc]
                    else:
                        neighbor_enc = [[int(n) for n in l] for l in neighbor_enc]

                    return phn_token, neighbor_enc
            else:  # stretch the audio with ffmpeg-python
                process = (
                    ffmpeg.input(neighbor_audio_path, ss=0, t=neighbor_dur)
                    .output(
                        "pipe:1",
                        format="f32le",
                        ac=1,
                        ar=self.audio_tokenizer.sample_rate,
                        filter="atempo={}".format(speed_factor),
                    )
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
                # Read the processed audio from ffmpeg stdout
                output, _ = process.communicate()

                # Convert the output to a numpy array
                output_np = np.frombuffer(output, dtype=np.float32).copy()

                # Reshape the numpy array back to the expected shape (1, samples for mono)
                waveform = torch.from_numpy(output_np)
                waveform = waveform.unsqueeze(0).unsqueeze(0)
                assert (
                    waveform.ndim == 3
                    and waveform.shape[0] == 1
                    and waveform.shape[1] == 1
                ), waveform.shape
                with torch.no_grad():
                    encos = self.audio_tokenizer.encode(
                        waveform.to(self.audio_tokenizer._device)
                    )
                assert (
                    encos.shape[1] == self.args.n_codebooks
                ), f"encos.shape: {encos.shape}"
                neighbor_enc = encos.cpu().squeeze(0).numpy().tolist()  # [K, T]
                return phn_token, neighbor_enc
        ####################### TODO for now always use the entire neighbor for emilia
        ####################### TODO for now always use the entire neighbor for emilia
        ipa_alignment_fn = os.path.join(
            dataset_dir, self.args.ipa_alignment_folder_name, neighbor[0]
        )
        if not os.path.isfile(ipa_alignment_fn):
            # print(f"file not found: {ipa_alignment_fn}", flush=True)
            return None, None
        with open(ipa_alignment_fn, "r") as f:
            alignments = [l.strip().split("\t") for l in f.readlines()]
        alignments = [
            [float(l[0]), float(l[1]), l[2]] for l in alignments if len(l) == 3
        ]
        alignments = [
            l
            for l in alignments
            if self.args.min_prompt_len
            < (l[1] - l[0]) * duration_factor
            < self.args.max_prompt_len
        ]
        if len(alignments) == 0:
            # print(f"no valid alignment found for {ipa_alignment_fn}")
            return None, None
        idx = random.choice(range(len(alignments)))
        while (
            (alignments[idx][1] - alignments[idx][0]) * duration_factor
            + y_len / self.args.encodec_sr
            > self.args.audio_max_length
        ):
            idx -= 1
            if idx < 0:
                # print(f"too long combined with y_len {ipa_alignment_fn=}, and {y_len=}")
                return None, None
        if (
            alignments[idx][1] - alignments[idx][0]
        ) * duration_factor < self.args.min_prompt_len:
            return None, None

        start_time, end_time = alignments[idx][:2]
        phn = alignments[idx][2].split(" ")
        phn_token = [self.phn2num[item] for item in phn if item in self.phn2num]
        if len(phn_token) == 0:
            return None, None

        if time_stretch_flag:
            duration = end_time - start_time
            process = (
                ffmpeg.input(neighbor_audio_path, ss=start_time, t=duration)
                .output(
                    "pipe:1",
                    format="f32le",
                    ac=1,
                    ar=self.audio_tokenizer.sample_rate,
                    filter="atempo={}".format(speed_factor),
                )
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            # Read the processed audio from ffmpeg stdout
            output, _ = process.communicate()

            # Convert the output to a numpy array
            output_np = np.frombuffer(output, dtype=np.float32).copy()

            # Reshape the numpy array back to the expected shape (1, samples for mono)
            waveform = torch.from_numpy(output_np)
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            assert (
                waveform.ndim == 3 and waveform.shape[0] == 1 and waveform.shape[1] == 1
            ), waveform.shape
            try:
                with torch.no_grad():
                    encos = self.audio_tokenizer.encode(
                        waveform.to(self.audio_tokenizer._device)
                    )
            except:
                logging.info(
                    f"failed with time stretch for {neighbor_audio_path}, from {start_time} to {end_time} with duration factor {duration_factor}, which leads to {duration*duration_factor} seconds"
                )
                return None, None
            assert (
                encos.shape[1] == self.args.n_codebooks
            ), f"encos.shape: {encos.shape}"
            neighbor_enc = encos.cpu().squeeze(0).numpy().tolist()  # [K, T]
            return phn_token, neighbor_enc
        else:
            # get encodec codes from storage
            with open(neighbor_enc_fn, "r") as f:
                neighbor_enc = [l.strip().split() for l in f.readlines()]
                if len(neighbor_enc) != self.args.n_codebooks:
                    # print(f"wrong number of codebooks for {neighbor_enc_fn}")
                    return None, None
                else:
                    # trim the encodec codes to the segment
                    start_enc_frame = int(start_time * self.args.encodec_sr)
                    end_enc_frame = int(end_time * self.args.encodec_sr)
                    neighbor_enc = [
                        l[start_enc_frame:end_enc_frame] for l in neighbor_enc
                    ]
                    if len(neighbor_enc[0]) == 0:
                        # print(f"no valid encodec codes found for {neighbor_enc_fn}")
                        return None, None
                    if self.args.special_first:
                        raise NotImplementedError
                    else:
                        neighbor_enc = [[int(n) for n in l] for l in neighbor_enc]
                    return phn_token, neighbor_enc

    def __getitem__(self, index):
        x, y, dataset_dir, audio_ext = self._load_phn_enc(index)
        x_len, y_len = len(x), len(y[0])
        extra_ret = {"x_sep_token_position": 0, "y_sep_token_position": 0}
        if x_len == 0 or y_len == 0:
            ret = {
                "x": None,
                "x_len": None,
                "y": None,
                "y_len": None,
            }
            ret.update(extra_ret)
            return ret
        while y_len < self.args.encodec_sr * self.args.audio_min_length:
            assert not self.args.dynamic_batching
            index = random.choice(range(len(self)))  # regenerate an index
            x, y, dataset_dir, audio_ext = self._load_phn_enc(index)
            x_len, y_len = len(x), len(y[0])

        # if use neighbor prompt
        x_neighbor, y_neighbor = None, None
        use_neighbor_prob = random.random()
        neighbor_fn = os.path.join(
            dataset_dir, self.args.neighbor_folder_name, self.data[index][0] + ".txt"
        )
        if (
            self.args.neighbor_prompt_prob > 0
            and use_neighbor_prob < self.args.neighbor_prompt_prob
            and os.path.isfile(neighbor_fn)
        ):  # it might not exist, just because we didn't find neighbor for this file (other than itself, which is common for emilia)
            with open(neighbor_fn, "r") as f:
                neighbors = [l.strip().split("\t") for l in f.readlines()]
            # select neighbors
            if "maxdist" in self.args.neighbor_selection_method:
                maxdist = int(self.args.neighbor_selection_method.split("_")[-1])
                # only keep neighbors with distance within maxdist
                neighbors = [n for n in neighbors if float(n[1]) <= maxdist]
            else:
                raise NotImplementedError
            x_neighbor, y_neighbor = None, None
            if len(neighbors) > 0:
                x_neighbor, y_neighbor = self.find_neighbor(
                    neighbors, y_len, dataset_dir, audio_ext
                )
                i_trial = 0
                while (
                    x_neighbor is None
                    and i_trial < self.args.num_trial
                    and i_trial < len(neighbors)
                ):
                    x_neighbor, y_neighbor = self.find_neighbor(
                        neighbors, y_len, dataset_dir, audio_ext
                    )
                    i_trial += 1

        if x_neighbor != None:
            if self.args.x_sep_token != None:
                x = x_neighbor + [self.args.x_sep_token] + x
            else:
                x = x_neighbor + x
            if self.args.y_sep_token != None:
                y = [
                    y_neighbor[i] + [self.args.y_sep_token] + y[i]
                    for i in range(len(y))
                ]
            else:
                y = [y_neighbor[i] + y[i] for i in range(len(y))]
            extra_ret["y_sep_token_position"] = (
                len(y_neighbor[0]) + 1
            )  # if using y_sep_token, this is actually the position of the token right before the y_sep_token, but since y_sep_token is ignored in loss computation, it's fine that we use the position of the token right before it
            extra_ret["x_sep_token_position"] = len(x_neighbor) + 1
            x_len, y_len = len(x), len(y[0])

        # consider adding eos to the end of the text
        if self.args.add_eos_to_text != 0:
            x.append(self.args.add_eos_to_text)
            x_len += 1
        if getattr(self.args, "add_bos_to_text", 0) != 0:
            x = [self.args.add_bos_to_text] + x
            x_len += 1
        ### padding and cropping ###
        ### padding and cropping ###
        # adjust the length of encodec codes, pad to max_len or randomly crop
        orig_y_len = copy.copy(y_len)
        max_len = int(self.args.audio_max_length * self.args.encodec_sr)
        if y_len > max_len + 10:  # give it some margin for rounding error
            raise RuntimeError(f"audio is too long, {y_len=}, {max_len=}")
        else:
            audio_start = 0
            if not self.args.dynamic_batching:
                pad = (
                    [0] * (max_len - y_len)
                    if self.args.sep_special_token
                    else [self.args.audio_pad_token] * (max_len - y_len)
                )
                for i in range(len(y)):
                    y[i] = y[i] + pad

        if self.args.pad_x and x_len <= self.args.text_max_length:
            pad = (
                [0] * (self.args.text_max_length - x_len)
                if self.args.sep_special_token
                else [self.args.text_pad_token] * (self.args.text_max_length - x_len)
            )
            x = x + pad

        ret = {
            "x": torch.LongTensor(x),
            "x_len": x_len,
            "y": torch.LongTensor(y),
            "y_len": y_len,
        }
        ret.update(extra_ret)

        return ret

    def collate(self, batch):
        # make sure keys in every batch is the same
        for batch1, batch2 in zip(batch[:-1], batch[1:]):
            assert set(batch1.keys()) == set(
                batch2.keys()
            ), f"keys in batch1: {batch1.keys()} and keys in batch2: {batch2.keys()} are different"
        out = {key: [] for key in batch[0]}
        for item in batch:
            if item["x"] == None:  # deal with load failure
                continue
            for key, val in item.items():
                out[key].append(val)
        res = {}
        if self.args.pad_x:
            res["x"] = torch.stack(out["x"], dim=0)
        else:
            res["x"] = torch.nn.utils.rnn.pad_sequence(
                out["x"], batch_first=True, padding_value=self.args.text_pad_token
            )
        res["x_lens"] = torch.LongTensor(out["x_len"])
        if self.args.dynamic_batching:
            res["y"] = torch.nn.utils.rnn.pad_sequence(
                [item.transpose(1, 0) for item in out["y"]],
                padding_value=self.args.audio_pad_token,
            )
            res["y"] = res["y"].permute(1, 2, 0)  # T B K -> B K T
        else:
            res["y"] = torch.stack(out["y"], dim=0)
        res["y_lens"] = torch.LongTensor(out["y_len"])
        res["text_padding_mask"] = torch.arange(res["x"][0].shape[-1]).unsqueeze(
            0
        ) >= res["x_lens"].unsqueeze(1)
        res["audio_padding_mask"] = torch.arange(res["y"][0].shape[-1]).unsqueeze(
            0
        ) >= res["y_lens"].unsqueeze(1)
        if "y_sep_token_position" in out:
            res["y_sep_token_position"] = torch.LongTensor(out["y_sep_token_position"])
        if "x_sep_token_position" in out:
            res["x_sep_token_position"] = torch.LongTensor(out["x_sep_token_position"])
        return res
