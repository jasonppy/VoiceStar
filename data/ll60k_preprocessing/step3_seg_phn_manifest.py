from importlib.resources import path
import pathlib
import soundfile as sf
import numpy as np
import json
import multiprocessing
import argparse
import tqdm
import gzip
import time
import os
from tokenizer import TextTokenizer, tokenize_text
import glob
import sys
import os, random, numpy as np, socket
import json
import tqdm
def write_jsonl(data, fn):
    with open(fn, "w") as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
def read_jsonl(file_path):
    cur_data = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            cur_data.append(json.loads(line.strip()))
    return cur_data
def save_audio(seq, fn):
    output = seq
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    sf.write(fn, output, samplerate=16000)

def save_text(text, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as wwf:
        wwf.writelines(text)

def phonemize_and_save(text, fn):
    phn = tokenize_text(text_tokenizer, text)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as f:
        f.write(' '.join(phn))
    return set(phn)

def cut_sequence(task):
    in_audio_fn, output_dir, metadata = task
    if not os.path.isfile(in_audio_fn):
        # print("missing: ", in_audio_fn)
        return None
    data, samplerate = sf.read(in_audio_fn)
    assert len(data.shape) == 1
    assert samplerate == 16000
    all_phns = set()
    for item in metadata:
        out_fn = item['file_id']
        out_audio_fn = os.path.join(output_dir, "audio", out_fn)
        out_text_fn = os.path.join(output_dir, "audio", out_fn.replace(".flac", ".txt"))
        out_phn_fn = os.path.join(output_dir, "phoneme", out_fn.replace(".flac", ".txt"))
        save_audio(data[int(item['vad'][0]*samplerate):int(item['vad'][1]*samplerate)], out_audio_fn)
        save_text(item['text'], out_text_fn)
        phns = phonemize_and_save(item['text'], out_phn_fn)
        all_phns.update(phns)
    
    return all_phns


from collections import defaultdict
# Function to create a defaultdict recursively
def nested_defaultdict(levels, inner_type):
    if levels <= 1:
        return defaultdict(inner_type)
    return defaultdict(lambda: nested_defaultdict(levels-1, inner_type))


def open_mani(fn):
    print("load segmentation and transcription metadata...")
    stime = time.time()
    data = []
    with gzip.open(fn, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"loading done, took {time.time() - stime:.4f} seconds")
    return data

def cut(split,
        audio_dir,
        mani_dir,
        output_dir,
        n_process=32,
        percent=0.5):
    split2manifest = {
            "train": [
                "libriheavy_long_cuts_small.jsonl", 
                "libriheavy_long_cuts_medium.jsonl", 
                "libriheavy_long_cuts_large.jsonl",
                "libriheavy_cuts_small.jsonl", 
                "libriheavy_cuts_medium.jsonl", 
                "libriheavy_cuts_large.jsonl",
            ],
            "valid": [
                    "libriheavy_cuts_dev.jsonl",
                    "libriheavy_long_cuts_dev.jsonl"
            ],
            "test": [
                    "libriheavy_cuts_test_clean.jsonl",
                    "libriheavy_cuts_test_other.jsonl",
                    "libriheavy_long_cuts_test_clean.jsonl",
                    "libriheavy_long_cuts_test_other.jsonl"
            ]
        }

    print("organize data by recording_id (i.e. the original big .flac file name)...")
    stime = time.time()
    organized_data = nested_defaultdict(4, list)
    manifest_fn = os.path.join(output_dir, "manifest_mimi", split+".txt")
    os.makedirs(os.path.join(output_dir, "manifest_mimi"), exist_ok=True)
    with open(manifest_fn, "w") as wf:
        for mani_fn in split2manifest[split]:
            # data = open_mani(os.path.join(mani_dir, mani_fn))
            data = read_jsonl(os.path.join(mani_dir, mani_fn))
            for item in data:
                file_id = item['supervisions'][0]['id'] + '.flac'
                recording_id = item['recording']['id'] + '.flac'
                sizeSplit, spk, book, flac = recording_id.split("/") # e.g. 'medium/100/emerald_city_librivox_64kb_mp3/emeraldcity_01_baum_64kb'
                if os.path.isfile(os.path.join(audio_dir, recording_id)):
                    vad = (item['start'], item['start']+item['duration'])
                    text = item['supervisions'][0]['custom']['texts'][0]
                    file_id = file_id.replace(".flac", "") + f"_{vad[0]:.2f}_{vad[1]:.2f}.flac"
                    organized_data[sizeSplit][spk][book][recording_id].append({"file_id": file_id, "vad":vad, "text": text})
                    wf.writelines(f"{file_id}\t{item['duration']}\n")
    
    # #### take only a subet of tasks
    tasks = [(os.path.join(audio_dir, recording_id), output_dir, organized_data[sizeSplit][spk][book][recording_id], spk) for sizeSplit in organized_data for spk in organized_data[sizeSplit] for book in organized_data[sizeSplit][spk] for recording_id in organized_data[sizeSplit][spk][book]]
    ntasks = len(tasks)
    spk2tasks = defaultdict(list)
    for task in tasks:
        spk2tasks[task[3]].append(task)
    # randomly shuffle each task list for each speaker
    for spk in spk2tasks:
        random.shuffle(spk2tasks[spk])
    # take only 20% of the tasks, uniformly sampled from each speaker
    # randomly pick a speaker, and then randomly pick a task from that speaker
    tasks = []
    while len(tasks) < ntasks * percent:
        spk = random.choice(list(spk2tasks.keys()))
        if len(spk2tasks[spk]) == 0:
            continue
        tasks.append(spk2tasks[spk].pop()[:-1])
    print(f"take only {percent*100:.2f}% of the tasks, {len(tasks)} out of {ntasks} tasks")
    #### take only a subet of tasks

    print(f"organizing done, took {time.time() - stime:.4f} seconds")
    print(f"Launching {n_process} processes")
    phn_vocab = set()
    cnt = 0
    with multiprocessing.Pool(processes=n_process) as pool:
        for phns in tqdm.tqdm(pool.imap_unordered(cut_sequence, tasks), total=len(tasks)):
            cnt += 1
            if phns != None:
                phn_vocab.update(phns)

    # save phn vocabulary
    if split == "train":
        vocab_fn = os.path.join(output_dir, "vocab.txt")
        with open(vocab_fn, "w") as f:
            for i, phn in enumerate(list(phn_vocab)):
                if i < len(list(phn_vocab)) - 1:
                    f.write(f"{str(i)}\t{phn}\n")
                else:
                    f.write(f"{str(i)}\t{phn}")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Cut a dataset in small "
                                     "sequences using VAD files")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'], help="train = libriheavy_cuts_{small,medium,large}.jsonl.gz, valid = libriheavy_cuts_dev_{clean,other}.jsonl.gz, test = libriheavy_cuts_test_{clean,other}.jsonl.gz")
    parser.add_argument('--audio_dir', type=str, default="/data/scratch/pyp/datasets/librilight_example",
                        help="Path to the audio directory")
    parser.add_argument('--manifest_dir', type=str, default="/data/scratch/pyp/datasets/librilight/libriheavy", help="path to the transcription file's dir, can be downloaded https://huggingface.co/datasets/pkufool/libriheavy/tree/main/v0.1")
    parser.add_argument('--output_dir', type=str, default="/data/scratch/pyp/datasets/librilight/librilight_example_preprocessed",
                        help="Path to the output directory")
    parser.add_argument('--n_workers', type=int, default=16,
                        help="Number of parallel worker processes")
    parser.add_argument('--percent', type=float, default=0.5, help="take only this percent of the tasks, randomly sampled from each speaker")


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    text_tokenizer = TextTokenizer()
    cut(args.split, args.audio_dir, args.manifest_dir, args.output_dir, args.n_workers, args.percent)