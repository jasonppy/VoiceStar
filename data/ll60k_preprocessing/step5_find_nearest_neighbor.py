# for each each audio segment, find the non-overlapping neighboring segments
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
import json
import tqdm


def write_jsonl(data, fn):
    with open(fn, "w") as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_jsonl(file_path):
    cur_data = []
    with open(file_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            cur_data.append(json.loads(line.strip()))
    return cur_data


from collections import defaultdict


# Function to create a defaultdict recursively
def nested_defaultdict(levels, inner_type):
    if levels <= 1:
        return defaultdict(inner_type)
    return defaultdict(lambda: nested_defaultdict(levels - 1, inner_type))


def find_neighbor(args):
    split2manifest = {
        "train": [
            "libriheavy_cuts_small.jsonl",
            "libriheavy_cuts_medium.jsonl",
            "libriheavy_cuts_large.jsonl",
            "libriheavy_long_cuts_small.jsonl",
            "libriheavy_long_cuts_medium.jsonl",
            "libriheavy_long_cuts_large.jsonl",
        ],
        "valid": ["libriheavy_cuts_dev.jsonl", "libriheavy_long_cuts_dev.jsonl"],
        "test": [
            "libriheavy_cuts_test_clean.jsonl",
            "libriheavy_cuts_test_other.jsonl",
            "libriheavy_long_cuts_test_clean.jsonl",
            "libriheavy_long_cuts_test_other.jsonl",
        ],
    }

    stime = time.time()
    organized_data = nested_defaultdict(4, list)
    for mani_fn in split2manifest[args.split]:
        # data = open_mani(os.path.join(mani_dir, mani_fn))
        mani_full_fn = os.path.join(args.manifest_dir, mani_fn)
        data = read_jsonl(mani_full_fn)
        for item in data:
            file_id = item["supervisions"][0]["id"] + ".flac"
            recording_id = item["recording"]["id"] + ".flac"
            sizeSplit, spk, book, flac = recording_id.split(
                "/"
            )  # e.g. 'medium/100/emerald_city_librivox_64kb_mp3/emeraldcity_01_baum_64kb'
            if os.path.isfile(os.path.join(args.audio_dir, recording_id)):
                vad = (item["start"], item["start"] + item["duration"])
                text = item["supervisions"][0]["custom"]["texts"][0]
                file_id = (
                    file_id.replace(".flac", "") + f"_{vad[0]:.2f}_{vad[1]:.2f}.flac"
                )
                organized_data[sizeSplit][spk][book][recording_id].append(
                    {"file_id": file_id, "vad": vad, "text": text}
                )

    # # for each recording_id, find the non-overlapping neighboring segments based on vad
    # for sizeSplit in organized_data:
    #     for spk in organized_data[sizeSplit]:
    #         for book in organized_data[sizeSplit][spk]:
    #             for recording_id in organized_data[sizeSplit][spk][book]:
    #                 segments = organized_data[sizeSplit][spk][book][recording_id]
    #                 segments.sort(key=lambda x: x['vad'][0])
    #                 for i in range(len(segments)):
    #                     # for segment i, find the non-overlapping neighboring segments
    #                     write_fn = os.path.join(args.output_dir, f"{segments[i]['file_id'].replace('.flac', '.txt')}")
    #                     neighbors = []
    #                     distance = []
    #                     for j in range(len(segments)):
    #                         if segments[i]['vad'][1] < segments[j]['vad'][0] or segments[i]['vad'][0] > segments[j]['vad'][0]:
    #                             neighbors.append(segments[j]['file_id'].replace('.flac', '.txt'))
    #                             distance.append(min(abs(segments[i]['vad'][1] - segments[j]['vad'][0]), abs(segments[i]['vad'][0] - segments[j]['vad'][1])))
    #                     # order neighbors by distance
    #                     neighbors_distance = [[x, dist] for dist, x in sorted(zip(distance, neighbors))]
    #                     os.makedirs(os.path.dirname(write_fn), exist_ok=True)
    #                     with open(write_fn, "w") as f:
    #                         # note that there might be no neighbors, in which case the file is empty
    #                         for neighbor, dist in neighbors_distance:
    #                             f.write(f"{neighbor}\t{dist}\n")

    # use multiprocessing.Pool for the above
    segments = [
        organized_data[sizeSplit][spk][book][recording_id]
        for sizeSplit in organized_data
        for spk in organized_data[sizeSplit]
        for book in organized_data[sizeSplit][spk]
        for recording_id in organized_data[sizeSplit][spk][book]
    ]
    # only keep those that are exist
    print(f"originally total {len(segments)} segments")
    segments = [
        seg
        for seg in segments
        if os.path.isfile(
            os.path.join(
                "/".join(args.output_dir.split("/")[:-1]), "audio", seg[0]["file_id"]
            )
        )
    ]
    print(f"after check existance, total {len(segments)} segments")
    print(f"organizing took {(time.time()-stime)/60:.2f} minutes")
    with multiprocessing.Pool(processes=args.n_workers) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(find_neighbor_each, segments), total=len(segments)
        ):
            pass


# audio_root = "/data/scratch/pyp/datasets/librilight/preprocessed/audio"
def find_neighbor_each(segments):
    # for each recording_id, find the non-overlapping neighboring segments based on vad
    # only keep segments that have audio files
    # actually only keep segments that have ipa_alignment files
    segments = [
        seg
        for seg in segments
        if os.path.isfile(
            os.path.join(
                "/".join(args.output_dir.split("/")[:-1]),
                "ipa_alignment",
                seg["file_id"].replace(".flac", ".txt"),
            )
        )
    ]
    if len(segments) <= 1:
        return
    for i in range(len(segments)):
        # for segment i, find the non-overlapping neighboring segments
        write_fn = os.path.join(
            args.output_dir, f"{segments[i]['file_id'].replace('.flac', '.txt')}"
        )
        neighbors = []
        distance = []
        for j in range(len(segments)):
            if (
                segments[i]["vad"][1] < segments[j]["vad"][0]
                or segments[i]["vad"][0] > segments[j]["vad"][0]
            ):
                neighbors.append(segments[j])
                distance.append(
                    min(
                        abs(segments[i]["vad"][1] - segments[j]["vad"][0]),
                        abs(segments[i]["vad"][0] - segments[j]["vad"][1]),
                    )
                )
        if len(neighbors) == 0:
            continue
        # order neighbors by distance
        index = np.argsort(distance)
        neighbors_distance = [[neighbors[ind], distance[ind]] for ind in index]
        os.makedirs(os.path.dirname(write_fn), exist_ok=True)
        with open(write_fn, "w") as f:
            # note that there might be no neighbors, in which case the file is empty
            for neighbor, dist in neighbors_distance:
                f.write(
                    f"{neighbor['file_id'].replace('.flac', '.txt')}\t{dist}\t{neighbor['vad'][1] - neighbor['vad'][0]}\n"
                )  # file_id, distance, duration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut a dataset in small " "sequences using VAD files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="train = libriheavy_cuts_{small,medium,large}.jsonl.gz, valid = libriheavy_cuts_dev_{clean,other}.jsonl.gz, test = libriheavy_cuts_test_{clean,other}.jsonl.gz",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/data/scratch/pyp/datasets/librilight_example",
        help="Path to the audio directory",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default="/data/scratch/pyp/datasets/librilight/libriheavy",
        help="path to the transcription file's dir, can be downloaded https://huggingface.co/datasets/pkufool/libriheavy/tree/main/v0.1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/scratch/pyp/datasets/librilight/librilight_example_preprocessed/neighbors",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--n_workers", type=int, default=16, help="Number of parallel worker processes"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    find_neighbor(args)
