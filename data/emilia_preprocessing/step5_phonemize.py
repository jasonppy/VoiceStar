import sys, copy
import os, random, numpy as np, socket

import json
import tqdm
from multiprocessing import Pool
import glob, os, fire
from collections import defaultdict
sys.path.insert(0, "../../")
from data.tokenizer import TextTokenizer, tokenize_text

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


def phonemize_and_save(text, fn, text_tokenizer):
    """Phonemizes the text and saves the result to a file."""
    phn = tokenize_text(text_tokenizer, text)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as f:
        f.write(" ".join(phn))
    return set(phn)


def process_item(item, root, sub_root, audio_folder, phn_folder, audio_ext, text_ext, phn_ext, text_tokenizer):
    """Worker function to process a single item."""
    text_path = os.path.join(root, sub_root, audio_folder, item[0].replace(audio_ext, text_ext))
    if not os.path.exists(text_path):
        return {"missing_text": text_path, "success": False, "cur_phn_set": set()}

    with open(text_path, "r") as f:
        text = [line.strip() for line in f.readlines()]
        text = " ".join(text)

    phn_path = os.path.join(root, sub_root, phn_folder, item[0].replace(audio_ext, phn_ext))
    cur_phn_set = phonemize_and_save(text, phn_path, text_tokenizer)
    return {"missing_text": None, "success": True, "cur_phn_set": cur_phn_set}


def process_item_star(args):
    """Unpacks arguments for `process_item` to work with `imap`."""
    return process_item(*args)

def main(
    root="/data/scratch/pyp/datasets/emilia",
    sub_root="preprocessed",
    manifest_folder="manifest_for_codec",
    audio_folder="audio",
    phn_folder="phoneme",
    audio_ext=".mp3",
    text_ext=".txt",
    phn_ext=".txt",
    num_workers=8,
):
    """Main function to process phoneme generation in parallel."""
    # # Initialize the tokenizer
    text_tokenizer = TextTokenizer()
    all_fns = glob.glob(f"{root}/{sub_root}/{manifest_folder}/*.txt")
    print(f"found {len(all_fns)} manifest files")
    print(f"{all_fns[:3]=}")

    data = []
    for fn in all_fns:
        with open(fn, "r") as f:
            data += [line.strip().split("\t") for line in f]
    
    vocab = set()

    ################## parallel processing ##################
    ################## parallel processing ##################
    ################## parallel processing ##################
    # Prepare arguments for the worker function
    # tasks = [
    #     (
    #         item,
    #         root,
    #         sub_root,
    #         audio_folder,
    #         phn_folder,
    #         audio_ext,
    #         text_ext,
    #         phn_ext,
    #         text_tokenizer,
    #     )
    #     for item in data
    # ]

    # # Parallel processing with progress monitoring
    # results = []
    # with Pool(num_workers) as pool:
    #     for result in tqdm.tqdm(
    #         pool.imap_unordered(process_item_star, tasks),
    #         total=len(tasks),
    #         desc="Processing items",
    #     ):
    #         results.append(result)
    # # read all manifest endswith .txt
    # missing_text = [result["missing_text"] for result in results if not result["success"]]
    # for result in results:
    #     if result['success']:
    #         vocab.update(result['cur_phn_set'])
    ################## parallel processing ##################
    ################## parallel processing ##################
    ################## parallel processing ##################

    ################## sequential processing ##################
    ################## sequential processing ##################
    ################## sequential processing ##################
    missing_text = []
    for item in tqdm.tqdm(data):
        text_path = os.path.join(root, sub_root, audio_folder, item[0].replace(audio_ext, text_ext))
        if not os.path.exists(text_path):
            missing_text.append(text_path)
            continue
        try:
            with open(text_path, "r") as f:
                text = [line.strip() for line in f.readlines()]
                text = " ".join(text)
        except:
            print(f"Error reading {text_path}")
            continue
        cur_phn_set = phonemize_and_save(text, os.path.join(root, sub_root, phn_folder, item[0].replace(audio_ext, phn_ext)), text_tokenizer)
        vocab.update(cur_phn_set)
    ################## sequential processing ##################
    ################## sequential processing ##################
    ################## sequential processing ##################

    # save the vocab
    vocab = list(vocab)
    # sort the vocab
    vocab.sort()
    with open(os.path.join(root, sub_root, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab))

    # Collect missing text paths
    print(f"Missing text files: {len(missing_text)}")
    if missing_text:
        print("Some missing files:", missing_text[:10])  # Print the first 10 missing files as an example
    

if __name__ == "__main__":
    fire.Fire(main)


    



