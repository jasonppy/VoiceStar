# construct manifest file for training, note that we only have one train split
# also create neighbors folder for each sample, which is simply done through speaker label in the original manifest where each file has rows
# path\tdistance\tduration
# where distance is always 0 because we don't know the distance between the samples

# waiting on Yushen Chen to provide data filtering approach
import sys, copy
import os, random, numpy as np, socket

import json
import tqdm
from multiprocessing import Pool
import glob, os
from collections import defaultdict


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


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


out_en = {
    "EN_B00013_S00913",
    "EN_B00042_S00120",
    "EN_B00055_S04111",
    "EN_B00061_S00693",
    "EN_B00061_S01494",
    "EN_B00061_S03375",
    "EN_B00059_S00092",
    "EN_B00111_S04300",
    "EN_B00100_S03759",
    "EN_B00087_S03811",
    "EN_B00059_S00950",
    "EN_B00089_S00946",
    "EN_B00078_S05127",
    "EN_B00070_S04089",
    "EN_B00074_S09659",
    "EN_B00061_S06983",
    "EN_B00061_S07060",
    "EN_B00059_S08397",
    "EN_B00082_S06192",
    "EN_B00091_S01238",
    "EN_B00089_S07349",
    "EN_B00070_S04343",
    "EN_B00061_S02400",
    "EN_B00076_S01262",
    "EN_B00068_S06467",
    "EN_B00076_S02943",
    "EN_B00064_S05954",
    "EN_B00061_S05386",
    "EN_B00066_S06544",
    "EN_B00076_S06944",
    "EN_B00072_S08620",
    "EN_B00076_S07135",
    "EN_B00076_S09127",
    "EN_B00065_S00497",
    "EN_B00059_S06227",
    "EN_B00063_S02859",
    "EN_B00075_S01547",
    "EN_B00061_S08286",
    "EN_B00079_S02901",
    "EN_B00092_S03643",
    "EN_B00096_S08653",
    "EN_B00063_S04297",
    "EN_B00063_S04614",
    "EN_B00079_S04698",
    "EN_B00104_S01666",
    "EN_B00061_S09504",
    "EN_B00061_S09694",
    "EN_B00065_S05444",
    "EN_B00063_S06860",
    "EN_B00065_S05725",
    "EN_B00069_S07628",
    "EN_B00083_S03875",
    "EN_B00071_S07665",
    "EN_B00071_S07665",
    "EN_B00062_S04187",
    "EN_B00065_S09873",
    "EN_B00065_S09922",
    "EN_B00084_S02463",
    "EN_B00067_S05066",
    "EN_B00106_S08060",
    "EN_B00073_S06399",
    "EN_B00073_S09236",
    "EN_B00087_S00432",
    "EN_B00085_S05618",
    "EN_B00064_S01262",
    "EN_B00072_S01739",
    "EN_B00059_S03913",
    "EN_B00069_S04036",
    "EN_B00067_S05623",
    "EN_B00060_S05389",
    "EN_B00060_S07290",
    "EN_B00062_S08995",
}
en_filters = ["ا", "い", "て"]


from multiprocessing import Pool


def process_meta_item(item, root, sub_root, audio_folder, audio_ext, text_ext):
    global filtered_duration, filtered_count, total_duration, total_count
    # Data filtering following Yushen's approach
    if (
        item["wav"].split("/")[-1] in out_en
        or any(t in item["text"] for t in en_filters)
        or repetition_found(item["text"], length=4)
    ):
        return None, item["duration"], 1, 0, 0, (None, None)  # Return filtered results

    # Trim leading space from text if exists
    if item["text"].startswith(" "):
        item["text"] = item["text"][1:]

    # write text to text file
    text_fn = os.path.join(
        root, sub_root, audio_folder, item["wav"].replace(audio_ext, text_ext)
    )
    os.makedirs(os.path.dirname(text_fn), exist_ok=True)
    with open(text_fn, "w") as f:
        f.write(item["text"])

    # spk2info[item["speaker"]].append(item)
    return (
        f"{item['wav']}\t{item['duration']}\n",
        0,
        0,
        item["duration"],
        1,
        (item["speaker"], item),
    )  # Return processed results


def parallel_process_meta(
    meta, root, sub_root, audio_folder, num_workers, audio_ext, text_ext
):
    with Pool(num_workers) as pool:
        results = pool.starmap(
            process_meta_item,
            [
                (item, root, sub_root, audio_folder, audio_ext, text_ext)
                for item in meta
            ],
        )

    processed_items = []
    spkitem = []
    filtered_duration = 0
    filtered_count = 0
    total_duration = 0
    total_count = 0

    for result in results:
        if result[0]:  # If the item was processed
            processed_items.append(result[0])
        filtered_duration += result[1]
        filtered_count += result[2]
        total_duration += result[3]
        total_count += result[4]
        spkitem.append(result[5])

    return (
        processed_items,
        filtered_duration,
        filtered_count,
        total_duration,
        total_count,
        spkitem,
    )


def main(
    root: str = "/data/scratch/pyp/datasets/emilia",
    sub_root: str = "preprocessed",
    audio_folder: str = "audio",
    manifest_folder: str = "manifest_for_codec",
    neighbors_folder: str = "neighbors",
    audio_ext: str = ".mp3",
    text_ext: str = ".txt",
    num_workers: int = 8,  # Specify the number of workers
):
    # Find the segments that are untarred
    all_fns = [
        item
        for item in glob.glob(f"{root}/{sub_root}/{audio_folder}/*")
        if os.path.basename(item).startswith("EN_") and os.path.isdir(item)
    ]
    print(f"found {len(all_fns)} untarred segments")
    print(f"{all_fns[:3]}")

    res = []
    total_duration = 0
    total_count = 0
    filtered_duration = 0
    filtered_count = 0

    for fn in tqdm.tqdm(all_fns, desc="overall progress"):
        spk2info = defaultdict(list)
        metafn = os.path.join(root, "EN", os.path.basename(fn) + ".jsonl")
        meta = read_jsonl(metafn)

        # Parallel process metadata
        processed_items, fd, fc, td, tc, spkitem = parallel_process_meta(
            meta, root, sub_root, audio_folder, num_workers, audio_ext, text_ext
        )

        # Aggregate results
        res.extend(processed_items)
        filtered_duration += fd
        filtered_count += fc
        total_duration += td
        total_count += tc

        for spk, item in spkitem:
            if spk:
                spk2info[spk].append(item)

        # Save neighbor files
        for spk in spk2info:
            for item in spk2info[spk]:
                neighbor_fn = os.path.join(
                    root,
                    sub_root,
                    neighbors_folder,
                    item["wav"].replace(audio_ext, text_ext),
                )
                os.makedirs(os.path.dirname(neighbor_fn), exist_ok=True)
                tobe_write = [
                    f"{neighbor_item['wav'].replace(audio_ext, text_ext)}\t0\t{neighbor_item['duration']}\n"
                    for neighbor_item in spk2info[spk]
                    if neighbor_item["wav"] != item["wav"]
                ]
                if tobe_write:
                    with open(neighbor_fn, "w") as f:
                        f.writelines(tobe_write)

    print(
        f"total duration: {total_duration / 3600:.2f} hours, total count: {total_count}"
    )
    print(
        f"filtered duration: {filtered_duration / 3600:.2f} hours, filtered count: {filtered_count}"
    )
    save_fn = os.path.join(root, sub_root, manifest_folder, "train.txt")
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    with open(save_fn, "w") as f:
        for item in res:
            f.write(item)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
