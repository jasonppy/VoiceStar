# find split, spk, books in libriheavy_cuts_dev.jsonl, libriheavy_cuts_test_clean.jsonl, libriheavy_cuts_test_other.jsonl
# those would be in "id" field

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
    with open(file_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            cur_data.append(json.loads(line.strip()))
    return cur_data


import os

dataroot = os.environ["DATAROOT"]
manifestroot = os.path.join(dataroot, "libriheavy")
tgt_names = [
    "libriheavy_cuts_dev.jsonl",
    "libriheavy_cuts_test_clean.jsonl",
    "libriheavy_cuts_test_other.jsonl",
]
orig_names = [
    "libriheavy_long_original_cuts_small.jsonl",
    "libriheavy_long_original_cuts_medium.jsonl",
    "libriheavy_long_original_cuts_large.jsonl",
]

id2split = {}
data = read_jsonl(os.path.join(manifestroot, "libriheavy_cuts_dev.jsonl"))
dev_ids = set(["/".join(item["id"].split("/")[:3]) for item in data])
data = read_jsonl(os.path.join(manifestroot, "libriheavy_cuts_test_clean.jsonl"))
test_clean_ids = set(["/".join(item["id"].split("/")[:3]) for item in data])
data = read_jsonl(os.path.join(manifestroot, "libriheavy_cuts_test_other.jsonl"))
test_other_ids = set(["/".join(item["id"].split("/")[:3]) for item in data])

long_dev = []
long_test_clean = []
long_test_other = []
for orig_name in orig_names:
    keep = []
    data = read_jsonl(os.path.join(manifestroot, orig_name))
    for item in tqdm.tqdm(data):
        if "/".join(item["id"].split("/")[:3]) in dev_ids:
            long_dev.append(item)
        elif "/".join(item["id"].split("/")[:3]) in test_clean_ids:
            long_test_clean.append(item)
        elif "/".join(item["id"].split("/")[:3]) in test_other_ids:
            long_test_other.append(item)
        else:
            keep.append(item)
    write_jsonl(keep, os.path.join(manifestroot, orig_name.replace("_original", "")))

write_jsonl(long_dev, os.path.join(manifestroot, "libriheavy_long_cuts_dev.jsonl"))
write_jsonl(
    long_test_clean, os.path.join(manifestroot, "libriheavy_long_cuts_test_clean.jsonl")
)
write_jsonl(
    long_test_other, os.path.join(manifestroot, "libriheavy_long_cuts_test_other.jsonl")
)
