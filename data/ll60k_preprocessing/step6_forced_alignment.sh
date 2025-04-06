#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate voicecraft

partition=$1
file_root=$2
max_spk=${max_spk:-100}
n_workers=${n_workers:-64}
python step6_forced_alignment.py \
--partition $partition \
--file_root $file_root \
--max_spk $max_spk \
--n_workers $n_workers