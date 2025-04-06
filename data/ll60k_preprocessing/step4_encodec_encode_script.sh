#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate voicestar

dir=${dir:-/data/scratch/pyp/datasets/librilight}
sub_root=${sub_root:-preprocessed}
encodec_name=${encodec_name:-encodec_6f79c6a8.th} # or encodec_8cb1024_giga.th
n_workers=${n_workers:-12}
batch_size=${batch_size:-64}
audio_sr=16000
model_sr=16000
downsample_rate=320
model_code_sr=50
len_cap=1000
min_len=0.5
partition=${partition:-"1/1"}
split=${split:-"valid"}

python step4_encodec_encode.py --dir $dir --sub_root ${sub_root} --encodec_name ${encodec_name} --n_workers $n_workers --batch_size $batch_size --audio_sr $audio_sr --model_sr $model_sr --downsample_rate $downsample_rate --model_code_sr $model_code_sr --len_cap $len_cap --min_len $min_len --partition $partition --split $split