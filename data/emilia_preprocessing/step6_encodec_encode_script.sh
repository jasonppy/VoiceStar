#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate voicestar

dir=${processed_dir:-/data/scratch/pyp/datasets/emilia}
sub_root=${sub_root:-preprocessed}
encodec_name=${encodec_name:-"encodec_6f79c6a8.th"}
n_workers=${n_workers:-64}
batch_size=${batch_size:-512}
audio_sr=16000
model_sr=16000
downsample_rate=320
model_code_sr=50
len_cap=1000
min_len=0.5
partition=${partition:-"1/1"}
split=${split:-"train"}

python step6_encodec_encode.py --root $dir --sub_root ${sub_root} --encodec_name ${encodec_name} --n_workers $n_workers --batch_size $batch_size --audio_sr $audio_sr --model_sr $model_sr --downsample_rate $downsample_rate --model_code_sr $model_code_sr --len_cap $len_cap --min_len $min_len --partition $partition --split $split