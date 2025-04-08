#!/bin/bash

# TODO SLURM header goes here

source ~/miniconda3/etc/profile.d/conda.sh
conda activate voicestar

### ====================
### ====================
### ====================
dataset=librilight
mkdir -p ./logs

exp_root="path/to/save/log_and_ckpt/VoiceStar/runs"
exp_name="VoiceStar_840M_30s_new" # 
dataset_dir="['/path/to/librilight/preprocessed','/path/to/emilia/preprocessed']"
manifest_folder_name='manifest_final_encodec'
encodec_codes_folder_name="encodec_4cb"

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=11751
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

GRAD_ACC_STEPS=3
srun torchrun --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=${SLURM_GPUS_PER_NODE} --rdzv_backend=c10d --rdzv_id=${SLURM_NODEID} --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
../main.py \
--uniform_weight_start_step 50000 \
--local_wandb 1 \
--resume \
--multinodes 1 \
--no_loss_on_prefix 1 \
--neighbor_prompt_prob 0.5 \
--x_sep_token 498 \
--y_sep_token 2052 \
--n_special 5 \
--rope_base 10000 \
--progress_no_multiple 1 \
--progress_scale 2000 \
--early_stop_step 7500 \
--early_stop_threshold -1 \
--precision "float16" \
--use_sinusoidal 0 \
--drop_long 1 \
--pad_x 0 \
--codebook_weight "[5,1,0.5,0.1]" \
--encodec_sr 50 \
--num_steps 100000 \
--lr 0.035 \
--warmup_fraction 0.02 \
--optimizer_name "ScaledAdam" \
--pseudo_epoch_size 3000 \
--reduce_lr_start_step 3000 \
--reduce_lr_start_epoch 4 \
--clipping_update_period 1000 \
--d_model 1024 \
--audio_embedding_dim 1024 \
--nhead 16 \
--num_encoder_layer 12 \
--num_decoder_layer 40 \
--max_num_tokens 20000 \
--gradient_accumulation_steps ${GRAD_ACC_STEPS} \
--val_max_num_tokens 8000 \
--num_buckets 20 \
--max_prompt_len 10 \
--audio_max_length 30 \
--audio_min_length 1 \
--text_max_length 2000 \
--text_min_length 10 \
--tb_write_every_n_steps 50 \
--print_every_n_steps 500 \
--val_every_n_steps 1000 \
--save_every_n_steps 1000000 \
--text_vocab_size 500 \
--text_pad_token 500 \
--phn_folder_name "phoneme" \
--manifest_name ${manifest_folder_name} \
--encodec_folder_name ${encodec_codes_folder_name} \
--enc_dec 1 \
--audio_vocab_size 2048 \
--reduced_eog 1 \
--empty_token 2048 \
--eog 2049 \
--audio_pad_token 2050 \
--eos 2051 \
--n_codebooks 4 \
--exp_dir "${exp_root}/${exp_name}" \
--dataset_dir ${dataset_dir} \
>> ./logs/${exp_name}_${SLURM_JOB_ID}_gradAccSteps${GRAD_ACC_STEPS}.log 2>&1