#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o posttrain_procy_qa-%j.out
#SBATCH --gres gpu:2

export HF_DATASETS_CACHE='/sdb/zke4/dataset_cache'
export TRANSFORMERS_CACHE='/sdb/zke4/model_cache'
#export TRANSFORMERS_OFFLINE=1
max_samples=640000



for idrandom in  0
do
  for task in 0
  do
    CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --use_env posttrain.py \
    --per_device_train_batch_size 62 \
    --fp16\
    --max_seq_length 164 \
    --max_samples ${max_samples} \
    --idrandom ${idrandom} \
    --ntasks 6 \
    --task ${task} \
    --baseline 'dga'
  done
done
