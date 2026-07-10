#!/bin/bash

# ====================
# MSRVTT-10 train
# ====================
OUTPUT_DIR="/mnt/data/wang_shaokun/StructAlign/outputs_v1"
mkdir -p ${OUTPUT_DIR}
# Resume option: omit --start_task or keep it at 1 for normal training; set it to >1 to resume from a later task.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} nohup ~/anaconda3/envs/kk/bin/python -u main.py \
    --exp_name="msrvtt_10" \
    --config="config/sa_config.yaml" \
    --dataset_name="MSRVTT" \
    --path_data="data/MSRVTT_10_dataset.pkl" \
    --videos_dir="datasets/MSRVTT/MSRVTT_Frames" \
    --arch="StructAlign" \
    --seed=42 \
    --start_task=1 \
    --task_num=10 \
    --output_dir "${OUTPUT_DIR}" \
    > ${OUTPUT_DIR}/msrvtt_10.log 2>&1 &

# ====================
# MSRVTT-10 eval
# ====================
# OUTPUT_DIR="/mnt/data/wang_shaokun/StructAlign/outputs_v1"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ~/anaconda3/envs/kk/bin/python -u main.py \
#     --exp_name="msrvtt_10" \
#     --config="config/sa_config.yaml" \
#     --dataset_name="MSRVTT" \
#     --path_data="data/MSRVTT_10_dataset.pkl" \
#     --videos_dir="datasets/MSRVTT/MSRVTT_Frames" \
#     --arch="StructAlign" \
#     --seed=42 \
#     --task_num=10 \
#     --eval \
#     --eval_mode="all" \
#     --eval_path="outputs_v1/msrvtt_10" \
#     --output_dir "${OUTPUT_DIR}"

# ====================
# MSRVTT-20 train
# ====================
OUTPUT_DIR="/mnt/data/wang_shaokun/StructAlign/outputs_v2"
mkdir -p ${OUTPUT_DIR}
# Resume option: omit --start_task or keep it at 1 for normal training; set it to >1 to resume from a later task.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} nohup ~/anaconda3/envs/kk/bin/python -u main.py \
    --exp_name="msrvtt_20" \
    --config="config/sa_config.yaml" \
    --dataset_name="MSRVTT" \
    --path_data="data/MSRVTT_20_dataset.pkl" \
    --videos_dir="datasets/MSRVTT/MSRVTT_Frames" \
    --arch="StructAlign" \
    --seed=42 \
    --start_task=1 \
    --task_num=20 \
    --output_dir "${OUTPUT_DIR}" \
    > ${OUTPUT_DIR}/msrvtt_20.log 2>&1 &

# ====================
# MSRVTT-20 eval
# ====================
# OUTPUT_DIR="/mnt/data/wang_shaokun/StructAlign/outputs_v2"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ~/anaconda3/envs/kk/bin/python -u main.py \
#     --exp_name="msrvtt_20" \
#     --config="config/sa_config.yaml" \
#     --dataset_name="MSRVTT" \
#     --path_data="data/MSRVTT_20_dataset.pkl" \
#     --videos_dir="datasets/MSRVTT/MSRVTT_Frames" \
#     --arch="StructAlign" \
#     --seed=42 \
#     --task_num=20 \
#     --eval \
#     --eval_mode="all" \
#     --eval_path="outputs_v2/msrvtt_20" \
#     --output_dir "${OUTPUT_DIR}"
