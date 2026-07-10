#!/bin/bash
# Unified ACTNET entrypoint aligned with the latest StructAlign v2 logic.
# Toggle by commenting/uncommenting the block you want to run.

# ====================
# ACTNET-10 train
# ====================
OUTPUT_DIR="/mnt/data/wang_shaokun/StructAlign/outputs_actnet10_v2"
mkdir -p ${OUTPUT_DIR}
# Resume option: omit --start_task or keep it at 1 for normal training; set it to >1 to resume from a later task.
CUDA_VISIBLE_DEVICES=7 nohup ~/anaconda3/envs/kk/bin/python -u main.py \
    --exp_name="actnet_10" \
    --config="config/actnet_sa_config.yaml" \
    --dataset_name="ACTNET" \
    --path_data="data/ACTNET_10_dataset.pkl" \
    --videos_dir="/mnt/data/wang_shaokun/CTVR/datasets/ACTNET/Activity_Clip_Frames/" \
    --arch="StructAlign" \
    --seed=42 \
    --start_task=1 \
    --task_num=10 \
    --output_dir "${OUTPUT_DIR}" \
    > ${OUTPUT_DIR}/ACTNET_10.log 2>&1 &

# ====================
# ACTNET-10 eval
# ====================
# OUTPUT_DIR="/mnt/data/wang_shaokun/StructAlign/outputs_actnet10_v2"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ~/anaconda3/envs/kk/bin/python -u main.py \
#     --exp_name="actnet_10" \
#     --config="config/actnet_sa_config.yaml" \
#     --dataset_name="ACTNET" \
#     --path_data="data/ACTNET_10_dataset.pkl" \
#     --videos_dir="/mnt/data/wang_shaokun/CTVR/datasets/ACTNET/Activity_Clip_Frames/" \
#     --arch="StructAlign" \
#     --seed=42 \
#     --task_num=10 \
#     --eval \
#     --eval_mode="all" \
#     --eval_path="outputs_actnet10_v2/actnet_10" \
#     --output_dir "${OUTPUT_DIR}"

# ====================
# ACTNET-20 train
# ====================
OUTPUT_DIR="/mnt/data/wang_shaokun/StructAlign/outputs_actnet20_v2"
mkdir -p ${OUTPUT_DIR}
# Resume option: omit --start_task or keep it at 1 for normal training; set it to >1 to resume from a later task.
CUDA_VISIBLE_DEVICES=7 nohup ~/anaconda3/envs/kk/bin/python -u main.py \
    --exp_name="actnet_20" \
    --config="config/actnet_sa_config.yaml" \
    --dataset_name="ACTNET" \
    --path_data="data/ACTNET_20_dataset.pkl" \
    --videos_dir="/mnt/data/wang_shaokun/CTVR/datasets/ACTNET/Activity_Clip_Frames/" \
    --arch="StructAlign" \
    --seed=42 \
    --start_task=1 \
    --task_num=20 \
    --output_dir "${OUTPUT_DIR}" \
    > ${OUTPUT_DIR}/ACTNET_20.log 2>&1 &

# ====================
# ACTNET-20 eval
# ====================
# OUTPUT_DIR="/mnt/data/wang_shaokun/StructAlign/outputs_actnet20_v2"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ~/anaconda3/envs/kk/bin/python -u main.py \
#     --exp_name="actnet_20" \
#     --config="config/actnet_sa_config.yaml" \
#     --dataset_name="ACTNET" \
#     --path_data="data/ACTNET_20_dataset.pkl" \
#     --videos_dir="/mnt/data/wang_shaokun/CTVR/datasets/ACTNET/Activity_Clip_Frames/" \
#     --arch="StructAlign" \
#     --seed=42 \
#     --task_num=20 \
#     --eval \
#     --eval_mode="all" \
#     --eval_path="outputs_actnet20_v2/actnet_20" \
#     --output_dir "${OUTPUT_DIR}"
