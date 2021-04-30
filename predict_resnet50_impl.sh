#!/bin/bash
set -e

visable_device=$1
xpu_device_id=$2
all_xpu_device=$3
log_file=$4

export XPU_VISIBLE_DEVICES=${visable_device}
export XPU_CONV_AUTOTUNE=1

model_dir="/data/zhupengyang/xingchuang_4pd_v2/models/resnet50/"
data_dir="/data/houjue/20210305-resnet-train/ILSVRC2012_w/"

python tools/infer/predict.py \
    --image_file="${data_dir}" \
    --model_file="${model_dir}/inference.pdmodel" \
    --params_file="${model_dir}/inference.pdiparams" \
    --use_xpu=true \
    --top_k=5 \
    --batch_size=64 \
    --enable_calc_topk=true \
    --gt_label_path="${data_dir}/val_list.txt" \
    --xpu_device_id=${xpu_device_id} \
    --all_xpu_device=${all_xpu_device} \
    --log_file=${log_file}


