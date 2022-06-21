#!/bin/bash

export FLAGS_cudnn_deterministic=True

unset CUDA_VISIBLE_DEVICES
TRAIN_SET="./dureader-retrieval-baseline-dataset/train/cross.train.tsv"

node=4
epoch=3
lr=1e-5
batch_size=32
train_exampls=`cat $TRAIN_SET | wc -l`
save_steps=$[$train_exampls/$batch_size/$node]
new_save_steps=$[$save_steps*$epoch/2]

#export CUDA_VISIBLE_DEVICES=2,3,4,5
#python train.py \
#python -u -m paddle.distributed.launch --gpus "2,3,4,5" train.py \
python -u -m paddle.distributed.launch --gpus "4,5,6,7" --log_dir="debug_log" train_ce.py \
        --device gpu \
        --train_set ${TRAIN_SET} \
        --save_dir ./checkpoints \
        --batch_size ${batch_size} \
        --save_steps ${new_save_steps} \
        --max_seq_len 384 \
        --learning_rate 1E-5 \
        --weight_decay  0.01 \
        --warmup_proportion 0.0 \
        --logging_steps 10 \
        --seed 1 \
        --epochs ${epoch}
        > ./log/train.log 2>&1 &
