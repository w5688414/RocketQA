# export CUDA_VISIBLE_DEVICES=1,2,4,5

# export PYTHONPATH=/wugaosheng/Paddle/python:$PYTHONPATH
# export PYTHONPATH=/wugaosheng/Paddle:$PYTHONPATH
# export PYTHONPATH=/wugaosheng/Paddle
unset CUDA_VISIBLE_DEVICES
set -x

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
# export FLAGS_fraction_of_gpu_memory_to_use=0.95

# export GLOG_v=1

# TRAIN_SET="dureader-retrieval-baseline-dataset/train/dual.train.tsv"
TRAIN_SET="dureader-retrieval-baseline-dataset/train/dual.train.tsv"
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
                    train_de.py \
                   --train_set_file ${TRAIN_SET} \
                   --batch_size 128 \
                   --save_steps 8685 \
                   --query_max_seq_length 32 \
                   --title_max_seq_length 384 \
                   --learning_rate 3e-5 \
                   --epochs 10 \
                   --weight_decay 0.0 \
                   --warmup_proportion 0.1 \
                   --use_cross_batch \
                   --seed 1 \
                   --use_amp