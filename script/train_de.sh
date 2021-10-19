
export PYTHONPATH="/home/tianxin04/PaddleNLP/"

# cuda-11.2
#PYTHON_BIN="python"

# cuda-10.1 paddle-2.2
PYTHON_BIN="/usr/local/bin/python3.8"

# cuda-10.1 paddle-2.1
# PYTHON_BIN="/usr/local/bin/python3.7"

train_data="./data_train/marco_merge_de2_denoise.tsv.std"

# ${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "0" \
bs=512
lr=3E-5
q_max_len=32
p_max_len=128
warmup_proportion=0.1
epoch=10

#export CUDA_VISIBLE_DEVICES=7
#${PYTHON_BIN} -u \

unset CUDA_VISIBLE_DEVICES
${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "2,3" --log_dir "logs" \
    train_de.py \
    --device gpu \
    --save_dir ./checkpoints/ \
    --batch_size ${bs} \
    --learning_rate ${lr} \
    --warmup_proportion ${warmup_proportion} \
    --epochs ${epoch} \
    --save_steps 10000 \
    --query_max_seq_length ${q_max_len} \
    --title_max_seq_length ${p_max_len} \
    --train_set_file ${train_data} \
    --use_cross_batch
