
export PYTHONPATH="/home/tianxin04/PaddleNLP/"
PYTHON_BIN="/usr/local/bin/python3.7"
train_data="./train_data/marco_merge_de2_denoise.tsv.std"

# ${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "0" \
#bs=512
bs=32
lr=3E-5
q_max_len=32
p_max_len=128
warmup_proportion=0.1
epoch=10

export CUDA_VISIBLE_DEVICES=7
${PYTHON_BIN} -u \
    ./src/train_de.py \
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
