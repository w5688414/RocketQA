#!/bin/bash
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1
TEST_SET="./dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv"
#MODEL_PATH="./checkpoints/model_26040/model_state.pdparams"
MODEL_PATH=./cross_model/model_state.pdparams

python predict.py \
                --device 'gpu' \
                --params_path ${MODEL_PATH} \
                --test_set ${TEST_SET} \
                --batch_size 128 \
                --max_seq_length 384
