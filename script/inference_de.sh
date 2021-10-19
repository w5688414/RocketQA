
export PYTHONPATH="/home/tianxin04/PaddleNLP/"

# cuda-11.2
#PYTHON_BIN="python"

# cuda-10.1 paddle-2.2
# PYTHON_BIN="/usr/local/bin/python3.8"

# cuda-10.1 paddle-2.1
PYTHON_BIN="/usr/local/bin/python3.7"

test_data="./test_data"

unset CUDA_VISIBLE_DEVICES
${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "7" \
    inference_de.py \
    --device gpu \
    --params_path "./model_13330/model_state.pdparams" \
    --batch_size 128 \
    --max_seq_length 128 \
    --text_file "./demo_data/test_data" \
    --output_file "para_embeddings"
