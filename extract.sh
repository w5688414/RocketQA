unset CUDA_VISIBLE_DEVICES
CHECKPOINT_PATH=model_17370/
TEST_SET="dureader-retrieval-baseline-dataset/dev/dev.q.format" 
DATA_PATH="dureader-retrieval-baseline-dataset/passage-collection"
CUDA_VISIBLE_DEVICES=0 python inference_de.py --text_file $TEST_SET \
                    --output_file output \
                    --params_path checkpoint_single/${CHECKPOINT_PATH}/model_state.pdparams \
                    --output_emb_size 0 \
                    --mode query \
                    --batch_size 256 \
                    --max_seq_length 32
# extract para
for part in 0 1 2 3;do
    TASK_DATA_PATH=${DATA_PATH}/part-0${part}
    count=$((part))
    CUDA_VISIBLE_DEVICES=${count} nohup python inference_de.py --text_file $TASK_DATA_PATH \
                        --output_file output \
                        --mode title \
                        --params_path checkpoint_single/${CHECKPOINT_PATH}/model_state.pdparams \
                        --output_emb_size 0 \
                        --batch_size 256 \
                        --max_seq_length 384 >> output/test.log &
    pid[$part]=$!
    echo $part start: pid=$! >> output/test.log
done
wait

TOP_K=50
TEST_SET="dureader-retrieval-baseline-dataset/dev/dev.q.format"
python build_index.py
for part in 0 1 2 3;do
    CUDA_VISIBLE_DEVICES=1 python index_search.py $part $TOP_K $TEST_SET
done
