# export CUDA_VISIBLE_DEVICES=3,6
# .5 服务器
unset CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_sync_nccl_allreduce=1
# export FLAGS_fraction_of_gpu_memory_to_use=0.95
# CHECKPOINT_PATH=model_17370/
CHECKPOINT_PATH=model_26055/
TEST_SET="dureader-retrieval-baseline-dataset/dev/dev.q.format" 
DATA_PATH="dureader-retrieval-baseline-dataset/passage-collection"
CUDA_VISIBLE_DEVICES=4 python inference_de_query.py --text_file $TEST_SET \
                    --output_file output \
                    --params_path checkpoint/${CHECKPOINT_PATH}/query_model_state.pdparams \
                    --output_emb_size 0 \
                    --batch_size 256 \
                    --max_seq_length 32

# extract para
for part in 0 1 2 3;do
    # nohup python src/index_search.py $part $TOP_K $QUERY_FILE >> output/test.log &
    TASK_DATA_PATH=${DATA_PATH}/part-0${part}
    count=$((part+4))
    CUDA_VISIBLE_DEVICES=${count} nohup python inference_de_title.py --text_file $TASK_DATA_PATH \
                        --output_file output \
                        --params_path checkpoint/${CHECKPOINT_PATH}/title_model_state.pdparams \
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
    CUDA_VISIBLE_DEVICES=4 python index_search.py $part $TOP_K $TEST_SET
done

# extrace para
# TEST_SET="dureader-retrieval-baseline-dataset/passage-collection/test_para"
# python inference_de_title.py --text_file $TEST_SET \
#                     --output_file output \
#                     --params_path checkpoint/model_8685/title_model_state.pdparams \
#                     --output_emb_size 0 \
#                     --batch_size 256 \
#                     --max_seq_length 384