DATA_PATH="dureader-retrieval-baseline-dataset/passage-collection"
TOP_K=50
para_part_cnt=`cat $DATA_PATH/part-00 | wc -l`
python merge.py $para_part_cnt $TOP_K 4 

QUERY2ID="dureader-retrieval-baseline-dataset/dev/q2qid.dev.json"
PARA2ID="dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json"
MODEL_OUTPUT="output/dev.res.top50"
python metric/convert_recall_res_to_json.py $QUERY2ID $PARA2ID $MODEL_OUTPUT


REFERENCE_FIEL="dureader-retrieval-baseline-dataset/dev/dev.json"
PREDICTION_FILE="output/dual_res.json"
python metric/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE