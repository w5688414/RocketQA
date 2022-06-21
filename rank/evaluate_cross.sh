MODEL_OUTPUT="result.txt"
ID_MAP="dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.id_map.tsv"
python metric/convert_rerank_res_to_json.py $MODEL_OUTPUT $ID_MAP 
REFERENCE_FIEL="dureader-retrieval-baseline-dataset/dev/dev.json"
PREDICTION_FILE="output/cross_res.json"
python metric/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE