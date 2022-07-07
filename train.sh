unset CUDA_VISIBLE_DEVICES

TRAIN_SET="dureader-retrieval-baseline-dataset/train/dual.train.tsv"
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
                    train_de.py \
                   --train_set_file ${TRAIN_SET} \
                   --save_dir ./checkpoint \
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