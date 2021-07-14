CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/berta-base
export GLUE_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="tnews"

/home/jiaweili/anaconda3/envs/pytorch1.6/bin/python run_classifier.py \
  --data_dir='sentiment-analysis-on-movie-reviews/' \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_kd \
  --do_lower_case \
  --max_seq_length=50 \
  --per_gpu_train_batch_size=256 \
  --per_gpu_eval_batch_size=256 \
  --learning_rate=1e-5 \
  --weight_decay=0 \
  --num_train_epochs=5 \
  --logging_steps=300 \
  --save_steps=300 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=1 \
  --gpu=1 \
  --run_id=1 \
  --kd_coeff=0
