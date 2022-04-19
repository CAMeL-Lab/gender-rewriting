#!/bin/bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120000
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge


################################
# GENDER ID FINE-TUNING SCRIPT
################################
# export DATA_DIR=/home/ba63/gender-rewriting/data/gender-id/single_user
# export DATA_DIR=/home/ba63/gender-rewriting/data/gender-id/multi_user/
export DATA_DIR=/home/ba63/gender-rewriting/data/gender-id/multi_user/augmented_data
export MAX_LENGTH=128
export BERT_MODEL=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa/
# export OUTPUT_DIR=/scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/camera_ready/single_user/models_f1
export OUTPUT_DIR=/scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/camera_ready/multi_user_with_clitics/augmented_models/5000/models_f1
# export OUTPUT_DIR=/scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/single_user/models_f1

export BATCH_SIZE=32
export NUM_EPOCHS=3
# export SAVE_STEPS=500
# export EVAL_STEPS=500
export SAVE_STEPS=5000
export EVAL_STEPS=5000
export SEED=12345

python gender_identifcation.py \
--data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--eval_steps $EVAL_STEPS \
--evaluation_strategy steps \
--seed $SEED \
--do_train \
--do_eval \
--load_best_model_at_end \
--metric_for_best_model f1_macro \
--overwrite_output_dir \
--overwrite_cache \
