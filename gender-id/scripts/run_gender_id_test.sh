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

#################################
# GENDER ID TEST EVAL SCRIPT
#################################

export DATA_DIR=/home/ba63/gender-rewriting/data/gender-id/multi_user
export MAX_LENGTH=128
# export OUTPUT_DIR=/scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/camera_ready/multi_user_with_clitics/models_f1/checkpoint-10000
export OUTPUT_DIR=/scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/camera_ready/multi_user_with_clitics/augmented_models/5000/models_f1/checkpoint-55000
export BATCH_SIZE=32
export SEED=12345


python gender_identifcation.py \
--data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $OUTPUT_DIR \
--output_dir $OUTPUT_DIR/ \
--max_seq_length  $MAX_LENGTH \
--per_device_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--overwrite_cache \
--do_pred \
--pred_mode dev # or test to get the test predictions
