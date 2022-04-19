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

export TRAIN_DATA_FILE=/home/ba63/gender-rewriting/data/mlm/train.txt
export DEV_DATA_FILE=/home/ba63/gender-rewriting/data/mlm/dev.txt
export MODEL=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
export OUTPUT_DIR=/scratch/ba63/gender-rewriting/mlm_lm/bert-base-arabic-camelbert-msa-mlm-88

python run_mlm_no_trainer.py \
--model_name_or_path $MODEL \
--train_file $TRAIN_DATA_FILE \
--validation_file $DEV_DATA_FILE \
--num_train_epochs 3 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--line_by_line True \
--overwrite_cache True \
--seed 88 \
--output_dir $OUTPUT_DIR
