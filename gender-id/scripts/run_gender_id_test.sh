#!/bin/bash
#SBATCH -p condo
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

export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/new_token_data/gender_tagger_data
# export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/gender_tagger_data/new_token_data
# export DATA_DIR=/scratch/ba63/gender-rewriting/raw_openSub/augmentation
export MAX_LENGTH=128
# export OUTPUT_DIR=/scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_10_5000_acc
export OUTPUT_DIR=/scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/first_person/models_acc
export BATCH_SIZE=32
export SEED=12345


python gender_identifcation.py \
--data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $OUTPUT_DIR \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--per_device_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--overwrite_cache \
--do_pred \
--pred_mode test # or dev to get the dev predictions
