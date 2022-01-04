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
export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/new_token_data/gender_tagger_data
# export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/gender_tagger_data/new_token_data
# export DATA_DIR=/scratch/ba63/gender-rewriting/raw_openSub/augmentation/augmented_data
export MAX_LENGTH=128
export BERT_MODEL=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa/
# export OUTPUT_DIR=/scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/models_acc
export OUTPUT_DIR=/scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/first_person/models_f1
export BATCH_SIZE=32
export NUM_EPOCHS=10
export SAVE_STEPS=500
export EVAL_STEPS=500
export SEED=12345

# --load_best_model_at_end \
# --metric_for_best_model f1_macro \
# --eval_steps $EVAL_STEPS \
# --evaluation_strategy steps \

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
