#!/bin/bash
#SBATCH -p nvidia 
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120000
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

export REINFLECTION_TYPE=multi_user
# --first_person_only \
# --use_morph \
# /scratch/ba63/gender-identification/CAMeLBERT_MSA/$REINFLECTION_TYPE/checkpoint-500-best/dev_predictions.txt
# /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/

python main.py \
--data_dir /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/ \
--morph_db /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db \
--bert_model /scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa \
--src_bert_tags_dir /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/models/dev_predictions.txt \
--inference_mode dev \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--output_dir logs/debugging/multi_user_with_clitics/reinflection/CBR+backoff+all \
--error_analysis_dir logs/debugging/multi_user_with_clitics/error_analysis/CBR+backoff+all