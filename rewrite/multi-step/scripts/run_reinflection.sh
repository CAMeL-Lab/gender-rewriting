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

export REINFLECTION_TYPE=multi_user
# --first_person_only \
# --use_seq2seq \
# --seq2seq_model_path seq2seq_reinflector/saved_models \
# --top_n_best 5 \
# --beam_width 10 \
# --use_morph \
# --use_cbr \
# --cbr_ngram 2 \
# --cbr_backoff \
# --pick_top_mle \
# --use_rbr \
# --rbr_top_tgt_rule \
# --rbr_top_rule

# /scratch/ba63/gender-identification/CAMeLBERT_MSA/$REINFLECTION_TYPE/checkpoint-500-best/dev_predictions.txt
# /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/

# /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db
# /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod

# /scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
# /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm
# /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm-3
# /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm-augmented-10
# /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm-augmented-3

# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics_new/models_old_trans_10/checkpoint-10000/dev_predictions.txt
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_3_5000_acc/dev_predictions.txt
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/models/dev_predictions.txt
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/first_person/models_acc/dev_predictions.txt

# /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/new_token_data/
# /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm
# /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm-checking

python main.py \
--data_dir /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/ \
--morph_db /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
--bert_model /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm-checking-testing \
--src_bert_tags_dir /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_3_5000_acc/dev_predictions.txt \
--inference_mode dev \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path seq2seq_reinflector/saved_models/augmented_fix_mlm_with_seed \
--top_n_best 5 \
--beam_width 10 \
--use_gpu \
--output_dir logs/paper_results_with_mlm_ft_with_seed/augmentation/reinflection/CBR_filter+backoff+all+morph_newdb+mod_per_3rd_generator+neural_augmented_id_augmented \
--analyze_errors \
--error_analysis_dir logs/paper_results_with_mlm_ft_with_seed/augmentation/error_analysis/CBR_filter+backoff+all+morph_newdb+mod_per_3rd_generator+neural_augmented_id_augmented

# --output_dir /scratch/ba63/gender-rewriting/raw_openSub/augmentation/system_output/CBR_filter_2+backoff+all+morph_newdb+mod_per_3rd_generator+neural_fix_testing
# --analyze_errors \
# --error_analysis_dir /home/ba63/gender-rewriting/rewrite/hybrid-model/logs/multi_user_with_clitics_final/error_analysis/CBR_filter_2+backoff+all+morph_newdb+mod_per_3rd_generator+neural_fix
