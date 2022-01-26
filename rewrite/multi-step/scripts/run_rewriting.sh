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

# --first_person_only \
# --use_seq2seq \
# --seq2seq_model_path neural_rewriter/saved_models \
# --top_n_best 5 \
# --beam_width 10 \
# --use_morph \
# --use_cbr \
# --cbr_ngram 2 \
# --cbr_backoff \
# --reduce_cbr_noise \
# --pick_top_mle \
# --use_rbr \
# --rbr_top_tgt_rule \
# --rbr_top_rule


# /scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
# /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm

# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/models/dev_predictions.txt
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_3_5000_acc/dev_predictions.txt

#/scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/models/test_predictions.txt
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_3_5000_acc/test_predictions.txt

# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/single_user/models_acc/dev_predictions.txt
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/single_user/models_acc/test_predictions.txt

# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/models/google_MT/dev_predictions.txt
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_3_5000_acc/google_MT/dev_predictions.txt

# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/models/google_MT/test_predictions.txt
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_3_5000_acc/google_MT/test_predictions.txt

# /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/new_token_data/

python main.py \
--data_dir /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/ \
--morph_db /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
--bert_model /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm \
--src_bert_tags_dir /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_3_5000_acc/test_predictions.txt \
--inference_mode test \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models/augmented_fix/ \
--top_n_best 5 \
--beam_width 10 \
--use_gpu \
--output_dir logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/augmentation/rewriting/CBR_MorphR_NeuralR_aug_id_aug_test \
--analyze_errors \
--error_analysis_dir logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/augmentation/error_analysis/CBR_MorphR_NeuralR_aug_id_aug_test

# --output_dir /scratch/ba63/gender-rewriting/raw_openSub/augmentation/system_output/CBR_filter_2+backoff+all+morph_newdb+mod_per_3rd_generator+neural_fix_testing
# --analyze_errors \
# --error_analysis_dir /home/ba63/gender-rewriting/rewrite/hybrid-model/logs/multi_user_with_clitics_final/error_analysis/CBR_filter_2+backoff+all+morph_newdb+mod_per_3rd_generator+neural_fix
