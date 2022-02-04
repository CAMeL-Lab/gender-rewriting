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

# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/models
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_3_5000_acc

# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/single_user/models_acc

# /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/new_token_data/

python main.py \
--data_dir /home/ba63/gender-rewriting/data/rewrite/apgc-v2.0/ \
--morph_db /home/ba63/gender-rewriting/data/utils/calima-msa-s31_0.4.2.utf8.db.copy-mod \
--gender_id_model /home/ba63/gender-rewriting-models/gender-id/multi_user/augmented \
--bert_model /home/ba63/gender-rewriting-models/bert-base-arabic-camelbert-msa-mlm \
--inference_mode dev \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--save_cbr_model \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models/augmented_fix/ \
--top_n_best 5 \
--beam_width 10 \
--use_gpu \
--output_dir logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/augmentation/rewriting/CBR_MorphR_NeuralR_aug_id_aug_checking \
--analyze_errors \
--error_analysis_dir logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/augmentation/error_analysis/CBR_MorphR_NeuralR_aug_id_aug_checking

# --output_dir /scratch/ba63/gender-rewriting/raw_openSub/augmentation/system_output/CBR_filter_2+backoff+all+morph_newdb+mod_per_3rd_generator+neural_fix_testing
# --analyze_errors \
# --error_analysis_dir /home/ba63/gender-rewriting/rewrite/hybrid-model/logs/multi_user_with_clitics_final/error_analysis/CBR_filter_2+backoff+all+morph_newdb+mod_per_3rd_generator+neural_fix
