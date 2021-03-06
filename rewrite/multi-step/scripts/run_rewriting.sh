#!/bin/bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
#SBATCH --mem=120000
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

# /home/ba63/gender-rewriting/data/rewrite/apgc-v1.0/
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/camera_ready/multi_user_with_clitics/models_f1/checkpoint-10000
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/camera_ready/single_user/models_f1/checkpoint-1000/
# /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/camera_ready/multi_user_with_clitics/augmented_models/5000/models_f1_final/checkpoint-65000

python main.py \
--data_dir /home/ba63/gender-rewriting/data/rewrite/apgc-v2.1 \
--morph_db /scratch/ba63/gender-rewriting/models/calima-msa-s31_0.4.2.db \
--bert_model /scratch/ba63/gender-rewriting/models/bert-base-camel-bert-msa-apgcv2.1 \
--gender_id_model /scratch/ba63/gender-rewriting/models/gender-id-apgcv2.1-aug \
--inference_mode test \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models/multi_user_augmented \
--top_n_best 3 \
--beam_width 10 \
--use_gpu \
--output_dir /home/ba63/gender-rewriting/rewrite/multi-step/logs/multi_user/augmentation/rewriting/CorpusR_MorphR_NeuralR_aug_GID_aug_test \
--analyze_errors \
--error_analysis_dir /home/ba63/gender-rewriting/rewrite/multi-step/logs/multi_user/augmentation/error_analysis/CorpusR_MorphR_NeuralR_aug_GID_aug_test
