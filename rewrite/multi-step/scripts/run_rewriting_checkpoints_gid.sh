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


# for gid_model in /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/camera_ready/multi_user_with_clitics/models_f1/checkpoint*

# do

#     python main.py \
#     --data_dir /home/ba63/gender-rewriting/data/rewrite/apgc-v2.0/ \
#     --morph_db /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
#     --bert_model  /scratch/ba63/gender-rewriting/mlm_lm/bert-base-arabic-camelbert-msa-mlm-88 \
#     --gender_id_model $gid_model \
#     --inference_mode dev \
#     --use_cbr \
#     --cbr_ngram 2 \
#     --cbr_backoff \
#     --reduce_cbr_noise \
#     --use_gpu \
#     --output_dir $gid_model
# done

for gid_model in /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/camera_ready/multi_user_with_clitics/augmented_models/5000/models_f1_final/checkpoint*

do

    python main.py \
    --data_dir /home/ba63/gender-rewriting/data/rewrite/apgc-v2.0/ \
    --morph_db /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
    --bert_model  /scratch/ba63/gender-rewriting/mlm_lm/bert-base-arabic-camelbert-msa-mlm-88 \
    --gender_id_model $gid_model \
    --inference_mode dev \
    --use_cbr \
    --cbr_ngram 2 \
    --cbr_backoff \
    --reduce_cbr_noise \
    --use_morph \
    --use_seq2seq \
    --seq2seq_model_path neural_rewriter/saved_models/multi_user \
    --top_n_best 3 \
    --beam_width 10 \
    --use_gpu \
    --output_dir $gid_model
done

# for gid_model in /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/camera_ready/single_user/models_f1/checkpoint*

# do

#     python main.py \
#     --data_dir /home/ba63/gender-rewriting/data/rewrite/apgc-v1.0/ \
#     --morph_db /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
#     --first_person_only \
#     --bert_model  /scratch/ba63/gender-rewriting/mlm_lm/bert-base-arabic-camelbert-msa-mlm-88 \
#     --gender_id_model $gid_model \
#     --inference_mode dev \
#     --use_cbr \
#     --cbr_ngram 2 \
#     --cbr_backoff \
#     --reduce_cbr_noise \
#     --use_morph \
#     --use_seq2seq \
#     --seq2seq_model_path neural_rewriter/saved_models/single_user \
#     --top_n_best 3 \
#     --beam_width 10 \
#     --use_gpu \
#     --output_dir $gid_model
# done