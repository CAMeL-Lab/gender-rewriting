#!/bin/bash
#SBATCH -p nvidia -q nlp
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=100GB
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

#  --embed_trg_gender \
#  --trg_gender_embed_dim 10 \
#  --first_person_only \
#  --add_side_constraints \
#  --use_morph_features \
#  --analyzer_db_path /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
#  --morph_features_path saved_models/multi_user_side_constraints_newdb_clean_train/morph_features_top_1_analyses.json \

export DATA_DIR=/home/ba63/gender-rewriting-camera-ready/data/rewrite/apgc-v2.0/joint

python main.py \
 --data_dir $DATA_DIR \
 --embed_trg_gender \
 --trg_gender_embed_dim 10 \
 --vectorizer_path saved_models/multi_user_check/vectorizer.json \
  --use_morph_features \
 --analyzer_db_path /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
 --morph_features_path saved_models/multi_user_check/morph_features_top_1_analyses.json \
 --cache_files \
 --num_train_epochs 50 \
 --embed_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --batch_size 32 \
 --use_cuda \
 --seed 21 \
 --do_train \
 --dropout 0.2 \
 --clip_grad 1.0 \
 --do_early_stopping \
 --model_path saved_models/multi_user_check/joint+morph.pt
