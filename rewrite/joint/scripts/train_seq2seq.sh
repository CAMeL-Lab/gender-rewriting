#!/bin/bash
#SBATCH -p condo
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

# export DATA_DIR=/home/ba63/gender-reinflection/data/alhafni/
#  --embed_trg_gender \
#  --trg_gender_embed_dim 10 \
#  --first_person_only \
# --use_morph_features \
export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/
# export DATA_DIR=/home/ba63/gender-reinflection/data/alhafni/

python main.py \
 --data_dir $DATA_DIR \
 --analyzer_db_path /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
 --use_morph_features \
 --morph_features_path saved_models/multi_user_side_constraints_newdb/morph_features_top_1_analyses.json \
 --add_side_constraints \
 --vectorizer_path saved_models/multi_user_side_constraints_newdb/vectorizer.json \
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
 --visualize_loss \
 --model_path saved_models/multi_user_side_constraints_newdb/joint+morph.pt
