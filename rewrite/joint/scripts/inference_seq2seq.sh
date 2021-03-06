#!/bin/bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=20GB
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

export DATA_DIR=/home/ba63/gender-rewriting/data/rewrite/apgc-v2.1/joint

#  --first_person_only \
#  --add_side_constraints \
#  --embed_trg_gender \
#  --trg_gender_embed_dim 10 \
#  --use_morph_features 
#  --analyzer_db_path /scratch/ba63/gender-rewriting/models/calima-msa-s31_0.4.2.db \

python main.py \
 --data_dir $DATA_DIR \
 --add_side_constraints \
 --use_morph_features \
 --analyzer_db_path /scratch/ba63/gender-rewriting/models/calima-msa-s31_0.4.2.db \
 --embed_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --seed 21 \
 --model_path saved_models/multi_user_side_constraints/joint+morph.pt \
 --do_inference \
 --inference_mode dev \
 --preds_dir logs/multi_user_side_constraints/dev.joint+morph
