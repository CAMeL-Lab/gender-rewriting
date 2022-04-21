#!/bin/bash
#SBATCH -p nvidia 
# use gpus
#SBATCH --gres=gpu:v100:1 
# memory
#SBATCH --mem=100GB
# Walltime format hh:mm:ss
#SBATCH --time=59:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

export DATA_DIR=/home/ba63/gender-rewriting/data/rewrite/apgc-v2.0/
# export DATA_DIR=/home/ba63/gender-rewriting/data/rewrite/apgc-v2.0/augmentation/

python main.py \
 --data_dir $DATA_DIR \
 --add_side_constraints \
 --vectorizer_path saved_models/multi_user/vectorizer.json \
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
 --model_path saved_models/multi_user/joint.pt
