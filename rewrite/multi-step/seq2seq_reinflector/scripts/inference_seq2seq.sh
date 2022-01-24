#!/bin/bash
#SBATCH -p condo 
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

export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/
# export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/new_token_data/

python main.py \
 --data_dir $DATA_DIR \
 --embed_dim 128 \
 --add_side_constraints \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --seed 21 \
 --model_path saved_models/checking/joint.pt \
 --do_inference \
 --inference_mode dev \
 --beam_size 10 \
 --n_best 5 \
 --preds_dir logs/reinflection/dev.checking.joint
