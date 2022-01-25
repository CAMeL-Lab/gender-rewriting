#!/bin/bash
#SBATCH -p condo
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=100GB
# Walltime format hh:mm:ss
#SBATCH --time=59:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

# export DATA_DIR=/home/ba63/gender-reinflection/data/alhafni/
#  --embed_trg_gender \
#  --trg_gender_embed_dim 10 \
#  --first_person_only \
#  --use_morph_features \
#  --analyzer_db_path /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
#  --morph_features_path saved_models/morph_features_top_1_analyses.json \
#  --add_side_constraints \
#  --visualize_loss \
#  --do_early_stopping \

# export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/new_token_data/
# export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/
# export DATA_DIR=/scratch/ba63/gender-rewriting/raw_openSub/augmentation/augmented_data/CBR+backoff+all+morph_newdb+mod_per_3rd_generator+neural_mlm3_augmented
# export DATA_DIR=/scratch/ba63/gender-rewriting/raw_openSub/augmentation/augmented_data/CBR+backoff+all+morph_newdb+mod_per_3rd_generator+neural_augmented
export DATA_DIR=/scratch/ba63/gender-rewriting/raw_openSub/augmentation/augmented_data/CBR_filter_2+backoff+all+morph_newdb+mod_per_3rd_generator+neural_fix_augmented
# export DATA_DIR=/scratch/ba63/gender-rewriting/raw_openSub/augmentation/augmented_data/CBR_filter+backoff+all+morph_newdb+mod_per_3rd_generator+neural_MLM_with_seed

python main.py \
 --data_dir $DATA_DIR \
 --add_side_constraints \
 --analyzer_db_path /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
 --vectorizer_path saved_models/augmented_fix_mlm_with_seed/vectorizer.json \
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
 --model_path saved_models/augmented_fix/joint.pt