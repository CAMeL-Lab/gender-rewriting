#!/bin/bash
#SBATCH -p nvidia -q nlp 
# Set number of nodes to run
#SBATCH --nodes=1
# Set number of tasks to run
#SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=10 
# memory
#SBATCH --mem=10GB
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

module purge

export DATA_DIR=/home/ba63/gender-rewriting/data/m2_edits/v2.1
export DATA_SPLIT=dev
export DECODING=beam
export GOLD_DATA=norm_data/$DATA_SPLIT.ar.MM+$DATA_SPLIT.ar.FM+$DATA_SPLIT.ar.MF+$DATA_SPLIT.ar.FF.norm
export EDITS_ANNOTATIONS=edits/$DATA_SPLIT.arin+$DATA_SPLIT.arin+$DATA_SPLIT.arin+$DATA_SPLIT.arin.to.$DATA_SPLIT.ar.MM+$DATA_SPLIT.ar.FM+$DATA_SPLIT.ar.MF+$DATA_SPLIT.ar.FF.norm
export SYSTEM_HYP=logs/multi_user_side_constraints/$DATA_SPLIT.joint+morph.$DECODING.norm

export GOLD_ANNOTATION=$DATA_DIR/$EDITS_ANNOTATIONS

export TRG_GOLD_DATA=$DATA_DIR/$GOLD_DATA

# run M2 Scorer evaluation
eval "$(conda shell.bash hook)"
conda activate python2

m2_eval=$(python /home/ba63/gender-rewriting/m2scorer/m2scorer $SYSTEM_HYP $GOLD_ANNOTATION)

conda activate gender_rewriting

# run accuracy evaluation
accuracy=$(python utils/metrics.py --trg_directory $TRG_GOLD_DATA --pred_directory $SYSTEM_HYP)

# run BLEU evaluation
bleu=$(sacrebleu $TRG_GOLD_DATA  -i $SYSTEM_HYP -m bleu -w 2 --force)

printf "%s\n%s\n%-12s%s" "$m2_eval" "$accuracy" "BLEU" ": $bleu" > $DATA_SPLIT.joint+morph.$DECODING.norm.eval
