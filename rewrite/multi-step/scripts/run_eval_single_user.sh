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

export EXPERIMENT=CorpusR_MorphR_NeuralR_test
export SYSTEM_HYP=/home/ba63/gender-rewriting/rewrite/multi-step/logs/single_user/rewriting/$EXPERIMENT

# preparing the preds
cat $SYSTEM_HYP/arin.to.M.preds $SYSTEM_HYP/arin.to.F.preds > $SYSTEM_HYP/$EXPERIMENT.inf

# normalizing the preds
python /home/ba63/gender-rewriting/rewrite/multi-step/utils/normalize.py --input_file $SYSTEM_HYP/$EXPERIMENT.inf --output_file $SYSTEM_HYP/$EXPERIMENT.inf.norm


export DATA_DIR=/home/ba63/gender-rewriting/data/m2_edits/v1.0/
export DATA_SPLIT=test
export GOLD_DATA=norm_data/D-set-$DATA_SPLIT.ar.M+D-set-$DATA_SPLIT.ar.F.norm
export EDITS_ANNOTATIONS=edits/$DATA_SPLIT.arin+$DATA_SPLIT.arin.to.$DATA_SPLIT.ar.M+$DATA_SPLIT.ar.F.norm
export GOLD_ANNOTATION=$DATA_DIR/$EDITS_ANNOTATIONS
export TRG_GOLD_DATA=$DATA_DIR/$GOLD_DATA


eval "$(conda shell.bash hook)"
conda activate python2

m2_eval=$(python /home/ba63/gender-rewriting/m2scorer/m2scorer $SYSTEM_HYP/$EXPERIMENT.inf.norm $GOLD_ANNOTATION)

conda activate gender_rewriting

# run accuracy evaluation
accuracy=$(python /home/ba63/gender-rewriting/rewrite/joint/utils/metrics.py --trg_directory $TRG_GOLD_DATA --pred_directory $SYSTEM_HYP/$EXPERIMENT.inf.norm)

# run BLEU evaluation
bleu=$(sacrebleu $TRG_GOLD_DATA  -i $SYSTEM_HYP/$EXPERIMENT.inf.norm -m bleu -w 2 --force)

printf "%s\n%s\n%-12s%s" "$m2_eval" "$accuracy" "BLEU" ": $bleu" > eval.$EXPERIMENT

