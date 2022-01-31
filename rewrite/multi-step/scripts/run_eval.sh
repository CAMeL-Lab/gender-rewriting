#!/bin/bash
#SBATCH -p condo 
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

export EXPERIMENT=CBR_MorphR_NeuralR_aug_id_aug_test
export SYSTEM_HYP=logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/augmentation/rewriting/$EXPERIMENT

# removing last empty line from the preds files
# sed -i '$ d' $SYSTEM_HYP/arin.to.MM.preds
# sed -i '$ d' $SYSTEM_HYP/arin.to.FM.preds
# sed -i '$ d' $SYSTEM_HYP/arin.to.MF.preds
# sed -i '$ d' $SYSTEM_HYP/arin.to.FF.preds

# preparing the preds
cat $SYSTEM_HYP/arin.to.MM.preds $SYSTEM_HYP/arin.to.FM.preds  $SYSTEM_HYP/arin.to.MF.preds  $SYSTEM_HYP/arin.to.FF.preds > $SYSTEM_HYP/$EXPERIMENT.inf

# normalizing the preds
python /home/ba63/gender-rewriting/rewrite/multi-step/utils/normalize.py --input_file $SYSTEM_HYP/$EXPERIMENT.inf --output_file $SYSTEM_HYP/$EXPERIMENT.inf.norm


export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/m2_edits/v2.0/
export DATA_SPLIT=test
export GOLD_DATA=norm_data/$DATA_SPLIT.ar.MM+$DATA_SPLIT.ar.FM+$DATA_SPLIT.ar.MF+$DATA_SPLIT.ar.FF.norm
export EDITS_ANNOTATIONS=edits/$DATA_SPLIT.arin+$DATA_SPLIT.arin+$DATA_SPLIT.arin+$DATA_SPLIT.arin.to.$DATA_SPLIT.ar.MM+$DATA_SPLIT.ar.FM+$DATA_SPLIT.ar.MF+$DATA_SPLIT.ar.FF.norm
export GOLD_ANNOTATION=$DATA_DIR/$EDITS_ANNOTATIONS
export TRG_GOLD_DATA=$DATA_DIR/$GOLD_DATA


eval "$(conda shell.bash hook)"
conda activate python2

m2_eval=$(python /home/ba63/gender-rewriting/m2scorer/m2scorer $SYSTEM_HYP/$EXPERIMENT.inf.norm $GOLD_ANNOTATION)

conda activate python3

# run accuracy evaluation
accuracy=$(python /home/ba63/gender-rewriting/rewrite/joint/utils/metrics.py --trg_directory $TRG_GOLD_DATA --pred_directory $SYSTEM_HYP/$EXPERIMENT.inf.norm)

# run BLEU evaluation
bleu=$(sacrebleu $TRG_GOLD_DATA  -i $SYSTEM_HYP/$EXPERIMENT.inf.norm -m bleu --force)

printf "%s\n%s\n%-12s%s" "$m2_eval" "$accuracy" "BLEU" ": $bleu" > eval.$EXPERIMENT

