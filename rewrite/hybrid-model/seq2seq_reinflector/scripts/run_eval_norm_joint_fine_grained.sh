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

module purge


export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/m2_edits/v2.0/
export DATA_SPLIT=dev
export DECODING=beam


for exp in BB MB.FB BM.BF MM.FM.MF.FF
do

    export GOLD_DATA=norm_data_fine_grained/$DATA_SPLIT.ar.MM+$DATA_SPLIT.ar.FM+$DATA_SPLIT.ar.MF+$DATA_SPLIT.ar.FF.$exp.norm
    export EDITS_ANNOTATIONS=fine_grained_edits/$DATA_SPLIT.arin+$DATA_SPLIT.arin+$DATA_SPLIT.arin+$DATA_SPLIT.arin.to.$DATA_SPLIT.ar.MM+$DATA_SPLIT.ar.FM+$DATA_SPLIT.ar.MF+$DATA_SPLIT.ar.FF.$exp.norm
    export SYSTEM_HYP=logs/reinflection/multi_user_side_constraints_newdb/fine_grained_output/$DATA_SPLIT.joint+morph.$DECODING.$exp.norm

    export GOLD_ANNOTATION=$DATA_DIR/$EDITS_ANNOTATIONS

    export TRG_GOLD_DATA=$DATA_DIR/$GOLD_DATA

    echo "Evaluating $SYSTEM_HYP..\n"

    # run M2 Scorer evaluation
    eval "$(conda shell.bash hook)"
    conda activate python2

    m2_eval=$(python /home/ba63/gender-rewriting/m2scorer/m2scorer $SYSTEM_HYP $GOLD_ANNOTATION)

    conda activate python3

    # run accuracy evaluation
    accuracy=$(python utils/metrics.py --trg_directory $TRG_GOLD_DATA --pred_directory $SYSTEM_HYP)

    # run BLEU evaluation
    bleu=$(sacrebleu $TRG_GOLD_DATA  -i $SYSTEM_HYP -m bleu --force)

    printf "%s\n%s\n%-12s%s" "$m2_eval" "$accuracy" "BLEU" ": $bleu" > eval.$DECODING.$exp.norm

done
