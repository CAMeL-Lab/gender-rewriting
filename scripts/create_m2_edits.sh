#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate python2

export M2_SCORER=/home/ba63/gender-rewriting/m2scorer

if [ "$1" = "v1.0" ]; then
    export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/m2_edits/v1.0/norm_data/
    export EDITS_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/m2_edits/v1.0/edits

    printf "Creating M2 Edits for APGC v1.0:\n"

    for split in dev test
    do
        printf "\nCreating $split set M2 edits:\n\n"
        cat $DATA_DIR/D-set-$split.arin.norm \
        $DATA_DIR/D-set-$split.arin.norm \
        > $DATA_DIR/D-set-$split.arin+D-set-$split.arin.norm

        # combining the files together to create parallels
        # arin+arin --> M+F
        cat $DATA_DIR/D-set-$split.ar.M.norm \
        $DATA_DIR/D-set-$split.ar.F.norm \
        > $DATA_DIR/D-set-$split.ar.M+D-set-$split.ar.F.norm

        # creating the edits
        python  $M2_SCORER/scripts/edit_creator.py \
        $DATA_DIR/D-set-$split.arin+D-set-$split.arin.norm \
        $DATA_DIR/D-set-$split.ar.M+D-set-$split.ar.F.norm \
        > $EDITS_DIR/$split.arin+$split.arin.to.$split.ar.M+$split.ar.F.norm

        rm $DATA_DIR/D-set-$split.arin+D-set-$split.arin.norm
        rm $DATA_DIR/D-set-$split.ar.M+D-set-$split.ar.F.norm
        printf "=================================================\n"
    done


elif [ "$1" = "v2.0" ]; then
    export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/m2_edits/v2.0/norm_data/
    export EDITS_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/m2_edits/v2.0/edits

    printf "Creating M2 Edits for APGC v2.0:\n"

    # m2 edits annotations
    for split in dev test
    do
        printf "\nCreating $split set M2 edits:\n\n"

        # Taking out the <s></s> markers
        cat $DATA_DIR/$split.arin.norm | sed 's/^<s>//g' | sed 's/<\/s>$//g' \
        > $DATA_DIR/$split.arin.norm.no_marks

        cat $DATA_DIR/$split.ar.MM.norm | sed 's/^<s>//g' | sed 's/<\/s>$//g' \
        > $DATA_DIR/$split.ar.MM.norm.no_marks

        cat $DATA_DIR/$split.ar.FM.norm | sed 's/^<s>//g' | sed 's/<\/s>$//g' \
        > $DATA_DIR/$split.ar.FM.norm.no_marks

        cat $DATA_DIR/$split.ar.MF.norm | sed 's/^<s>//g' | sed 's/<\/s>$//g' \
        > $DATA_DIR/$split.ar.MF.norm.no_marks

        cat $DATA_DIR/$split.ar.FF.norm | sed 's/^<s>//g' | sed 's/<\/s>$//g' \
        > $DATA_DIR/$split.ar.FF.norm.no_marks

        # combining the files together to create parallels
        # arin+arin+arin+arin --> MM+FM+MF+FF
        cat $DATA_DIR/$split.arin.norm.no_marks \
        $DATA_DIR/$split.arin.norm.no_marks \
        $DATA_DIR/$split.arin.norm.no_marks \
        $DATA_DIR/$split.arin.norm.no_marks \
        > $DATA_DIR/$split.arin+$split.arin+$split.arin+$split.arin.norm.no_marks

        cat $DATA_DIR/$split.ar.MM.norm.no_marks \
        $DATA_DIR/$split.ar.FM.norm.no_marks \
        $DATA_DIR/$split.ar.MF.norm.no_marks\
        $DATA_DIR/$split.ar.FF.norm.no_marks \
        > $DATA_DIR/$split.ar.MM+$split.ar.FM+$split.ar.MF+$split.ar.FF.norm.no_marks

        # creating the edits
        python $M2_SCORER/scripts/edit_creator.py \
        $DATA_DIR/$split.arin+$split.arin+$split.arin+$split.arin.norm.no_marks \
        $DATA_DIR/$split.ar.MM+$split.ar.FM+$split.ar.MF+$split.ar.FF.norm.no_marks \
        > $EDITS_DIR/$split.arin+$split.arin+$split.arin+$split.arin.to.$split.ar.MM+$split.ar.FM+$split.ar.MF+$split.ar.FF.norm


        rm $DATA_DIR/$split.arin+$split.arin+$split.arin+$split.arin.norm.no_marks
        rm $DATA_DIR/$split.ar.MM+$split.ar.FM+$split.ar.MF+$split.ar.FF.norm.no_marks
        printf "=================================================\n"
    done

else
    printf "APGC version not found: Please choose v1.0 or v2.0!\n"
fi



