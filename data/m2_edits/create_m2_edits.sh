#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate python2

export M2_SCORER=/home/ba63/gender-rewriting/m2scorer

if [ "$1" = "v1.0" ]; then
    export DATA_DIR=v1.0/norm_data
    export EDITS_DIR=v1.0/edits

    printf "Creating M2 Edits for APGC v1.0:\n"

    for split in dev test
    do
        printf "\nCreating $split set M2 edits:\n\n"
        # creating the edits
        # arin+arin --> M+F
        python  $M2_SCORER/scripts/edit_creator.py \
        $DATA_DIR/D-set-$split.arin+D-set-$split.arin.norm \
        $DATA_DIR/D-set-$split.ar.M+D-set-$split.ar.F.norm \
        > $EDITS_DIR/$split.arin+$split.arin.to.$split.ar.M+$split.ar.F.norm

        printf "=================================================\n"
    done


elif [ "$1" = "v2.1" ]; then
    export DATA_DIR=v2.1/norm_data_check/
    export EDITS_DIR=v2.1/edits_check

    printf "Creating M2 Edits for APGC v2.1:\n"

    # m2 edits annotations
    for split in dev test
    do
        printf "\nCreating $split set M2 edits:\n\n"

        # creating the edits
        # arin+arin+arin+arin --> MM+FM+MF+FF
        python $M2_SCORER/scripts/edit_creator.py \
        $DATA_DIR/$split.arin+$split.arin+$split.arin+$split.arin.norm \
        $DATA_DIR/$split.ar.MM+$split.ar.FM+$split.ar.MF+$split.ar.FF.norm \
        > $EDITS_DIR/$split.arin+$split.arin+$split.arin+$split.arin.to.$split.ar.MM+$split.ar.FM+$split.ar.MF+$split.ar.FF.norm

        printf "=================================================\n"
    done

else
    printf "APGC version not found: Please choose v1.0 or v2.1!\n"
fi
