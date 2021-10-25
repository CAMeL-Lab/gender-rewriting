
eval "$(conda shell.bash hook)"
conda activate python2

export M2_SCORER=/home/ba63/gender-rewriting/m2scorer

if [ "$1" = "v1.0" ]; then
    export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/
    export EDITS_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/m2_edits/v1.0-edits

    printf "Creating M2 Edits for APGC v1.0:\n"
    printf "\nCreating dev set M2 edits:\n\n"
    # Dev set annotations
    cat $DATA_DIR/D-set-dev.arin $DATA_DIR/D-set-dev.arin > $DATA_DIR/D-set-dev.arin+D-set-dev.arin
    cat $DATA_DIR/D-set-dev.ar.M $DATA_DIR/D-set-dev.ar.F > $DATA_DIR/D-set-dev.ar.M+D-set-dev.ar.F

    python  $M2_SCORER/scripts/edit_creator.py \
    $DATA_DIR/D-set-dev.arin+D-set-dev.arin \
    $DATA_DIR/D-set-dev.ar.M+D-set-dev.ar.F \
    > $EDITS_DIR/dev.arin+dev.arin.to.dev.ar.M+dev.ar.F

    rm $DATA_DIR/D-set-dev.arin+D-set-dev.arin
    rm $DATA_DIR/D-set-dev.ar.M+D-set-dev.ar.F

    printf "================================================="
    printf "\nCreating test set M2 edits:\n\n"

    # Test set annotations
    cat $DATA_DIR/D-set-test.arin \
    $DATA_DIR/D-set-test.arin \
    > $DATA_DIR/D-set-test.arin+D-set-test.arin

    cat $DATA_DIR/D-set-test.ar.M \
    $DATA_DIR/D-set-test.ar.F \
    > $DATA_DIR/D-set-test.ar.M+D-set-test.ar.F

    python  $M2_SCORER/scripts/edit_creator.py \
    $DATA_DIR/D-set-test.arin+D-set-test.arin \
    $DATA_DIR/D-set-test.ar.M+D-set-test.ar.F \
    > $EDITS_DIR/test.arin+test.arin.to.test.ar.M+test.ar.F

    rm $DATA_DIR/D-set-test.arin+D-set-test.arin
    rm $DATA_DIR/D-set-test.ar.M+D-set-test.ar.F

elif [ "$1" = "v2.0" ]; then
    export DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data
    export EDITS_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/m2_edits/v2.0-edits

    printf "Creating M2 Edits for APGC v2.0:\n"
    printf "\nCreating dev set M2 edits:\n\n"

    # Dev set annotations
    cat $DATA_DIR/dev/dev.arin \
    $DATA_DIR/dev/dev.arin \
    $DATA_DIR/dev/dev.arin \
    $DATA_DIR/dev/dev.arin \
    > $DATA_DIR/dev/dev.arin+dev.arin+dev.arin+dev.arin

    cat $DATA_DIR/dev/dev.ar.MM \
    $DATA_DIR/dev/dev.ar.FM \
    $DATA_DIR/dev/dev.ar.MF \
    $DATA_DIR/dev/dev.ar.FF \
    > $DATA_DIR/dev/dev.ar.MM+dev.ar.FM+dev.ar.MF+dev.ar.FF

    python  $M2_SCORER/scripts/edit_creator.py \
    $DATA_DIR/dev/dev.arin+dev.arin+dev.arin+dev.arin \
    $DATA_DIR/dev/dev.ar.MM+dev.ar.FM+dev.ar.MF+dev.ar.FF \
    > $EDITS_DIR/dev.arin+dev.arin+dev.arin+dev.arin.to.dev.ar.MM+dev.ar.FM+dev.ar.MF+dev.ar.FF.edits

    rm $DATA_DIR/dev/dev.arin+dev.arin+dev.arin+dev.arin
    rm $DATA_DIR/dev/dev.ar.MM+dev.ar.FM+dev.ar.MF+dev.ar.FF

    printf "================================================="
    printf "\nCreating test set M2 edits:\n\n"

    # Test set annotations
    cat $DATA_DIR/test/test.arin \
    $DATA_DIR/test/test.arin \
    $DATA_DIR/test/test.arin \
    $DATA_DIR/test/test.arin \
    > $DATA_DIR/test/test.arin+test.arin+test.arin+test.arin

    cat $DATA_DIR/test/test.ar.MM \
    $DATA_DIR/test/test.ar.FM \
    $DATA_DIR/test/test.ar.MF \
    $DATA_DIR/test/test.ar.FF \
    > $DATA_DIR/test/test.ar.MM+test.ar.FM+test.ar.MF+test.ar.FF

    python  $M2_SCORER/scripts/edit_creator.py \
    $DATA_DIR/test/test.arin+test.arin+test.arin+test.arin \
    $DATA_DIR/test/test.ar.MM+test.ar.FM+test.ar.MF+test.ar.FF \
    > $EDITS_DIR/test.arin+test.arin+test.arin+test.arin.to.test.ar.MM+test.ar.FM+test.ar.MF+test.ar.FF.edits

    rm $DATA_DIR/test/test.arin+test.arin+test.arin+test.arin
    rm $DATA_DIR/test/test.ar.MM+test.ar.FM+test.ar.MF+test.ar.FF

else
    printf "APGC version not found: Please choose v1.0 or v2.0!\n"
fi



