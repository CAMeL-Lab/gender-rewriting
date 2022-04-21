# EXPERIMENT=CorpusR_MorphR_NeuralR_aug_GID_aug_test
EXPERIMENT=google_translate
DATA_DIR=/home/ba63/gender-rewriting/rewrite/multi-step/logs/multi_user/MT/$EXPERIMENT
GOLD_DATA_DIR=/home/ba63/gender-rewriting/data/m2_edits/v2.0/norm_data/MT
split=dev

if [ "$EXPERIMENT" = "google_translate" ]; then

    for f in $GOLD_DATA_DIR/$split.ar.MM.norm $GOLD_DATA_DIR/$split.ar.FM.norm  $GOLD_DATA_DIR/$split.ar.MF.norm  $GOLD_DATA_DIR/$split.ar.FF.norm 
    do
    printf "Evaluation Against $f:\n"
        cat $DATA_DIR/$split.google.ar.norm | sacrebleu $f --force
    done

else

    python /home/ba63/gender-rewriting/rewrite/multi-step/utils/normalize.py \
        --input_file $DATA_DIR/arin.to.MM.preds \
        --output_file $DATA_DIR/arin.to.MM.preds.norm

     python /home/ba63/gender-rewriting/rewrite/multi-step/utils/normalize.py \
        --input_file $DATA_DIR/arin.to.FM.preds \
        --output_file $DATA_DIR/arin.to.FM.preds.norm

     python /home/ba63/gender-rewriting/rewrite/multi-step/utils/normalize.py \
        --input_file $DATA_DIR/arin.to.MF.preds \
        --output_file $DATA_DIR/arin.to.MF.preds.norm

     python /home/ba63/gender-rewriting/rewrite/multi-step/utils/normalize.py \
        --input_file $DATA_DIR/arin.to.FF.preds \
        --output_file $DATA_DIR/arin.to.FF.preds.norm

    printf "Evaluating  $DATA_DIR/arin.to.MM.preds.norm against $GOLD_DATA_DIR/$split.ar.MM.norm:\n"
    cat $DATA_DIR/arin.to.MM.preds.norm | sacrebleu $GOLD_DATA_DIR/$split.ar.MM.norm --force

    printf "Evaluating  $DATA_DIR/arin.to.FM.preds.norm against $GOLD_DATA_DIR/$split.ar.FM.norm:\n"
    cat $DATA_DIR/arin.to.FM.preds.norm | sacrebleu $GOLD_DATA_DIR/$split.ar.FM.norm --force

    printf "Evaluating  $DATA_DIR/arin.to.MF.preds.norm against $GOLD_DATA_DIR/$split.ar.MF.norm:\n"
    cat $DATA_DIR/arin.to.MF.preds.norm | sacrebleu $GOLD_DATA_DIR/$split.ar.MF.norm --force

    printf "Evaluating  $DATA_DIR/arin.to.FF.preds.norm against $GOLD_DATA_DIR/$split.ar.FF.norm:\n"
    cat $DATA_DIR/arin.to.FF.preds.norm | sacrebleu $GOLD_DATA_DIR/$split.ar.FF.norm --force

fi
