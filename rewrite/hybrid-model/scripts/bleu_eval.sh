EXPERIMENT=CBR_filter+backoff+all+morph_newdb+mod_per_3rd_generator+neural_augmented_id_augmented
# EXPERIMENT=Google
DATA_DIR=logs/paper_results/multi_user_with_clitics/MT/reinflection/$EXPERIMENT
GOLD_DATA_DIR=/scratch/ba63/Arabic-Parallel-Gender-Corpus/m2_edits/v2.0/norm_data/MT/
split=dev

if [ "$EXPERIMENT" = "Google" ]; then

    for f in $GOLD_DATA_DIR/$split.ar.MM.norm $GOLD_DATA_DIR/$split.ar.FM.norm  $GOLD_DATA_DIR/$split.ar.MF.norm  $GOLD_DATA_DIR/$split.ar.FF.norm 
    do
    printf "Evaluation Against $f:\n"
        cat $DATA_DIR/$split.google.ar.norm | sacrebleu $f --force
    done

    printf "\nMulti-Reference Evaluation\n"

    sacrebleu $GOLD_DATA_DIR/$split.ar.MM.norm $GOLD_DATA_DIR/$split.ar.FM.norm  $GOLD_DATA_DIR/$split.ar.MF.norm  $GOLD_DATA_DIR/$split.ar.FF.norm -i $DATA_DIR/$split.google.ar.norm -m bleu --force

else

    # sed -i '$ d' $DATA_DIR/arin.to.MM.preds
    # sed -i '$ d' $DATA_DIR/arin.to.FM.preds
    # sed -i '$ d' $DATA_DIR/arin.to.MF.preds
    # sed -i '$ d' $DATA_DIR/arin.to.FF.preds

    # python /home/ba63/gender-rewriting/rewrite/hybrid-model/utils/normalize.py --input_file $DATA_DIR/arin.to.MM.preds --output_file $DATA_DIR/arin.to.MM.preds.norm
    # python /home/ba63/gender-rewriting/rewrite/hybrid-model/utils/normalize.py --input_file $DATA_DIR/arin.to.FM.preds --output_file $DATA_DIR/arin.to.FM.preds.norm
    # python /home/ba63/gender-rewriting/rewrite/hybrid-model/utils/normalize.py --input_file $DATA_DIR/arin.to.MF.preds --output_file $DATA_DIR/arin.to.MF.preds.norm
    # python /home/ba63/gender-rewriting/rewrite/hybrid-model/utils/normalize.py --input_file $DATA_DIR/arin.to.FF.preds --output_file $DATA_DIR/arin.to.FF.preds.norm

    printf "Evaluating  $DATA_DIR/arin.to.MM.preds.norm againts $GOLD_DATA_DIR/$split.ar.MM.norm:\n"
    cat $DATA_DIR/arin.to.MM.preds.norm | sacrebleu $GOLD_DATA_DIR/$split.ar.MM.norm --force

    printf "Evaluating  $DATA_DIR/arin.to.FM.preds.norm againts $GOLD_DATA_DIR/$split.ar.FM.norm:\n"
    cat $DATA_DIR/arin.to.FM.preds.norm | sacrebleu $GOLD_DATA_DIR/$split.ar.FM.norm --force

    printf "Evaluating  $DATA_DIR/arin.to.MF.preds.norm againts $GOLD_DATA_DIR/$split.ar.MF.norm:\n"
    cat $DATA_DIR/arin.to.MF.preds.norm | sacrebleu $GOLD_DATA_DIR/$split.ar.MF.norm --force

    printf "Evaluating  $DATA_DIR/arin.to.FF.preds.norm againts $GOLD_DATA_DIR/$split.ar.FF.norm:\n"
    cat $DATA_DIR/arin.to.FF.preds.norm | sacrebleu $GOLD_DATA_DIR/$split.ar.FF.norm --force

fi
    # printf "\nMulti-Reference Evaluation\n"

    # sacrebleu $GOLD_DATA_DIR/$split.ar.MM.norm $GOLD_DATA_DIR/$split.ar.FM.norm  $GOLD_DATA_DIR/$split.ar.MF.norm  $GOLD_DATA_DIR/$split.ar.FF.norm -i $DATA_DIR/$split.google.ar.norm -m bleu --force
