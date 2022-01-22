# Word-Level Gender Identification:

For the word-level gender identification component, we fine-tune [CAMeLBERT MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa).

Note: All the fine-tuning experiments were done using Hugging Face's `transformers==4.11.3`

## Fine-tuning:


We fine-tune CAMeLBERT on the training split APGC v2.0 for multi-user gender idenfitication. The data used to fine-tune CAMeLBERT for multi-user word-level gender identification is [here](https://github.com/balhafni/gender-rewriting/tree/master/data/gender-id/multi_user).<br/>

We also fine-tuned CAMeLBERT on the augmented data of APGC v2.0 multi-user gender idenfitication as reported in our paper. The augmented training data we used is [here](https://github.com/balhafni/gender-rewriting/tree/master/data/gender-id/multi_user/augmented_data). 

To compare with previous work on the single-user rewriting task, we also fine-tune CAMeLBERT MSA on the training split of APGCv1.0. The data we used for the single-user word-level gender identification is [here](https://github.com/balhafni/gender-rewriting/tree/master/data/gender-id/single_user).

At the end of the fine-tuning, we pick the best checkpoint based on the overall performance on the gender-rewriting task on the dev set (either APGCv1.0 or APGCv2.0, depending if we're doing single-user or multi-user gender rewriting).<br/>

All of the three fine-tuned models CAMeLBERT models can be found [here]().<br/>

To run the fine-tuning:

```bash
export DATA_DIR=/path/to/data
export MAX_LENGTH=128
export BERT_MODEL=path/to/pretrained_model/ \ # Or huggingface model id 
export OUTPUT_DIR=/path/to/output_dir
export BATCH_SIZE=32
export NUM_EPOCHS=10 # or 3 for mutli-user gender id augmented 
export SAVE_STEPS=500 # or 5000 for mutli-user gender id augmented 
export EVAL_STEPS=500 # or 5000 for mutli-user gender id augmented  
export SEED=12345

python gender_identifcation.py \
--data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--eval_steps $EVAL_STEPS \
--evaluation_strategy steps \
--seed $SEED \
--do_train \
--do_eval \
--load_best_model_at_end \
--metric_for_best_model f1_macro \ # or acc for multi-user augmented and single-user gender id
--overwrite_output_dir \
--overwrite_cache \
```

## Inference:
To run inference:
```bash
export DATA_DIR=/path/to/data/
export MAX_LENGTH=128
export OUTPUT_DIR=/path/to/output_dir
export BATCH_SIZE=32
export SEED=12345

python gender_identifcation.py \
--data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $OUTPUT_DIR \
--output_dir $OUTPUT_DIR/google_MT \
--max_seq_length  $MAX_LENGTH \
--per_device_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--overwrite_cache \
--do_pred \
--pred_mode test # or dev to get the dev predictions
```

