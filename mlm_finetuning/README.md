# Masked Language Modeling Fine-tuning:

For the in-context selection step of our multi-step pipeline, we use [CAMeLBERT MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa) to compute the pseudo-log likelihood scores of the generated candidate sentences.</br>

We found that fine-tuning CAMeLBERT MSA as a MLM on the training split of APGCv2.0 yields better results. To run the fine-tuning:

```bash


export TRAIN_DATA_FILE=/home/ba63/gender-rewriting/data/mlm/train.txt
export DEV_DATA_FILE=/home/ba63/gender-rewriting/data/mlm/dev.txt
export MODEL=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
export OUTPUT_DIR=/scratch/ba63/gender-rewriting/mlm_lm/bert-base-arabic-camelbert-msa-mlm-88

python run_mlm_no_trainer.py \
--model_name_or_path $MODEL \
--train_file $TRAIN_DATA_FILE \
--validation_file $DEV_DATA_FILE \
--num_train_epochs 3 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--line_by_line True \
--overwrite_cache True \
--seed 88 \
--output_dir $OUTPUT_DIR
```

The fine-tuned model CAMeLBERT MSA model is available in this [release](https://github.com/balhafni/gender-rewriting/releases/tag/gender-rewriting-models) and the data we used to fine-tuned the model is [here](https://github.com/balhafni/gender-rewriting/tree/master/data/mlm).
