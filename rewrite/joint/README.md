# Joint Gender Rewriting Baseline:


The data we use to train and evaluate the joint baselines can be found [here](https://github.com/balhafni/gender-rewriting/tree/master/data/rewrite/apgc-v2.0/joint). 
To train the various joint baseline gender rewriting models we describe in our paper, you would need to run `scripts/train_seq2seq.py`. Here are the settings we used to train the three variants of the joint baselines we present in our paper: <br/>

### Joint+Morph:

```bash
export DATA_DIR=/path/to/data/

python main.py \
 --data_dir $DATA_DIR \
 --embed_trg_gender \
 --trg_gender_embed_dim 10 \
 --vectorizer_path saved_models/multi_user_side_constraints_newdb_clean_train/vectorizer.json \
 --use_morph_features \
 --analyzer_db_path /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
 --morph_features_path saved_models/multi_user_newdb_clean_train/morph_features_top_1_analyses.json \
 --cache_files \
 --num_train_epochs 50 \
 --embed_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --batch_size 32 \
 --use_cuda \
 --seed 21 \
 --do_train \
 --dropout 0.2 \
 --clip_grad 1.0 \
 --do_early_stopping \
 --model_path saved_models/multi_user_newdb_clean_train/joint+morph.pt
 
```

### Joint+Side Constraints:

```bash
export DATA_DIR=/path/to/data/

python main.py \
 --data_dir $DATA_DIR \
 --add_side_constraints \
 --vectorizer_path saved_models/multi_user_side_constraints_newdb_clean_train/vectorizer.json \
 --morph_features_path saved_models/multi_user_side_constraints_newdb_clean_train/morph_features_top_1_analyses.json \
 --cache_files \
 --num_train_epochs 50 \
 --embed_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --batch_size 32 \
 --use_cuda \
 --seed 21 \
 --do_train \
 --dropout 0.2 \
 --clip_grad 1.0 \
 --do_early_stopping \
 --model_path saved_models/multi_user_side_constraints_newdb_clean_train/joint.pt
 
```


### Joint+Side Constraints+Morph:

```bash
export DATA_DIR=/path/to/data/

python main.py \
 --data_dir $DATA_DIR \
 --add_side_constraints \
 --vectorizer_path saved_models/multi_user_side_constraints_newdb_clean_train/vectorizer.json \
 --use_morph_features \
 --analyzer_db_path /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
 --morph_features_path saved_models/multi_user_side_constraints_newdb_clean_train/morph_features_top_1_analyses.json \
 --cache_files \
 --num_train_epochs 50 \
 --embed_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --batch_size 32 \
 --use_cuda \
 --seed 21 \
 --do_train \
 --dropout 0.2 \
 --clip_grad 1.0 \
 --do_early_stopping \
 --model_path saved_models/multi_user_side_constraints_newdb_clean_train/joint+morph.pt
 
```

The joint baseline Joint+Morph pretraine model can be found in `saved_models/multi_user_newdb_clean_train`. The two joint baselines that uses side constraints (i.e., joint+side constraints and joint+side constraints+morph) can be found in  `saved_models/multi_user_side_constraints_newdb_clean_train`.

## Infernece:
For inference, you would need to run `scripts/inference_seq2seq.sh`. Here are the inference settings we used to get the outputs of our three baseline systems:

### Joint+Morph:
```bash
export DATA_DIR=/path/to/data/

python main.py \
 --data_dir $DATA_DIR \
 --embed_trg_gender \
 --trg_gender_embed_dim 10 \
 --use_morph_features \
 --analyzer_db_path /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
 --embed_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --seed 21 \
 --model_path saved_models/multi_user_newdb_clean_train/joint+morph.pt \
 --do_inference \
 --inference_mode dev \
 --preds_dir logs/reinflection/multi_user_newdb_clean_train/dev.joint+morph
 ```
 
 ### Joint+Side Constraints:
```bash
export DATA_DIR=/path/to/data/

python main.py \
 --data_dir $DATA_DIR \
 --add_side_constraints \
 --embed_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --seed 21 \
 --model_path saved_models/multi_user_side_constraints_newdb_clean_train/joint.pt \
 --do_inference \
 --inference_mode dev \
 --preds_dir logs/reinflection/multi_user_side_constraints_newdb_clean_train/dev.joint
 ```
 
 
### Joint+Side Constraints+Morph:
```bash
export DATA_DIR=/path/to/data/

python main.py \
 --data_dir $DATA_DIR \
 --add_side_constraints \
 --use_morph_features \
 --analyzer_db_path /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
 --embed_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --seed 21 \
 --model_path saved_models/multi_user_side_constraints_newdb_clean_train/joint+morph.pt \
 --do_inference \
 --inference_mode dev \
 --preds_dir logs/reinflection/multi_user_side_constraints_newdb_clean_train/dev.joint+morph
 ```
 
 To get the inference on the test set of the `Joint+Side Constraint+Morph` system, you just have to switch the `inference_mode` from dev to test.
 
 ## Evaluation:
After running the inference, all of the systems outputs will be in `logs/multi_user_side_constraints_newdb_clean_train` and `logs/multi_user_newdb_clean_train`. `logs/multi_user_newdb_clean_train` has the outputs of the `Joint+Morph` system and `logs/multi_user_side_constraints_newdb_clean_train` has the outputs of both `Joint+Side Constraints` and `Joint+Side Constraints+Morph`. Files ending with `inf` indicate that the inference was done using greedy decoding and files ending with `beam` indicate that the inference was done using beam search.</br>

To run the M<sup>2</sup> scorer and BLEU evaluation, you would need to run `sbatch/run_eval_norm_joint.sh`. Make sure to set the `DECODING` variable to `inf/beam` to switch between evaluating greedy and beam decoding outputs, respectively.  
