# Out-of-Context Neural Rewriter:

The data we use to train and evaluate the Neural Rewriter (NeuralR) model can be found [here](https://github.com/balhafni/gender-rewriting/tree/master/data/rewrite/apgc-v2.0/nn_token_data). The augmented data we used to train the NeuralR model is [here](https://github.com/balhafni/gender-rewriting/tree/master/data/rewrite/apgc-v2.0/augmentation/nn_token_data).
To train the NeuralR model we describe in our paper, you would need to run [scripts/train_seq2seq.sh](scripts/train_seq2seq.sh). Here are the settings we used to train the model: <br/>

```bash
export DATA_DIR=/path/to/data/

python main.py \
 --data_dir $DATA_DIR \
 --add_side_constraints \
 --analyzer_db_path /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
 --vectorizer_path saved_models/vectorizer.json \
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
 --model_path saved_models/joint.pt
 ```


The pretrained NeuralR model can be found in [saved_models/](saved_models/) and the pretrained NeuralR model on the augmented data can be found n [saved_models/augmented_fix](saved_models/augmented_fix). The NeuralR model which was pretrained on the first-person-only [data](https://github.com/balhafni/gender-rewriting/tree/master/data/rewrite/apgc-v1.0/nn_token_data) can be found in [saved_models/single_user](saved_models/single_user).

## Infernece:
For inference, you would need to run [scripts/inference_seq2seq.sh](scripts/inference_seq2seq.sh). Although the NeuralR system was used to do inference as part of the multi-step model as a back-off mode, here are the inference settings that could be used to get the 5-best hypotheses:

```bash
export DATA_DIR=/path/to/data/

python main.py \
 --data_dir $DATA_DIR \
 --embed_dim 128 \
 --add_side_constraints \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --seed 21 \
 --model_path saved_models/joint.pt \
 --do_inference \
 --inference_mode dev \
 --beam_size 10 \
 --n_best 5 \
 --preds_dir logs/reinflection/dev.joint
 ```
 
