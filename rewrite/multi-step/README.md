# Multi-step Gender Rewriting:

### Generating Gender Alternatives:
To generate gender alternatives for the target users preferences we model (i.e., MM, FM, MF, FF), you would need to run `scripts/run_rewriting.sh`. This script generates files for each target user preference. It also generates an error analysis report indicating the cases where the model failed to generate a correct output. It is very important to note that for this step to work, the predicted word-level gender labels must be provided. The word-level gender predictions can be obtained from [here](https://drive.google.com/drive/folders/1vEkuhP4zW4PqEPd3u5F8LD5AH6Xnb5_m?usp=sharing); where `dev_predictions.txt` has the word-level predictions on the dev set and `test_predictions.txt` has the word-level predicitons on the test set of APGCv2.0. The fine-tuned BERT model (CAMeLBERT MSA) we use for selection is [here](https://drive.google.com/drive/folders/1WnJXhLxexrwlCNrG8mxpY-5schKMrmp-?usp=sharing)<br/>

The `scripts/run_rewriting.sh` script has all the parameters needed to replicate the experiments we report in our paper. All of these parameters are self-explanatory and more details on them can be found in [main.py](https://github.com/balhafni/gender-rewriting/blob/master/rewrite/multi-step/main.py). Here's an example on how to run the **GID + CorpusR >> MorphR >> NeuralR + Selection** (i.e., the best performing system) on the dev set:

```bash
python main.py \
--data_dir /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/ \
--morph_db /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
--bert_model /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm \
--src_bert_tags_dir /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/models/dev_predictions.txt \
--inference_mode dev \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models \
--top_n_best 5 \
--beam_width 10 \
--use_gpu \
--output_dir logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/rewriting/CBR_MorphR_NeuralR \
--analyze_errors \
--error_analysis_dir logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/error_analysis/CBR_MorphR_NeuralR
```

We also use the same script to generate gender alternatives for the first-person only version of this task (i.e., M and F). Here's an example on how to run our best system on the test set of APGC v1.0:

```bash
python main.py \
--data_dir /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-1.0/new_token_data/ \
--morph_db /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
--bert_model /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm \
--src_bert_tags_dir /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/single_user/models_acc/test_predictions.txt \
--first_person_only \
--inference_mode test \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models/single_user \
--top_n_best 5 \
--beam_width 10 \
--use_gpu \
--output_dir logs/paper_results_with_mlm_ft_final/single_user/rewriting/CBR_MorphR_NeuralR_test \
--analyze_errors \
--error_analysis_dir logs/paper_results_with_mlm_ft_final/single_user/error_analysis/CBR_MorphR_NeuralR_test
```

The gender rewriting outputs and eval scores of the various systems we report on in our paper can be found in `logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/rewriting/` and their corresponding error analyses can be found in `logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/error_analysis/`.<br/>

We also did some experiments to demonstrate the effectiveness of using the fine-tuned CAMeLBERT MSA model instead of the generic one for our selection component. The gender rewriting outputs and eval scores of the experiments *without* doing any MLM fine-tuning can be found in `logs/paper_results_no_mlm_ft_final/rewriting`.

The outputs of the first-person only task can be found in `logs/paper_results_with_mlm_ft_final/single_user/rewriting/`.

### Augmentation Experiments:
Replecating the augmentation experiments is also straight forward and done using the `scripts/run_rewriting.sh` script. Here's an example on how to get the outputs of the **GID<sub>Aug</sub> + CorpusR >> MorphR >> NeuralR<sub>Aug</sub> + Selection** system on the dev set of APGC v2.0:

```bash
python main.py \
--data_dir /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/ \
--morph_db /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
--bert_model /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm \
--src_bert_tags_dir /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_3_5000_acc/dev_predictions.txt \
--inference_mode dev \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models/augmented_fix \
--top_n_best 5 \
--beam_width 10 \
--use_gpu \
--output_dir logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/augmentation/rewriting/CBR_MorphR_NeuralR_aug_id_aug \
--analyze_errors \
--error_analysis_dir logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/augmentation/error_analysis/CBR_MorphR_NeuralR_aug_id_aug
```

The outputs of the various systems we report on in our augmentation experiments can be found in `logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/augmentation/rewriting` and their error analyses in ` logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/augmentation/error_analysis/`.


### Post-Editing Machine Translation Output:
Getting the post-edited Google Translate output is also done using the `scripts/run_rewriting.sh` script. The predicted word-level labels of the Google Translate outputs are available [here](https://drive.google.com/drive/folders/1ZyOj1fb3UX527THm2_0LUGLoQyTpFYO2?usp=sharing).</br>
Here's how to run our best augmented system (**GID<sub>Aug</sub> + CorpusR >> MorphR >> NeuralR<sub>Aug</sub> + Selection**) on the Google Translate output of the test set of APGCv2.0:

```bash
python main.py \
--data_dir /scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/ \
--morph_db /scratch/ba63/calima_databases/calima-msa/calima-msa-s31_0.4.2.utf8.db.copy-mod \
--bert_model /scratch/ba63/mlm_lm/bert-base-arabic-camelbert-msa-mlm \
--src_bert_tags_dir /scratch/ba63/gender-rewriting/gender-id/CAMeLBERT_MSA/multi_user_with_clitics/controlled_settings/augmented_models_3_5000_acc/google_MT/test_predictions.txt \
--inference_mode test \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models/augmented_fix \
--top_n_best 5 \
--beam_width 10 \
--use_gpu \
--output_dir logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/MT/CBR_MorphR_NeuralR_aug_id_aug_test
```

The post-edited MT output for our best system can be found in `logs/paper_results_with_mlm_ft_final/multi_user_with_clitics/MT/`

## Evaluation:
Running the M<sup>2</sup> scorer and BLEU evaluations for gender-rewriting is done through the `script/run_eval.sh` script. You probably need to change the `EXPERIMENT` parameter to point the right outputs directory.</br>

For the post-editing MT experiments, we run BLEU evaluation across the four target corpora (i.e., MM, FM, MF, FF) of APGVv2.0. This is done through the `script/bleu_eval.sh` script.
