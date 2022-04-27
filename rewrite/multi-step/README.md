# Multi-step Gender Rewriting:

### Generating Gender Alternatives:
To generate gender alternatives for the target users preferences we model (i.e., MM, FM, MF, FF), you would need to run [scripts/run_rewriting.sh](scripts/run_rewriting.sh). This script generates files for each target user preference. It also generates an error analysis report indicating the cases where the model failed to generate a correct output. It is very important to note that for this step to work, the word-level gender identification and the ranking and selection models must be provided. The three word-level gender identification models (i.e., multi-user, multi-user augmented, and single user) we use throughout our experiments and the fine-tuned CAMeLBERT MSA BERT model we use for selection are available as part of this [release](https://github.com/balhafni/gender-rewriting/releases/tag/gender-rewriting-models).<br/>

The [scripts/run_rewriting.sh](scripts/run_rewriting.sh) script has all the parameters needed to replicate the experiments we report in our paper. All of these parameters are self-explanatory and more details on them can be found in [main.py](https://github.com/balhafni/gender-rewriting/blob/master/rewrite/multi-step/main.py). Here's an example on how to run the **GID + CorpusR >> MorphR >> NeuralR + Selection** (i.e., the best performing system) on the dev set:

```bash
python main.py \
--data_dir /home/ba63/gender-rewriting/data/rewrite/apgc-v2.0/ \
--morph_db /scratch/ba63/gender-rewriting/models/calima-msa-s31_0.4.2.db \
--bert_model /scratch/ba63/gender-rewriting/models/bert-base-camel-bert-msa-apgcv2.0 \
--gender_id_model /scratch/ba63/gender-rewriting/models/gender-id-apgcv2.0 \
--inference_mode dev \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models/multi_user \
--top_n_best 3 \
--beam_width 10 \
--use_gpu \
--output_dir /home/ba63/gender-rewriting/rewrite/multi-step/logs/multi_user/rewriting/CorpusR_MorphR_NeuralR
--analyze_errors \
--error_analysis_dir /home/ba63/gender-rewriting/rewrite/multi-step/logs/multi_user/error_analysis/CorpusR_MorphR_NeuralR
```

We also use the same script to generate gender alternatives for the first-person only version of this task (i.e., M and F). Here's an example on how to run our best system on the test set of APGC v1.0:

```bash
python main.py \
--data_dir /home/ba63/gender-rewriting-camera-ready/data/rewrite/apgc-v1.0/ \
--morph_db /scratch/ba63/gender-rewriting/models/calima-msa-s31_0.4.2.db \
--bert_model /scratch/ba63/gender-rewriting/models/bert-base-camel-bert-msa-apgcv2.0 \
--gender_id_model /scratch/ba63/gender-rewriting/models/gender-id-apgcv1.0 \
--first_person_only \
--inference_mode test \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models/single_user \
--top_n_best 3 \
--beam_width 10 \
--use_gpu \
--output_dir /home/ba63/gender-rewriting/rewrite/multi-step/logs/single_user/CorpusR_MorphR_NeuralR_test \
--analyze_errors \
--error_analysis_dir /home/ba63/gender-rewriting/rewrite/multi-step/logs/single_user/CorpusR_MorphR_NeuralR_test
```

The gender rewriting outputs and eval scores of the various systems we report on in our paper can be found in [logs/multi_user/rewriting/](logs/multi_user/rewriting/) and their corresponding error analyses can be found in [logs/multi_user/error_analysis/](logs/multi_user/error_analysis/).<br/>

We also did some experiments to demonstrate the effectiveness of using the fine-tuned CAMeLBERT MSA model instead of the generic one for our selection component.

The outputs of the first-person only task can be found in [logs/single_user/rewriting/](logs/single_user/rewriting/).

### Augmentation Experiments:
Replecating the augmentation experiments is also straight forward and done using the [scripts/run_rewriting.sh](scripts/run_rewriting.sh) script. Here's an example on how to get the outputs of the **GID<sub>Aug</sub> + CorpusR >> MorphR >> NeuralR<sub>Aug</sub> + Selection** system on the dev set of APGC v2.0:

```bash
python main.py \
--data_dir /home/ba63/gender-rewriting/data/rewrite/apgc-v2.0/ \
--morph_db /scratch/ba63/gender-rewriting/models/calima-msa-s31_0.4.2.db \
--bert_model /scratch/ba63/gender-rewriting/models/bert-base-camel-bert-msa-apgcv2.0 \
--gender_id_model /scratch/ba63/gender-rewriting/models/gender-id-apgcv2.0-aug \
--inference_mode dev \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models/multi_user_augmented \
--top_n_best 3 \
--beam_width 10 \
--use_gpu \
--output_dir /home/ba63/gender-rewriting/rewrite/multi-step/logs/multi_user/augmentation/rewriting/CorpusR_MorphR_NeuralR_aug_GID_aug
--analyze_errors \
--error_analysis_dir /home/ba63/gender-rewriting/rewrite/multi-step/logs/multi_user/augmentation/error_analysis/CorpusR_MorphR_NeuralR_aug_GID_aug
```

The outputs of the various systems we report on in our augmentation experiments can be found in [logs/multi_user/augmentation/rewriting](logs/multi_user/augmentation/rewriting) and their error analyses in [logs/multi_user/augmentation/error_analysis/](logs/multi_user/augmentation/error_analysis).


### Post-Editing Machine Translation Output:
Getting the post-edited Google Translate output is also done using the [scripts/run_rewriting.sh](scripts/run_rewriting.sh) script.
Here's how to run our best augmented system (**GID<sub>Aug</sub> + CorpusR >> MorphR >> NeuralR<sub>Aug</sub> + Selection**) on the Google Translate output of the test set of APGCv2.0:
```bash
python main.py \
--data_dir /home/ba63/gender-rewriting/data/rewrite/apgc-v2.0/ \
--morph_db /scratch/ba63/gender-rewriting/models/calima-msa-s31_0.4.2.db \
--bert_model /scratch/ba63/gender-rewriting/models/bert-base-camel-bert-msa-apgcv2.0 \
--gender_id_model /scratch/ba63/gender-rewriting/models/gender-id-apgcv2.0-aug \
--inference_mode test \
--post_edit_MT \
--use_cbr \
--cbr_ngram 2 \
--cbr_backoff \
--reduce_cbr_noise \
--use_morph \
--use_seq2seq \
--seq2seq_model_path neural_rewriter/saved_models/multi_user_augmented \
--top_n_best 3 \
--beam_width 10 \
--use_gpu \
--output_dir /home/ba63/gender-rewriting/rewrite/multi-step/logs/multi_user/MT/CorpusR_MorphR_NeuralR_aug_GID_aug_test
```

The post-edited MT output for our best system can be found in [logs/multi_user/MT/](logs/multi_user/MT/)

## Evaluation:
Running the M<sup>2</sup> scorer and BLEU evaluations for gender-rewriting is done through the [scripts/run_eval.sh](scripts/run_eval.sh) script. You need to change the `EXPERIMENT` parameter to point the right outputs directory based on the system you want to evaluate.</br>

For the post-editing MT experiments, we run BLEU evaluation across the four target corpora (i.e., MM, FM, MF, FF) of APGCv2.0. This is done through the [scripts/bleu_eval.sh](scripts/bleu_eval.sh) script.
