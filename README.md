# User-Centric Gender Rewriting
This repo contains code and pretrained models to reproduce the results in our paper [User-Centric Gender Rewriting](https://www.aclweb.org/anthology/XXXX).


## Requirements:
The code was written for python>=3.7, pytorch 1.5.1, and transformers 4.11.3. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):
```bash
git clone https://github.com/CAMeL-Lab/gender-rewriting.git
cd gender-rewriting

conda create -n gender_rewriting python=3.7
conda activate gender_rewriting

pip install -r requirements.txt
```

## Experiments and Reproducibility:
This repo is organized as follows:</br>
1. [data](data/): includes all the data we used through out our paper to train and test various systems. This includes the joint gender rewriting baselines, the multi-step gender rewriting models, the gender identification component, and the in-context ranking and selection system. It also includes the augmentation data we created.
2. [gender-id](gender-id/): includes the scripts needed to fine-tune [CAMeLBERT MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa) for word-level gender identification.
3. [mlm_finetuning](mlm_finetuning/): includes the scripts needed to fine-tune [CAMeLBERT MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa) using the MLM objective.
4. [rewrite](rewrite/):</br>
   1. [joint](rewrite/joint/): includes the scripts needed to train and evaluate our sentence-level joint gender rewriting baselines.
   2. [multi-step](rewrite/multi-step/): includes the scripts needed to train and evaluate our word-level multi-step gender rewriting systems.
5. [m2scorer](m2scorer/): includes the m2scorer, which we use to evaluate our gender rewriting systems.

The gender identification systems and the fine-tuned CAMeLBERT MSA model we use throughout the paper are inlcuded in this [release](https://github.com/balhafni/gender-rewriting/releases/tag/gender-rewriting-models).

## License:
This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info.
