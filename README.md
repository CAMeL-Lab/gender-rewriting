# User-Centric Gender Rewriting
This repo contains code to reproduce the results in our paper [User-Centric Gender Rewriting](https://www.aclweb.org/anthology/XXXX).


## Requirements:
The code was written for python>=3.7, pytorch 1.5.1, and transformers 4.11.3. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):
```bash
git clone https://github.com/balhafni/gender-rewriting.git
cd gender-rewriting

conda create -n gender_rewriting python=3.7
conda activate gender_rewriting

pip install -r requirements.txt
```

## Experiments and Reproducibility:
This repo is organized as follows:</br>
1. [data](data/): includes all the data we used through out our paper to train and test various systems. These systems include the joint gender rewriting baselines, the multi-step gender rewriting models, and the gender identification components. It also includes the augmentation data we created.
2. [gender-id](gender-id/): includes the scripts needed to fine-tune [CAMeLBERT MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa) for word-level gender identification.
3. [m2scorer](m2scorer/): includes the m2scorer, which we use to evaluate our gender rewriting systems.
4. [mlm_finetuning](mlm_finetuning/): includes the scripts needed to fine-tune [CAMeLBERT MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa) on the MLM task.
5. [rewrite](rewrite/):</br>
   1. [joint](rewrite/joint/): includes the scripts needed to train and evaluate our sentence-level joint gender rewriting baselines.
   2. [multi-step](rewrite/multi-step/): includes the scripts needed to train and evaluate our word-level multi-step gender rewriting systems.

The fine-tuned gender identification models can be found [here](https://drive.google.com/drive/folders/1IxmvY5xrnAq5QXBhKOEK908Z7H5R7uYp?usp=sharing) and the fine-tuned CAMeLBERT MSA model can be found [here](https://drive.google.com/drive/folders/1WnJXhLxexrwlCNrG8mxpY-5schKMrmp-?usp=sharing).

## License:
This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info.
