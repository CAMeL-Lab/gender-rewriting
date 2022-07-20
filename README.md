# User-Centric Gender Rewriting
This repo contains code and pretrained models to reproduce the results in our paper [User-Centric Gender Rewriting](https://arxiv.org/pdf/2205.02211.pdf).


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

## Citation:

If you find the code or data in this repo helpful, please cite [our paper](https://www.aclweb.org/anthology/XXXX):

```bibtex
@inproceedings{alhafni-etal-2022-user,
    title = "User-Centric Gender Rewriting",
    author = "Alhafni, Bashar  and
      Habash, Nizar  and
      Bouamor, Houda",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.46",
    pages = "618--631",
    abstract = "In this paper, we define the task of gender rewriting in contexts involving two users (I and/or You) {--} first and second grammatical persons with independent grammatical gender preferences. We focus on Arabic, a gender-marking morphologically rich language. We develop a multi-step system that combines the positive aspects of both rule-based and neural rewriting models. Our results successfully demonstrate the viability of this approach on a recently created corpus for Arabic gender rewriting, achieving 88.42 M2 F0.5 on a blind test set. Our proposed system improves over previous work on the first-person-only version of this task, by 3.05 absolute increase in M2 F0.5. We demonstrate a use case of our gender rewriting system by using it to post-edit the output of a commercial MT system to provide personalized outputs based on the users{'} grammatical gender preferences. We make our code, data, and pretrained models publicly available.",
}
