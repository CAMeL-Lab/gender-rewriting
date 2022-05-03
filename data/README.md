# Data:

## Gender Identification:
We used APGCv2.1 to train and evaluate the gender identification systems is in [gender-id/multi_user](gender-id/multi_user). The train/dev/test.txt files are identical to train.arin.tokens/dev.arin.tokens/test.arin.tokens that are available with the release of APGC v2.0. The only difference is that the train/dev/test.txt contain the extended word-level gender annotations (i.e., base word gender + clitic gender). We also corrected two sentences in the train split of APGC v2.0 (sentences with ids: B-8397.1 and C-1225.2). 

[gender-id/multi_user/augmented_data](gender-id/multi_user/augmented_data) contains the augmented training data which we used in our augmentation experiments.

The data we used to train and evaluate word-level gender identification for the first-person only version of the task is in [gender-id/single_user](gender-id/single_user).


## Gender Rewriting:
The data we used to train our multi-step gender rewriting systems is in [rewrite/apgc-v2.1](rewrite/apgc-v2.1).<br/>
The train.\*.tokens/dev.\*.tokens/test.\*.tokens are identical to train.\*.tokens/dev.\*.tokens/test.\*.tokens that are available with the current release of APGC v2.0. Again, the only difference is that the data we provide contains the extended word-level gender annotations (i.e., base word gender + clitic gender), in addition to the corrected sentences mentioned above. The extended word-level gender annotations were obtained by using the [utils/clitic_and_form_tagger.ipynb](utils/clitic_and_form_tagger.ipynb) script.

### Neural Rewriter Data:
The data we used to train our out-of-context word-level neural rewriter component is in [rewrite/apgc-v2.1/nn_token_data](rewrite/apgc-v2.1/nn_token_data). To create this data we did the following:
1) Duplicated [rewrite/apgc-v2.1/train.arin.tokens](rewrite/apgc-v2.1/train.arin.tokens) four times and removed all the tokens that are marked as B+B. This results in [rewrite/apgc-v2.1/nn_token_data/train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.no_B+B.clean.words](rewrite/apgc-v2.1/nn_token_data/train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.no_B+B.clean.words)</br></br>
2) Concatenated [rewrite/apgc-v2.1/train.ar.MM.tokens](rewrite/apgc-v2.1/train.ar.MM.tokens), [rewrite/apgc-v2.1/train.ar.FM.tokens](rewrite/apgc-v2.1/train.ar.FM.tokens), [rewrite/apgc-v2.1/train.ar.MF.tokens](rewrite/apgc-v2.1/train.ar.MF.tokens), and [rewrite/apgc-v2.1/train.ar.FF.tokens](rewrite/apgc-v2.1/train.ar.FF.tokens) and removed all tokens that are maked as B+B. This results in [rewrite/apgc-v2.1/nn_token_data/train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B.clean.words](rewrite/apgc-v2.1/nn_token_data/train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B.clean.words) and [rewrite/apgc-v2.1/nn_token_data/train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B.clean.gender](rewrite/apgc-v2.1/nn_token_data/train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B.clean.gender), where the .words file contains the words and .gender file contains the word-level target gender labels.</br></br>
3) We repeated steps 2) and 3) to create the same files for the dev and test splits.

The above steps are applied using the [rewrite/apgc-v2.1/nn_token_data/get_nn_data.sh](https://github.com/balhafni/gender-rewriting/blob/master/data/rewrite/apgc-v2.1/nn_token_data/get_nn_data.sh) script.

### Augmented Data:
The training data we used in our augmentation experiments can be found in [rewrite/apgc-v2.1/augmentation](rewrite/apgc-v2.1/augmentation). The augmented training data used to train the neural rewriter model can be found in [rewrite/apgc-v2.1/augmentation/nn_token_data](rewrite/apgc-v2.1/augmentation/nn_token_data).

### Google Translate Data:
The word-level data of the Google Translate outputs which we used in our post-editing experiments are in [rewrite/apgc-v2.1/google_MT](rewrite/apgc-v2.1/google_MT).

### Joint Baselines Data:
The data we used to train our sentence-level joint baseline rewriting models are in [rewrite/apgc-v2.1/joint](rewrite/apgc-v2.1/joint). This data was created as follows:
1) Duplicated the train.arin file that is part of APGC v2.1 four times to create [rewrite/apgc-v2.1/joint/train.arin+train.arin+train.arin+train.arin](rewrite/apgc-v2.1/joint/train.arin+train.arin+train.arin+train.arin).<br/><br/>
2) Concatenated train.ar.MM, train.ar.FM, train.ar.MF, and train.ar.FF that are part of APGC v2.1 to create [rewrite/apgc-v2.1/joint/train.ar.MM+train.ar.FM+train.ar.MF+train.ar.FF](rewrite/apgc-v2.1/joint/train.ar.MM+train.ar.FM+train.ar.MF+train.ar.FF).<br/><br/>
3) We repeated steps 2) and 3) to create the same files for the dev and test splits.

The .gender files contain the target genders we are modeling (i.e., MM, FM, MF, and FF). The .label files contain the sentence-level labels that are part of APGC v2.1.


The data we used to train our multi-step gender rewriting system on the first-person only version of the task can be found in [rewrite/apgc-v1.0](rewrite/apgc-v1.0).

## M<sup>2</sup> Scorer Edits:
The gold M<sup>2</sup> word-level annotations which we used to evaluate our systems are in [m2_edits/](m2_edits/). The files in `m2_edits/[v1.0|v2.1]/edits/` were created by suing the [m2_edits/create_m2_edits.sh](m2_edits/create_m2_edits.sh) script.

## MLM Fine-tuning:
We used [mlm/train.txt](mlm/train.txt) and [mlm/dev.txt](mlm/dev.txt) to fine-tune CAMeLBERT MSA on the MLM task. The train and dev data are identical to the ones train.arin (with the two corrected sentences mentioned above) and dev.arin that are available as part of APGCv2.1.
