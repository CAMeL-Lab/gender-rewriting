# Data:

## Gender Identification:
The data we used to train and evaluate the gender identification systems is in `data/gender-id/multi_user`. The train/dev/test.txt files are identical to train.arin.tokens/dev.arin.tokens/test.arin.tokens that are available with the current release of APGC v2.0. The only difference is that the train/dev/test.txt contains the extended word-level gender annotations (i.e., base word gender + clitic gender).

`data/gender-id/multi_user/augmented_data` contains the augmented training data which we used in our augmentation experiments. `data/gender-id/multi_user/google_MT` has the word-level dev and test word-level data from the Google Translate outputs that are part of APGC v2.0 (i.e., dev.google.ar and test.google.ar).

The data we used to train and evaluate word-level gender identification for the first-person only version of the task is in `data/gender-id/singe_user`.


## Gender Rewriting:
The data we used to train our multi-step gender rewriting systems is in `data/gender_rewriting/apgc-v2.0`.<br/>
The train/dev/test.arin.tokens are identical to train.arin.tokens/dev.arin.tokens/test.arin.tokens that are available with the current release of APGC v2.0 (they are also same as the ones we used for to train and evaluation the gender id system).

### Neural Rewriter Data:
The data we used to train our out-of-context word-level neural rewriter component is in `gender_rewriting/apgc-v2.0/nn_token_data`. To create this data we did the following:
1) Duplicated `gender_rewriting/apgc-v2.0/train.arin.tokens` four times and remove all the tokens that are marked as B+B. This results in `gender_rewriting/apgc-v2.0/nn_token_data/train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.words`</br></br>
2) Concatenated `gender_rewriting/apgc-v2.0/train.ar.MM.tokens`, `gender_rewriting/apgc-v2.0/train.ar.FM.tokens`, `gender_rewriting/apgc-v2.0/train.ar.MF.tokens`, and `gender_rewriting/apgc-v2.0/train.ar.FF.tokens` and remove all tokens that are maked as B+B. This results in `gender_rewriting/apgc-v2.0/nn_token_data/train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.words` and `gender_rewriting/apgc-v2.0/nn_token_data/train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.gender`, where the .words file contains the words and .gender file contains the labels.</br></br>
3) We repeated steps 2) and 3) to create the same files for the dev and test splits.


### Augmented Data:
The training data we used in our augmentation experiments can be found in `gender_rewriting/apgc-v2.0/augmentation`. The augmented training data used to train the neural rewriter model can be found in `gender_rewriting/apgc-v2.0/augmentation/nn_token_data`.

### Google Translate Data:
The word-level data of the Google Translate outputs which we used in our post-editing experiments are in `gender_rewriting/apgc-v2.0/google_MT`.

### Joint Baselines Data:
The data we used to train our sentence-level joint baseline rewriting models are in `gender_rewriting/apgc-v2.0/joint`. This data was created as follows:
1) Duplicated the train.arin file that is part of APGC v2.0 four times to create `gender_rewriting/apgc-v2.0/joint/train.arin+train.arin+train.arin+train.arin`.<br/><br/>
2) Concatenated train.ar.MM, train.ar.FM, train.ar.MF, and train.ar.FF that are part of APGC v2.0 to create `gender_rewriting/apgc-v2.0/joint/train.ar.MM+train.ar.FM+train.ar.MF+train.ar.FF`.<br/><br/>
3) We repeated steps 2) and 3) to create the same files for the dev and test splits.

The .gender files contain the target genders we are modeling (i.e., MM, FM, MF, and FF). The .label files contain the sentence-level labels that are part of APGC v2.0.


The data we used to train our multi-step gender rewriting system on the first-person only version of the task can be found in `gender_rewriting/apgc-v1.0`.

## M<sup>2</sup> Scorer Edits:
The gold M<sup>2</sup> word-level annotations which we used to evaluate our systems are in `m2_edits/`. The files in `m2_edits/[v1.0|v2.0]/edits/` were created by suing the `m2_edits/create_m2_edits.sh` script.
