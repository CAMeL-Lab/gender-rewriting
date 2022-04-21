#!/bin/bash
#SBATCH -p nvidia -q nlp 
# Set number of nodes to run
#SBATCH --nodes=1
# Set number of tasks to run
#SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=10 
# memory
#SBATCH --mem=10GB
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

export train=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/train.arin.tokens
export train_mm=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/train.ar.MM.tokens
export train_fm=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/train.ar.FM.tokens
export train_mf=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/train.ar.MF.tokens
export train_ff=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/train.ar.FF.tokens

export dev=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/dev.arin.tokens
export dev_mm=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/dev.ar.MM.tokens
export dev_fm=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/dev.ar.FM.tokens
export dev_mf=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/dev.ar.MF.tokens
export dev_ff=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/dev.ar.FF.tokens

export test=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/test.arin.tokens
export test_mm=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/test.ar.MM.tokens
export test_fm=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/test.ar.FM.tokens
export test_mf=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/test.ar.MF.tokens
export test_ff=/scratch/ba63/Arabic-Parallel-Gender-Corpus/Arabic-parallel-gender-corpus-v-2.0/data/new_token_data/test.ar.FF.tokens

# concat
cat $train $train $train $train > train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens
cat $train_mm $train_fm $train_mf $train_ff > train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens

cat $dev  $dev  $dev  $dev  > dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens
cat $dev_mm $dev_fm $dev_mf $dev_ff > dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens

cat $test $test $test $test > test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens
cat $test_mm $test_fm $test_mf $test_ff > test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens


# remove B+B
cat train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens | grep -v "B+B" > train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.no_B+B
cat train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens | grep -v "B+B" > train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B

cat dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens | grep -v "B+B" > dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens.no_B+B
cat dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens | grep -v "B+B" > dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens.no_B+B

cat test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens | grep -v "B+B" > test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens.no_B+B
cat test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens | grep -v "B+B" > test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens.no_B+B

# removing spaces
python delete_spaces.py

cat train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.no_B+B.clean | cut -d' ' -f1 > train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.no_B+B.clean.words
cat train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.no_B+B.clean | cut -d' ' -f2 > train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.no_B+B.clean.gender

cat train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B.clean | cut -d' ' -f1 > train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B.clean.words
cat train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B.clean | cut -d' ' -f2 > train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B.clean.gender

cat dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens.no_B+B.clean | cut -d' ' -f1 > dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens.no_B+B.clean.words
cat dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens.no_B+B.clean | cut -d' ' -f2 > dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens.no_B+B.clean.gender

cat dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens.no_B+B.clean | cut -d' ' -f1 > dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens.no_B+B.clean.words
cat dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens.no_B+B.clean | cut -d' ' -f2 > dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens.no_B+B.clean.gender

cat test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens.no_B+B.clean | cut -d' ' -f1 > test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens.no_B+B.clean.words
cat test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens.no_B+B.clean | cut -d' ' -f2 > test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens.no_B+B.clean.gender

cat test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens.no_B+B.clean | cut -d' ' -f1 > test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens.no_B+B.clean.words
cat test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens.no_B+B.clean | cut -d' ' -f2 > test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens.no_B+B.clean.gender

