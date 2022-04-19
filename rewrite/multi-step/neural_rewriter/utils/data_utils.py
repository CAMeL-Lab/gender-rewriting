import json
import csv
import os
import copy
import numpy as np
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.mle import MLEDisambiguator
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample:
    """Simple object to encapsulate each data example"""
    def __init__(self, src_token, trg_token,
                 trg_gender):
        self.src_token = src_token
        self.trg_token = trg_token
        self.trg_gender = trg_gender

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

class RawDataset:
    """Encapsulates the raw examples in InputExample objects"""
    def __init__(self, data_dir, first_person_only=False):
        self.train_examples = self.get_train_examples(data_dir,
                                                      first_person_only=first_person_only)

        self.dev_examples = self.get_dev_examples(data_dir,
                                                  first_person_only=first_person_only)

        self.test_examples = self.get_test_examples(data_dir,
                                                    first_person_only=first_person_only)

    def create_examples(self, src_path, trg_path):

        src_tokens = self.get_token_examples(src_path + '.words')
        trg_tokens = self.get_token_examples(trg_path + '.words')
        trg_genders = self.get_trg_gender(trg_path + '.gender')

        examples = []

        for i in range(len(src_tokens)):
            src_token = src_tokens[i].strip()
            trg_token = trg_tokens[i].strip()
            trg_gender = trg_genders[i].strip()

            input_example = InputExample(src_token=src_token,
                                         trg_token=trg_token,
                                         trg_gender=trg_gender)

            examples.append(input_example)

        return examples

    def get_trg_gender(self, data_dir):
        with open(data_dir) as f:
            return f.readlines()

    def get_token_examples(self, data_dir):
        with open(data_dir, encoding='utf8') as f:
            return f.readlines()

    def get_train_examples(self, data_dir, first_person_only=False):
        """Reads the train examples of the dataset"""
        if first_person_only:
            return self.create_examples(os.path.join(data_dir, 'nn_token_data/D-set-train.arin.tokens+D-set-train.arin.tokens.no_B+B.clean'),
                                        os.path.join(data_dir, 'nn_token_data/D-set-train.ar.M.tokens+D-set-train.ar.F.tokens.no_B+B.clean'))

        else:
            #  return self.create_examples(os.path.join(data_dir, 'nn_token_data/train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens'),
            #                              os.path.join(data_dir, 'nn_token_data/train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens'))
            return self.create_examples(os.path.join(data_dir, 'nn_token_data/train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens.no_B+B.clean'),
                                       os.path.join(data_dir, 'nn_token_data/train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens.no_B+B.clean'))

    def get_dev_examples(self, data_dir, first_person_only=False):
        """Reads the dev examples of the dataset"""
        if first_person_only:
            return self.create_examples(os.path.join(data_dir, 'nn_token_data/D-set-dev.arin.tokens+D-set-dev.arin.tokens.no_B+B.clean'),
                                        os.path.join(data_dir, 'nn_token_data/D-set-dev.ar.M.tokens+D-set-dev.ar.F.tokens.no_B+B.clean'))

        else:
            return self.create_examples(os.path.join(data_dir, 'nn_token_data/dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens.no_B+B.clean'),
                                        os.path.join(data_dir, 'nn_token_data/dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens.no_B+B.clean'))

    def get_test_examples(self, data_dir, first_person_only=False):
        """Reads the test examples of the dataset"""
        if first_person_only:
            return self.create_examples(os.path.join(data_dir, 'nn_token_data/D-set-test.arin.tokens+D-set-test.arin.tokens.no_B+B.clean'),
                                         os.path.join(data_dir, 'nn_token_data/D-set-test.ar.M.tokens+D-set-test.ar.F.tokens.no_B+B.clean'))

        else:
            return self.create_examples(os.path.join(data_dir, 'nn_token_data/test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens.no_B+B.clean'),
                                        os.path.join(data_dir, 'nn_token_data/test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens.no_B+B.clean'))

class Vocabulary:
    """Base vocabulary class"""
    def __init__(self, token_to_idx=None):

        if token_to_idx is None:
            token_to_idx = dict()

        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        return self.token_to_idx[token]

    def lookup_index(self, index):
        return self.idx_to_token[index]

    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def __len__(self):
        return len(self.token_to_idx)

class SeqVocabulary(Vocabulary):
    """Sequence vocabulary class"""
    def __init__(self, token_to_idx=None, unk_token='<unk>',
                 pad_token='<pad>', sos_token='<s>',
                 eos_token='</s>'):

        super(SeqVocabulary, self).__init__(token_to_idx)

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.pad_idx = self.add_token(self.pad_token)
        self.unk_idx = self.add_token(self.unk_token)
        self.sos_idx = self.add_token(self.sos_token)
        self.eos_idx = self.add_token(self.eos_token)

    def to_serializable(self):
        contents = super(SeqVocabulary, self).to_serializable()
        contents.update({'unk_token': self.unk_token,
                         'pad_token': self.pad_token,
                         'sos_token': self.sos_token,
                         'eos_token': self.eos_token})

        return contents

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def lookup_token(self, token):
        return self.token_to_idx.get(token, self.unk_idx)

class MorphFeaturizer:
    """Morphological Featurizer Class"""
    def __init__(self, analyzer_db_path):
        self.db = MorphologyDB(analyzer_db_path)
        self.analyzer = Analyzer(self.db, cache_size=46000)
        # self.db = CalimaStarDB(analyzer_db_path)
        # self.analyzer = CalimaStarAnalyzer(self.db, cache_size=46000)
        # self.disambiguator = MLEDisambiguator(self.analyzer)

        self.w_to_features = {}

    def featurize_token(self, word):
        """
        Args:
            - token (str): a sentence in Arabic
        Returns:
            - a dictionary of word to vector mapping for each word in the sentence.
              Each vector will be a one-hot representing the following features:
              [lex+m lex+f spvar+m spvar+f]
        """

        if word not in self.w_to_features:
            analyses = self.analyzer.analyze(word)
            if analyses:
                for analysis in analyses:
                    self.w_to_features[word] = list()
                    # each analysis will have a vector
                    features = np.zeros(4)
                    # getting the source and gender features
                    src = analysis['source']
                    func_gen = analysis['gen']
                    #form_gen = analysis['form_gen']

                    # functional gender features
                    if src == 'lex' and func_gen == 'm':
                        features[0] = 1
                    elif src == 'lex' and func_gen == 'f':
                        features[1] = 1
                    elif src == 'spvar' and func_gen == 'm':
                        features[2] = 1
                    elif src == 'spvar' and func_gen == 'f':
                        features[3] = 1

                    # form gender features
                    #if src == 'lex' and form_gen == 'm':
                    #    features[0] = 1
                    #elif src == 'lex' and form_gen == 'f':
                    #    features[1] = 1
                    #elif src == 'spvar' and form_gen == 'm':
                    #    features[2] = 1
                    #elif src == 'spvar' and form_gen == 'f':
                    #    features[3] = 1

                    self.w_to_features[word].append(features)

                # squashing all the vectors into one
                self.w_to_features[word] = np.array(self.w_to_features[word])
                self.w_to_features[word] = self.w_to_features[word].sum(axis=0)
                # replacing all the elements > 0 with 1
                self.w_to_features[word][self.w_to_features[word] > 0] = 1
                # replacing all the 0 elements with 1e-6 
                self.w_to_features[word][self.w_to_features[word] == 0] = 1e-6
                self.w_to_features[word] = self.w_to_features[word].tolist()
            else:
                self.w_to_features[word] = np.full((4), 1e-6).tolist()

    def featurize(self, tokens):
        """Featurizes a list of tokens"""
        for token in tokens:
            self.featurize_token(token)

    def to_serializable(self):
        return {'morph_features': self.w_to_features}

    def from_serializable(self, contents):
        self.w_to_features = contents['morph_features']

    def save_morph_features(self, path):
        with open(path, mode='w', encoding='utf8') as f:
            return json.dump(self.to_serializable(), f, ensure_ascii=False)

    def load_morph_features(self, path):
        with open(path) as f:
            return self.from_serializable(json.load(f))

    def create_morph_embeddings(self, word_vocab):
        """Creating a morphological features embedding matrix"""
        morph_features = self.w_to_features

        # Note: morph_features will have all the words in word_vocab
        # except: <s>, <pad>, <unk>, </s>, ' ', and side constraints

        # Creating a 0 embedding matrix of shape: (len(word_vocab), 4)
        morph_embedding_matrix = torch.ones((len(word_vocab), 4)) * 1e-6
        for word in word_vocab.token_to_idx:
            if word in morph_features:
                index = word_vocab.lookup_token(word)
                morph_embedding_matrix[index] = torch.tensor(morph_features[word],
                                                             dtype=torch.float64)

        return morph_embedding_matrix

class Vectorizer:
    """Vectorizer Class"""
    def __init__(self, src_vocab_char, trg_vocab_char,
                 src_vocab_word, trg_gender_vocab,
                 add_side_constraints=False):
        """
        Args:
            - src_vocab_char (SeqVocabulary): source vocab on the char level
            - trg_vocab_char (SeqVocabulary): target vocab on the char level
            - src_vocab_word (SeqVocabulary): source vocab on the word level
            - trg_gender_vocab (Vocabulary): target gender vocab on the sentence level
        """

        self.src_vocab_char = src_vocab_char
        self.trg_vocab_char = trg_vocab_char
        self.src_vocab_word = src_vocab_word
        self.trg_gender_vocab = trg_gender_vocab
        self.add_side_constraints = add_side_constraints

    @classmethod
    def create_vectorizer(cls, data_examples,
                          add_side_constraints=False):
        """Class method which builds the vectorizer
        vocab

        Args:
            - data_examples: list of InputExample

        Returns:
            - Vectorizer object
        """

        src_vocab_char = SeqVocabulary()
        trg_vocab_char = SeqVocabulary()
        src_vocab_word = SeqVocabulary()
        trg_gender_vocab = Vocabulary()

        for ex in data_examples:
            src_token = ex.src_token
            trg_token = ex.trg_token
            trg_gender = ex.trg_gender

            src_vocab_word.add_token(src_token)

            src_vocab_char.add_many(list(src_token))

            trg_vocab_char.add_many(list(trg_token))

            # adding target gender to src and target vocab if needed
            if add_side_constraints:
                src_vocab_word.add_token(f'<{trg_gender}>')
                src_vocab_char.add_token(f'<{trg_gender}>')
                trg_vocab_char.add_token(f'<{trg_gender}>')
            
            # trg_gender_vocab is used for the non-side-constraints exps,
            # so we dont need <>
            trg_gender_vocab.add_token(trg_gender)

        logger.info(f"*** TRG Genders:  {trg_gender_vocab.token_to_idx} ***")
        return cls(src_vocab_char, trg_vocab_char,
                   src_vocab_word, trg_gender_vocab,
                   add_side_constraints=add_side_constraints)

    def get_src_indices(self, seq, trg_gender=None):
        """Converts the source sequence chars
        to indices

        Args:
          - seq (str): The source sequence

        Returns:
          - char_level_indices (list): <s> + List of chars to index mapping + </s>
        """

        char_level_indices = [self.src_vocab_char.sos_idx]
        word_level_indices = [self.src_vocab_word.sos_idx]

        if self.add_side_constraints:
            char_level_indices.append(self.src_vocab_char.lookup_token(f'<{trg_gender}>'))
            word_level_indices.append(self.src_vocab_word.lookup_token(f'<{trg_gender}>'))

        for char in seq:
            char_level_indices.append(self.src_vocab_char.lookup_token(char))
            word_level_indices.append(self.src_vocab_word.lookup_token(seq))

        word_level_indices.append(self.src_vocab_word.eos_idx)
        char_level_indices.append(self.src_vocab_char.eos_idx)

        assert len(word_level_indices) == len(char_level_indices)

        return char_level_indices, word_level_indices

    def get_trg_indices(self, seq):
        """Converts the target sequence chars
        to indices

        Args:
          - seq (str): The target sequence

        Returns:
          - trg_x_indices (list): <s> + List of chars to index mapping
          - trg_y_indices (list): List of chars to index mapping + </s>
        """
        indices = [self.trg_vocab_char.lookup_token(t) for t in seq]

        trg_x_indices = [self.trg_vocab_char.sos_idx] + indices
        trg_y_indices = indices + [self.trg_vocab_char.eos_idx]
        return trg_x_indices, trg_y_indices

    def vectorize(self, src, trg, trg_gender):
        """
        Args:
          - src (str): The source sequence
          - trg (str): The target sequence
          - trg_gender (str): The target sequence gender

        Returns:
          - vectorized_src_char (tensor): <s> + vectorized source seq on the char level + </s>
          - vectorized_src_word (tensor): <s> + vectorized source seq on the word level + </s>
          - vectorized_trg_x (tensor): <s> + vectorized target seq on the char level
          - vectorized_trg_y (tensor): vectorized target seq on the char level + </s>
          - vectorized_trg_gender (tensor): vectorized target gender
        """

        vectorized_src_char, vectorized_src_word = self.get_src_indices(src, trg_gender)
        vectorized_trg_x, vectorized_trg_y = self.get_trg_indices(trg)
        vectorized_trg_gender = self.trg_gender_vocab.lookup_token(trg_gender)

        return {'src_char': torch.tensor(vectorized_src_char, dtype=torch.long),
                'src_word': torch.tensor(vectorized_src_word, dtype=torch.long),
                'trg_x': torch.tensor(vectorized_trg_x, dtype=torch.long),
                'trg_y': torch.tensor(vectorized_trg_y, dtype=torch.long),
                'trg_gender': torch.tensor(vectorized_trg_gender, dtype=torch.long)
               }

    def to_serializable(self):
        return {'src_vocab_char': self.src_vocab_char.to_serializable(),
                'trg_vocab_char': self.trg_vocab_char.to_serializable(),
                'src_vocab_word': self.src_vocab_word.to_serializable(),
                'trg_gender_vocab': self.trg_gender_vocab.to_serializable()
               }

    @classmethod
    def from_serializable(cls, contents):
        src_vocab_char = SeqVocabulary.from_serializable(contents['src_vocab_char'])
        src_vocab_word = SeqVocabulary.from_serializable(contents['src_vocab_word'])
        trg_vocab_char = SeqVocabulary.from_serializable(contents['trg_vocab_char'])
        trg_gender_vocab = Vocabulary.from_serializable(contents['trg_gender_vocab'])

        return cls(src_vocab_char, trg_vocab_char,
                   src_vocab_word, trg_gender_vocab)

