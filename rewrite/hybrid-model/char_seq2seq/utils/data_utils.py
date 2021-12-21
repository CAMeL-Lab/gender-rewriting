import json
import csv
import os
import copy
import numpy as np
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.mle import MLEDisambiguator
import torch

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
            return self.create_examples(os.path.join(data_dir, 'joint_model/D-set-train.arin+D-set-train.arin'),
                                        os.path.join(data_dir, 'joint_model/D-set-train.ar.M+D-set-train.ar.F'))

        else:
            return self.create_examples(os.path.join(data_dir, 'nn_token_data/train.arin.tokens+train.arin.tokens+train.arin.tokens+train.arin.tokens'),
                                        os.path.join(data_dir, 'nn_token_data/train.ar.MM.tokens+train.ar.FM.tokens+train.ar.MF.tokens+train.ar.FF.tokens'))

    def get_dev_examples(self, data_dir, first_person_only=False):
        """Reads the dev examples of the dataset"""
        if first_person_only:
            return self.create_examples(os.path.join(data_dir, 'joint_model/D-set-dev.arin+D-set-dev.arin'),
                                        os.path.join(data_dir, 'joint_model/D-set-dev.ar.M+D-set-dev.ar.F'))

        else:
            return self.create_examples(os.path.join(data_dir, 'nn_token_data/dev.arin.tokens+dev.arin.tokens+dev.arin.tokens+dev.arin.tokens'),
                                        os.path.join(data_dir, 'nn_token_data/dev.ar.MM.tokens+dev.ar.FM.tokens+dev.ar.MF.tokens+dev.ar.FF.tokens'))

    def get_test_examples(self, data_dir, first_person_only=False):
        """Reads the test examples of the dataset"""
        if first_person_only:
            return self.create_examples(os.path.join(data_dir, 'joint_model/D-set-test.arin+D-set-test.arin'),
                                         os.path.join(data_dir, 'joint_model/D-set-test.ar.M+D-set-test.ar.F'))

        else:
            return self.create_examples(os.path.join(data_dir, 'nn_token_data/test.arin.tokens+test.arin.tokens+test.arin.tokens+test.arin.tokens'),
                                        os.path.join(data_dir, 'nn_token_data/test.ar.MM.tokens+test.ar.FM.tokens+test.ar.MF.tokens+test.ar.FF.tokens'))

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
