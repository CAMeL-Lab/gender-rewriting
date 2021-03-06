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
    def __init__(self, src, trg,
                 src_label, trg_label,
                 trg_gender):

        self.src = src
        self.trg = trg
        self.src_label = src_label
        self.trg_label = trg_label
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
    def __init__(self, data_dir, first_person_only=False, normalized=False):
        self.train_examples = self.get_train_examples(data_dir,
                                                      first_person_only=first_person_only,
                                                      normalized=normalized)

        self.dev_examples = self.get_dev_examples(data_dir,
                                                  first_person_only=first_person_only,
                                                  normalized=normalized)

        self.test_examples = self.get_test_examples(data_dir,
                                                    first_person_only=first_person_only,
                                                    normalized=normalized)

    def create_examples(self, src_path, trg_path):

        src_txt = self.get_txt_examples(src_path)
        src_labels = self.get_labels(src_path + '.label')
        trg_txt = self.get_txt_examples(trg_path)
        trg_labels = self.get_labels(trg_path + '.label')
        trg_genders = self.get_trg_gender(trg_path + '.gender')

        examples = []

        for i in range(len(src_txt)):
            src = src_txt[i].strip()
            trg = trg_txt[i].strip()
            src_label = src_labels[i].strip()
            trg_label = trg_labels[i].strip()
            trg_gender = trg_genders[i].strip()

            input_example = InputExample(src=src,
                                         trg=trg,
                                         src_label=src_label,
                                         trg_label=trg_label,
                                         trg_gender=trg_gender)

            examples.append(input_example)

        return examples

    def get_labels(self, data_dir):
        with open(data_dir) as f:
            return f.readlines()

    def get_trg_gender(self, data_dir):
        with open(data_dir) as f:
            return f.readlines()

    def get_txt_examples(self, data_dir):
        with open(data_dir, encoding='utf8') as f:
            return f.readlines()

    def get_train_examples(self, data_dir, first_person_only=False,
                           normalized=False):
        """Reads the train examples of the dataset"""
        #joint_model/S-set.M.uniq+S-set.M.uniq+S-set.F.uniq+S-set.F.uniq+D-set-train.arin+D-set-train.arin
        #joint_model/S-set.M.uniq+S-set.F.uniq+S-set.M.uniq+S-set.F.uniq+D-set-train.ar.M+D-set-train.ar.F
        #joint_model/D-set-train.arin+D-set-train.arin
        #joint_model/D-set-train.ar.M+D-set-train.ar.F
        if first_person_only:
            if normalized:
                return self.create_examples(os.path.join(data_dir, 'D-set-train.arin+D-set-train.arin.normalized'),
                                            os.path.join(data_dir, 'D-set-train.ar.M+D-set-train.ar.F.normalized'))
            else:
                return self.create_examples(os.path.join(data_dir, 'D-set-train.arin+D-set-train.arin'),
                                            os.path.join(data_dir, 'D-set-train.ar.M+D-set-train.ar.F'))

        else:
            if normalized:
                return self.create_examples(os.path.join(data_dir, ''),
                                            os.path.join(data_dir, ''))
            else:
                return self.create_examples(os.path.join(data_dir, 'train.arin+train.arin+train.arin+train.arin'),
                                            os.path.join(data_dir, 'train.ar.MM+train.ar.FM+train.ar.MF+train.ar.FF'))

    def get_dev_examples(self, data_dir, first_person_only=False,
                         normalized=False):
        """Reads the dev examples of the dataset"""
        if first_person_only:
            if normalized:
                return self.create_examples(os.path.join(data_dir, 'D-set-dev.arin+D-set-dev.arin.normalized'),
                                            os.path.join(data_dir, 'D-set-dev.ar.M+D-set-dev.ar.F.normalized'))
            else:
                return self.create_examples(os.path.join(data_dir, 'D-set-dev.arin+D-set-dev.arin'),
                                            os.path.join(data_dir, 'D-set-dev.ar.M+D-set-dev.ar.F'))

        else:
            if normalized:
                return self.create_examples(os.path.join(data_dir, ''),
                                            os.path.join(data_dir, ''))
            else:
                return self.create_examples(os.path.join(data_dir, 'dev.arin+dev.arin+dev.arin+dev.arin'),
                                            os.path.join(data_dir, 'dev.ar.MM+dev.ar.FM+dev.ar.MF+dev.ar.FF'))

    def get_test_examples(self, data_dir, first_person_only=False,
                          normalized=False):
        """Reads the test examples of the dataset"""
        if first_person_only:
            if normalized:
                return self.create_examples(os.path.join(data_dir, 'D-set-test.arin+D-set-test.arin.normalized'),
                                            os.path.join(data_dir, 'D-set-test.ar.M+D-set-test.ar.F.normalized'))
            else:
                return self.create_examples(os.path.join(data_dir, 'D-set-test.arin+D-set-test.arin'),
                                            os.path.join(data_dir, 'D-set-test.ar.M+D-set-test.ar.F'))

        else:
            if normalized:
                return self.create_examples(os.path.join(data_dir, ''),
                                            os.path.join(data_dir, ''))
            else:
                return self.create_examples(os.path.join(data_dir, 'test.arin+test.arin+test.arin+test.arin'),
                                            os.path.join(data_dir, 'test.ar.MM+test.ar.FM+test.ar.MF+test.ar.FF'))


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
                 eos_token='</s>',
                 first_person_only=False,
                 add_side_constraints=False,
                 sc_mm='<MM>', sc_fm='<FM>',
                 sc_mf='<MF>', sc_ff='<FF>',
                 sc_m='<M>', sc_f='<F>'):

        super(SeqVocabulary, self).__init__(token_to_idx)

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        if add_side_constraints:
            if first_person_only:
                self.sc_m = sc_m
                self.sc_f = sc_f
            else:
                self.sc_mm = sc_mm
                self.sc_fm = sc_fm
                self.sc_mf = sc_mf
                self.sc_ff = sc_ff

        self.pad_idx = self.add_token(self.pad_token)
        self.unk_idx = self.add_token(self.unk_token)
        self.sos_idx = self.add_token(self.sos_token)
        self.eos_idx = self.add_token(self.eos_token)

        if add_side_constraints:
            if first_person_only:
                self.sc_m_idx = self.add_token(self.sc_m)
                self.sc_f_idx = self.add_token(self.sc_f)
            else:
                self.sc_mm_idx = self.add_token(self.sc_mm)
                self.sc_fm_idx = self.add_token(self.sc_fm)
                self.sc_mf_idx = self.add_token(self.sc_mf)
                self.sc_ff_idx = self.add_token(self.sc_ff)

        self.add_side_constraints = add_side_constraints
        self.first_person_only = first_person_only

    def to_serializable(self):
        contents = super(SeqVocabulary, self).to_serializable()
        contents.update({'unk_token': self.unk_token,
                         'pad_token': self.pad_token,
                         'sos_token': self.sos_token,
                         'eos_token': self.eos_token})

        if self.add_side_constraints:
            if self.first_person_only:
                contents.update({'sc_m': self.sc_m,
                                 'sc_f': self.sc_f})
            else:
                contents.update({'sc_mm': self.sc_mm,
                                 'sc_fm': self.sc_fm,
                                 'sc_mf': self.sc_mf,
                                 'sc_ff': self.sc_ff})

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
        self.disambiguator = MLEDisambiguator(self.analyzer, top=1)
        # self.db = CalimaStarDB(analyzer_db_path)
        # self.analyzer = CalimaStarAnalyzer(self.db, cache_size=46000)
        # self.disambiguator = MLEDisambiguator(self.analyzer)

        self.w_to_features = {}

    def featurize(self, sentence):
        """
        Args:
            - sentence (str): a sentence in Arabic
        Returns:
            - a dictionary of word to vector mapping for each word in the sentence.
              Each vector will be a one-hot representing the following features:
              [lex+m lex+f spvar+m spvar+f]
        """
        # using the MLEDisambiguator to get the analyses
        disambiguations = self.disambiguator.disambiguate(sentence.split(' '))
        # disambiguations is a list of DisambiguatedWord objects
        # each DisambiguatedWord object is a tuple of: (word, scored_analyses)
        # scored_analyses is a list of ScoredAnalysis objects
        # each ScoredAnalysis object is a tuple of: (score, analysis)

        for disambig in disambiguations:
            word, scored_analyses = disambig
            if word not in self.w_to_features:
                self.w_to_features[word] = list()
                if scored_analyses:
                    for scored_analysis in scored_analyses:
                        # each analysis will have a vector
                        score, analysis = scored_analysis
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

    def featurize_sentences(self, sentences):
        """Featurizes a list of sentences"""
        for sentence in sentences:
            self.featurize(sentence)

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

