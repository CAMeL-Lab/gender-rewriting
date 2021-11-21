import copy
import json

class InputExample:
    def __init__(self, src_tokens, src_tags, tgt_tokens, tgt_tags,
                 src_bert_tags=None):
        """
        Args:
            - src_tokens (list of str): The source tokens.
            - src_tags (list of str): The gender tags for each source token.
            - tgt_token (list of str): The target tokens.
            - tgt_tags (list of str): The gender tags for each target token.
            - src_bert_tags (list of str): The src bert-tagger src gender tags.
                                           Defaults to None.
        """
        self.src_tokens = src_tokens
        self.src_tags = src_tags
        self.tgt_tokens = tgt_tokens
        self.tgt_tags = tgt_tags
        self.src_bert_tags = src_bert_tags

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

class OutputExample:
    def __init__(self, sentence, proposed_by, scored_candidates=None):
        """
        Args:
            - sentence (str): The output sentence.
            - proposed_by (list): morph or cbr or oov. This is used for
                                  error analysis.
            - scored_candidates(list): list of scored candidates objects.
                                       This is used for error analysis
        """
        self.sentence = sentence
        self.scored_candidates = scored_candidates
        self.proposed_by = proposed_by

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

class Dataset:
    def __init__(self, src_path, tgt_path, src_bert_tags_path=None):
        """
        Args:
            - src_path (str): dir to gold tagged src tokens.
            - tgt_path (str): dir to gold tagged tgt tokens.
            - src_bert_tags_path (str): dir to bert tagged src tokens.
                                        Defaults to None.
        """
        self.input_examples = self.create_dataset(src_path, tgt_path,
                                                  src_bert_tags_path)

    def get_raw_data(self, data_dir):
        with open(data_dir, encoding='utf8', mode='r') as f:
            return f.readlines()

    def collate(self, raw_data_lines):
        """
        Args:
            - raw_data_lines (list): Sentences are separated by a new line.
        Returns:
            - tokens (list of list of str)
            - tags (list of list of str)
        """
        sent_tokens = []
        all_tokens = []
        sent_tags = []
        all_tags = []
        for i, ex in enumerate(raw_data_lines):
            ex = ex.strip().split()
            if len(ex) == 0:
                all_tokens.append(sent_tokens)
                all_tags.append(sent_tags)
                sent_tokens = []
                sent_tags = []
            else:
                token, tag = ex[0], ex[1]
                sent_tokens.append(token)
                sent_tags.append(tag)

        # adding the last sentence info
        all_tokens.append(sent_tokens)
        all_tags.append(sent_tags)
        return all_tokens, all_tags

    def create_dataset(self, src_path, tgt_path, src_bert_tags_path):
        # Read and collate src and target data
        src_data = self.get_raw_data(src_path)
        src_tokens, src_tags = self.collate(src_data)

        tgt_data = self.get_raw_data(tgt_path)
        tgt_tokens, tgt_tags = self.collate(tgt_data)

        assert len(src_tokens) == len(src_tags) == len(tgt_tokens) \
            == len(tgt_tags)

        # Read and collate bert src tags if needed
        if src_bert_tags_path:
            bert_preds_data = self.get_raw_data(src_bert_tags_path)
            _, src_bert_tags = self.collate(bert_preds_data)
            assert len(src_bert_tags) == len(src_tokens)
        else:
            src_bert_tags = None

        # Creating input examples
        input_examples = []
        for i in range(len(src_tokens)):
            ex_src_tokens = src_tokens[i]
            ex_src_tags = src_tags[i]
            ex_tgt_tokens = tgt_tokens[i]
            ex_tgt_tags = tgt_tags[i]
            ex_src_bert_tags = src_bert_tags[i] if src_bert_tags else None

            input_examples.append(InputExample(src_tokens=ex_src_tokens,
                                               src_tags=ex_src_tags,
                                               tgt_tokens=ex_tgt_tokens,
                                               tgt_tags=ex_tgt_tags,
                                               src_bert_tags=ex_src_bert_tags))

        return input_examples

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, idx):
        return self.input_examples[idx]

class Candidate:
    def __init__(self, masked_sentence, targets, proposed_by):
        """
        Args:
            - masked_sentence (list of str): Mutli-masked sentence
            - targets (list of list of str): Proposals to fill.
            - proposed_by (list): morph or cbr or oov. This is used for
                                  error analysis.
        """
        self.masked_sentence = masked_sentence
        self.targets = targets
        self.proposed_by = proposed_by

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output
