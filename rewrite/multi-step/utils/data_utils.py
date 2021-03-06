import copy
import json

class InputExample:
    def __init__(self, src_tokens, src_tags, tgt_tokens, tgt_tags):
        """
        Args:
            - src_tokens (list of str): The source tokens.
            - src_tags (list of str): The gender tags for each source token.
            - tgt_token (list of str): The target tokens.
            - tgt_tags (list of str): The gender tags for each target token.
        """
        self.src_tokens = src_tokens
        self.src_tags = src_tags
        self.tgt_tokens = tgt_tokens
        self.tgt_tags = tgt_tags

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

class OutputExample:
    def __init__(self, sentence, proposed_by, pred_src_gen_tags,
                 scored_candidates=None):
        """
        Args:
            - sentence (str): The output sentence.
            - proposed_by (list): morph, cbr, neural or oov. This is used for
                                  error analysis.
            - pred_src_gen_tags (list): Predicted word-level source gender
                                        tags. This is used for error analysis.
            - scored_candidates(list): list of scored candidates objects.
                                       This is used for error analysis
        """
        self.sentence = sentence
        self.scored_candidates = scored_candidates
        self.pred_src_gen_tags = pred_src_gen_tags
        self.proposed_by = proposed_by

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

class Dataset:
    def __init__(self, src_path, tgt_path=None):
        """
        Args:
            - src_path (str): dir to gold tagged src tokens.
            - tgt_path (str): dir to gold tagged tgt tokens.
                              Defaults to None in case we are doing inference
                              with no labels.
        """
        self.input_examples = self.create_dataset(src_path, tgt_path)

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

            elif len(ex) == 2:
                token, tag = ex[0], ex[1]
                sent_tokens.append(token)
                sent_tags.append(tag)

            elif len(ex) == 1: # if there's no tag during test time
                token, tag = ex[0], None
                sent_tokens.append(token)
                sent_tags.append(tag)

        # adding the last sentence info
        # all_tokens.append(sent_tokens)
        # all_tags.append(sent_tags)
        return all_tokens, all_tags

    def create_dataset(self, src_path, tgt_path):
        # Read and collate src and target data
        src_data = self.get_raw_data(src_path)
        src_tokens, src_tags = self.collate(src_data)
        assert len(src_tokens) == len(src_tags)

        # Read and collate target data if needed
        if tgt_path:
            tgt_data = self.get_raw_data(tgt_path)
            tgt_tokens, tgt_tags = self.collate(tgt_data)
            assert len(tgt_tokens) == len(tgt_tags)
        else:
            tgt_tokens, tgt_tags = None, None

        # Creating input examples
        input_examples = []
        for i in range(len(src_tokens)):
            ex_src_tokens = src_tokens[i]
            ex_src_tags = src_tags[i]
            ex_tgt_tokens = tgt_tokens[i] if tgt_tokens else None
            ex_tgt_tags = tgt_tags[i] if tgt_tags else None

            input_examples.append(InputExample(src_tokens=ex_src_tokens,
                                               src_tags=ex_src_tags,
                                               tgt_tokens=ex_tgt_tokens,
                                               tgt_tags=ex_tgt_tags))

        return input_examples

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, idx):
        return self.input_examples[idx]

class Candidate:
    def __init__(self, masked_sentence, targets, pred_src_gen_tags,
                 proposed_by):
        """
        Args:
            - masked_sentence (list of str): Mutli-masked sentence
            - targets (list of list of str): Proposals to fill.
            - pred_src_gen_tags (list): Predicted word-level source gender
                                        tags. This is used for error analysis.
            - proposed_by (list): morph, cbr, neural or oov. This is used for
                                  error analysis.
        """
        self.masked_sentence = masked_sentence
        self.targets = targets
        self.pred_src_gen_tags = pred_src_gen_tags
        self.proposed_by = proposed_by

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output
