from collections import defaultdict
import json
import copy
import logging
import re
from utils.data_utils import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RBR:
    """
    Rule-based Rewriting
    """
    def __init__(self, model, counts, pick_top_rule=False,
                pick_top_tgt_rule=False):
        self.model = model
        self.counts = counts
        self.pick_top_rule = pick_top_rule
        self.pick_top_tgt_rule = pick_top_tgt_rule

    @classmethod
    def build_model(cls, dataset, pick_top_rule, pick_top_tgt_rule):
        """
        Args:
            - dataset (Dataset obj)
        Returns:
            - rbr model (a rule-based rewriting mode): The rbr model where the
            keys tuples of (rule, tgt_gender)
        """
        model = defaultdict(lambda: defaultdict(lambda: 0))
        counts = dict()

        for ex in dataset.input_examples:
            src_tokens = ex.src_tokens
            tgt_tokens = ex.tgt_tokens
            src_tokens_tags = ex.src_tags
            tgt_tokens_tags = ex.tgt_tags

            for i in range(len(src_tokens)):
                src_token = src_tokens[i]
                tgt_token = tgt_tokens[i]
                src_token_tag = src_tokens_tags[i]
                tgt_token_tag = tgt_tokens_tags[i]

                if src_token != tgt_token:
                    assert src_token_tag != tgt_token_tag

                    edit_distance_table = edit_distance(src_word=src_token,
                                                        tgt_word=tgt_token)
                    # print(f'Edit Dist: {edit_distance_table[0][0]}')
                    # print(f'Backtracking:')
                    src_align, tgt_align = backtrack_dp(edit_distance_table,
                                                        src_word=src_token,
                                                        tgt_word=tgt_token)

                    src_pattern, tgt_pattern = convert_alignment_to_rule((src_align,
                                                                         tgt_align))

                    # print(f'Rule: {rule}')
                    # print('================')
                    # we will learn rules in both directions (tgt_gender, rule)
                    model[(tgt_token_tag, src_pattern)][tgt_pattern] += 1
                    # rules counts
                    counts[(tgt_token_tag, src_pattern)] = (1 +
                                                            counts.get((tgt_token_tag, src_pattern), 0))

        return cls(model, counts, pick_top_rule, pick_top_tgt_rule)

    def __len__(self):
        return sum([val for key, val in self.counts.items()])

    def __getitem__(self, sw_tg):
        """
        Returns generated token(s) based on the matched rules given
        a source word and a target gender

        Note: we have two forms for rules to pick from:
            1) pick_top_rule: Selects the (src_pattern, tgt_gender)
                rule that occured the most in the training data.

            2) pick_top_tgt_rule: Selects the target_pattern that appeared
                the most for a given (src_pattern, tgt_gender). Because
                one (src_pattern, tgt_gender) could have multiple target
                patterns

            We also do not apply any pattern that appeared only 1 during
            training to reduce noisy outputs.
        """

        tgt_gender, src_word = sw_tg
        matched_rules = self.match_rule(src_word, tgt_gender)

        if not matched_rules: return None

        generated_tokens = []
        if self.pick_top_rule:
            matched_rule = max(matched_rules, key=lambda x: x['rule_freq'])
            # ignore the matched rules that appear only once
            if matched_rule['rule_freq'] != 1:
                if self.pick_top_tgt_rule:
                    tgt_rule, tgt_rule_freq = max(matched_rule['targets'].items(),
                                                 key=lambda x: x[1])

                    # ignore target rules that appear only once
                    if tgt_rule_freq != 1:
                        generated_token = self.generate_token(tgt_rule,
                                                        matched_rule['src_word'],
                                                        matched_rule['source_pattern'])
                        generated_tokens.append(generated_token)

                else:
                    generated_tokens = []
                    for tgt_rule, tgt_rule_freq in matched_rule['targets'].items():
                        # ignore target rules that appear only once
                        if tgt_rule_freq == 1: continue

                        generated_token = self.generate_token(tgt_rule,
                                                        matched_rule['src_word'],
                                                        matched_rule['source_pattern'])
                        generated_tokens.append(generated_token)

        else:
            generated_tokens = []
            for matched_rule in matched_rules:
                # ignore the matched rules that appear only once
                if matched_rule['rule_freq'] == 1: continue

                if self.pick_top_tgt_rule:
                    tgt_rule, tgt_rule_freq = max(matched_rule['targets'].items(),
                                                  key=lambda x: x[1])
                    # ignore target rules that appear only once
                    if tgt_rule_freq == 1: continue

                    generated_token = self.generate_token(tgt_rule,
                                                    matched_rule['src_word'],
                                                    matched_rule['source_pattern'])
                    generated_tokens.append(generated_token)

                else:
                    for tgt_rule, tgt_rule_freq in matched_rule['targets'].items():
                        # ignore target rules that appear only once
                        if tgt_rule_freq == 1: continue
                        
                        generated_token = self.generate_token(tgt_rule,
                                                        matched_rule['src_word'],
                                                        matched_rule['source_pattern'])
                        generated_tokens.append(generated_token)

        return generated_tokens

    def match_rule(self, src_word, tgt_gender):
        """
        Returns all the rules that match the src word pattern
        and the target gender
        """
        matched_rules = []
        for rule in self.model:
            _, src_pattern = rule
            pattern = src_pattern.replace('+', '').replace('-','').replace('X', '(.)')
            match = re.match(pattern, src_word)

            # matching on the pattern and the target gender
            if match and match[0] == src_word and tgt_gender == rule[0]:
                matched_rules.append({'src_word': src_word,
                                      'trg_gender': rule[0],
                                      'source_pattern': pattern,
                                      'targets': dict(self.model[rule]),
                                      'rule_freq': self.counts[rule]})

        return matched_rules

    def generate_token(self, tgt_rule, src_word, src_pattern):
        """
        Generates a token given a tgt pattern, src word, and a src pattern
        """
        tgt_pattern = tgt_rule.replace('+', '').replace('-','')
        x_count = 1
        while 'X' in tgt_pattern:
            tgt_pattern = tgt_pattern.replace('X', f'\\{x_count}', 1)
            x_count += 1
        # generate target words
        tgt_word = re.sub(src_pattern, tgt_pattern, src_word)
        return tgt_word

def edit_distance(src_word, tgt_word):
    """
    Computes the Levenshtein edit distance between src_word and tgt_word.
    Args:
        - src_word (str): the source input word.
        - tgt_word (str): the target input word.

    Returns:
        - dp (list of list of int): a matrix of edit distances where dp[0][0]
            contains the smallest edit distance between str1 and str1.
    """
    str1 = src_word if len(src_word) <= len(tgt_word) else tgt_word
    str2 = src_word if str1 == tgt_word else tgt_word
    m = len(str1)
    n = len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # base cases
    for row in range(m):
        dp[row][n] = m - row

    for col in range(n):
        dp[m][col] = n - col

    # Bottom up dp
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            add = 1 if str1[i] != str2[j] else 0
            dp[i][j] = min(dp[i + 1][j] + 1,
                           dp[i][j + 1] + 1,
                           dp[i + 1][j + 1] + add)

    return dp

def backtrack_dp(dp, src_word, tgt_word):
    """
    Takes a matrix of edit distances and backtracks to create an alignment
    between w1 and w2.
    Args:
        - dp (list of list of int): a matrix of edit distances
        - src_word (str): the source input word.
        - tgt_word (str): the target input word.

    Returns:
        - alignment (tuple of (str, str)): a tuple of two strings that
            represent the alignment between to go from one word to
            another or vice-versa.
    """

    w1 = src_word if len(src_word) <= len(tgt_word) else tgt_word
    w2 = src_word if w1 == tgt_word else tgt_word

    i, j = 0, 0
    w1_align, w2_align = "", ""

    while i < len(w1) and j < len(w2):
        if dp[i][j] == dp[i][j + 1] + 1:
            # print(f'Inserting {w2[j]} in {w1}')
            w1_align += '+'
            w2_align += w2[j]
            j += 1

        elif dp[i][j] == dp[i + 1][j] + 1:
            # print(f'Deleting {w1[i]} from {w1}')
            w1_align += w1[i]
            w2_align += '-'
            i += 1

        elif dp[i][j] == dp[i + 1][j + 1]:
            # print(f'Copying {w1[i]}')
            w1_align += w1[i]
            w2_align += w2[j]
            i += 1
            j += 1

        elif dp[i][j] == dp[i + 1][j + 1] + 1:
            # print(f'Subbing {w1[i]} with {w2[j]}')
            w1_align += w1[i]
            w2_align += w2[j]
            i += 1
            j += 1

    assert len(w1_align) <= len(w2_align)

    for k in range(j, len(w2)):
        # print(f'Inserting {w2[k]} in {w1}')
        w1_align += '+'
        w2_align += w2[k]

    assert len(w1_align) <= len(w2_align)

    if w1 == src_word and w2 == tgt_word:
        src_align, tgt_align = w1_align, w2_align

    elif w2 == src_word and w1 == tgt_word:
        src_align, tgt_align = w2_align, w1_align

    return src_align, tgt_align

def convert_alignment_to_rule(alignment):
    """
    Converts an alignment to a rule.
    Args:
        - alignment (tuple of (str, str)): a tuple of two strings that
            represent the alignment between to go from one word to
            another or vice-versa.

    Returns:
        - rule (tuple of (str, str)): a tuple of two strings that represent
            the rules to convert one word to another or vice-versa.
    """
    src_align, tgt_align = alignment
    assert len(src_align) == len(tgt_align)
    src_pattern = ""
    tgt_pattern = ""
    for i in range(len(src_align)):
        if src_align[i] == tgt_align[i]:
            src_pattern += 'X'
            tgt_pattern += 'X'
        else:
            src_pattern += src_align[i]
            tgt_pattern += tgt_align[i]

    return (src_pattern, tgt_pattern)

