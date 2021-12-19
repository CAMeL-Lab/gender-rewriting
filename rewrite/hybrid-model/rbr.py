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
    def __init__(self, model, verbose_model):
        self.model = model
        self.verbose_model = verbose_model

    @classmethod
    def build_model(cls, dataset):
        """
        Args:
            - dataset (Dataset obj)
        Returns:
            - rbr model (a rule-based rewriting mode): The rbr model where the
            keys are either a rule or a tuple of (rule, trg_gender)
        """
        model = dict()
        verbose_model = defaultdict(set)

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
                    # TODO: make the shorter word thingy part of the edit dist.
                    shorter_word = (src_token if len(src_token) <= len(tgt_token)
                                              else tgt_token)
                    longer_word = (src_token if shorter_word == tgt_token
                                             else tgt_token)

                    trg_tag_1 = (src_token_tag if shorter_word == src_token
                                               else tgt_token_tag)
                    trg_tag_2 = (tgt_token_tag if shorter_word == src_token
                                               else src_token_tag)

                    # print(f'Token 1: {shorter_word}')
                    # print(f'Token 2: {longer_word}')
                    edit_distance_table = edit_distance(shorter_word,
                                                        longer_word)
                    # print(f'Edit Dist: {edit_distance_table[0][0]}')
                    # print(f'Backtracking:')
                    alignment = backtrack_dp(edit_distance_table, shorter_word,
                                             longer_word)
                    rule = convert_alignment_to_rule(alignment)
                    r1, r2 = re.sub('X+', 'X', rule[0]), re.sub('X+', 'X', rule[1])
                    # print(f'Rule: {rule}')
                    # print(f'Condensed Rule: {(r1, r2)}')
                    # print('================')
                    # we will learn rules in both directions (trg_gender, rule)
                    model[(trg_tag_2, rule[0])] = rule[1]
                    model[(trg_tag_1, rule[1])] = rule[0]
                    verbose_model[(rule[0], rule[1])].add((src_token, tgt_token))
                    verbose_model[(rule[1], rule[0])].add((src_token, tgt_token))

        return cls(model, verbose_model)

    def __len__(self):
        return len(self.model)

    def __getitem__(self, sw_tg):
        tgt_forms = list()
        tgt_gender, src_word = sw_tg
        for rule in self.model:
            tgt_gender, src_pattern = rule
            # print(tgt_gender, src_pattern)
            # constructing a regex from the rule
            src_pattern = src_pattern.replace('+', '').replace('-','').replace('X', '(.)')
            # checking if the pattern matches the word we need to rewrite
            match = re.match(src_pattern, src_word)

            if match and match[0] == src_word:
                # if there's a match, construct the target regex
                tgt_pattern = self.model[rule]
                tgt_pattern = tgt_pattern.replace('+', '').replace('-','')
                x_count = 1
                while 'X' in tgt_pattern:
                    tgt_pattern = tgt_pattern.replace('X', f'\\{x_count}', 1)
                    x_count += 1
                # print(rule, tgt_pattern)
                # generate target words
                trg_word = re.sub(src_pattern, tgt_pattern, src_word)
                tgt_forms.append(trg_word)
                # print(trg_word)
        return tgt_forms

def edit_distance(str1, str2):
    """
    Computes the Levenshtein edit distance between str1 and str2.
    Args:
        - str1 (str): the first input word.
        - str2 (str): the second input word.

    Returns:
        - dp (list of list of int): a matrix of edit distances where dp[0][0]
            contains the smallest edit distance between str1 and str1.
    """
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

def backtrack_dp(dp, w1, w2):
    """
    Takes a matrix of edit distances and backtracks to create an alignment
    between w1 and w2.
    Args:
        - dp (list of list of int): a matrix of edit distances
        - w1 (str): the first input word.
        - w2 (str): the second input word.

    Returns:
        - alignment (tuple of (str, str)): a tuple of two strings that
            represent the alignment between to go from one word to
            another or vice-versa.
    """
    i, j = 0, 0
    res1, res2 = "", ""

    while i < len(w1) and j < len(w2):
        if dp[i][j] == dp[i][j + 1] + 1:
            # print(f'Inserting {w2[j]} in {w1}')
            res1 += '+'
            res2 += w2[j]
            j += 1

        elif dp[i][j] == dp[i + 1][j] + 1:
            # print(f'Deleting {w1[i]} from {w1}')
            res1 += w1[i]
            res2 += '-'
            i += 1

        elif dp[i][j] == dp[i + 1][j + 1]:
            # print(f'Copying {w1[i]}')
            res1 += w1[i]
            res2 += w2[j]
            i += 1
            j += 1

        elif dp[i][j] == dp[i + 1][j + 1] + 1:
            # print(f'Subbing {w1[i]} with {w2[j]}')
            res1 += w1[i]
            res2 += w2[j]
            i += 1
            j += 1

    assert len(res1) <= len(res2)

    for k in range(j, len(w2)):
        # print(f'Inserting {w2[k]} in {w1}')
        res1 += '+'
        res2 += w2[k]

    assert len(res1) <= len(res2)
    return res1, res2

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
    str1, str2 = alignment
    assert len(str1) == len(str2)
    rule1 = ""
    rule2 = ""
    for i in range(len(str1)):
        if str1[i] == str2[i]:
            rule1 += 'X'
            rule2 += 'X'
        else:
            rule1 += str1[i]
            rule2 += str2[i]

    return (rule1, rule2)

