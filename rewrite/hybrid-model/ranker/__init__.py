from mlm.scorers import MLMScorerPT, MLMScorer
from mlm.models import get_pretrained
import itertools
import json
import copy
import mxnet as mx

class ScoredCandidate:
  def __init__(self, sentence, pll, targets):
    """
    Args:
        - sentence (str): The filled sentence.
        - pll (float): pseudo-log-likelihood.
        - targets (list of list of str): The given targets.
    """
    self.sentence = sentence
    self.pll = pll
    self.targets = targets

  def __repr__(self):
      return str(self.to_json_str())

  def to_json_str(self):
      return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

  def to_dict(self):
      output = copy.deepcopy(self.__dict__)
      return output

class Ranker:
  def __init__(self, model_name, use_gpu=True):
    """
    Args:
      - model_name (str): BERT model card name.
      - use_gpu (bool): To use GPU or not.
        NOTE: As of now, the pretrained model name must start with
             "bert" or "roberta" etc.
              TODO: Bashar is to tweak mlm-scorer to handle other models.
              For now, we just load camel bert from a local copy.
    """
    if use_gpu:
      ctxs = [mx.gpu(0)]
    else:
      ctxs = [mx.cpu()]

    self.model, self.vocab, self.tokenizer = get_pretrained(ctxs, model_name)
    self.scorer = MLMScorerPT(self.model, self.vocab, self.tokenizer, ctxs)
    self.mask_token = self.tokenizer.mask_token
    self.mask_idx = self.tokenizer.mask_token_id

  def fill_and_rank(self, sentence, targets):
    """
    Args:
      - sentence (str): The mutli-mask input sentence.
      - targets (list of list of str): Each sublist correspond to a mask.

    Returns:
      - scored_candidates (list): list of ScoredCandidate objects.
    """

    # Get the mask indices
    mask_indices = [i for i, t in enumerate(sentence.split())
                    if t == self.mask_token]

    scored_candidates = []
    if len(mask_indices) == 1:
      # If there's a single mask, generate candidates for each target
      # and use the mlm-scorer to score them
      filled_sentences = self.fill_mask(sentence, targets[0], mask_indices)
      scores = self.scorer.score_sentences(filled_sentences)

      for sent, score, target in zip(filled_sentences, scores, targets[0]):
        scored_candidates.append(ScoredCandidate(sent, score, [target]))

      # print(scored_candidates)
    else:
      # import pdb; pdb.set_trace()
      # If there are multiple masks, generate candidates for each target
      # combination and score them using mlm scorer.
      target_combos = list(itertools.product(*targets))
      filled_sentences = []
      for combo in target_combos:
        filled_combo = self.fill_mask(sentence, list(combo), mask_indices)
        filled_sentences.append(filled_combo[0])

      # score the sentences using mlm-scorer
      scores = self.scorer.score_sentences(filled_sentences)
      scored_candidates = []

      for sent, score, target in zip(filled_sentences, scores, target_combos):
        scored_candidates.append(ScoredCandidate(sent, score, target))

    # sort the scored candidates
    scored_candidates = sorted(scored_candidates, key=lambda x: x.pll,
                               reverse=True)
    return scored_candidates

  def fill_mask(self, sentence, targets, mask_indices):
    """
    Args:
      - sentence (str): The mutli-mask input sentence.
      - targets (list of str): Each sublist correspond to a mask.
      - mask_indices (list of int): Mask indices.

    Returns:
      - filled_sentences (list of str)
    """
    filled_sentences = []
    if len(mask_indices) == 1:
      # In the case of a single mask, we have multiple filled candidates.
      zipped_targets_masks = itertools.zip_longest(targets, mask_indices,
                                                   fillvalue=mask_indices[0])
      for target, mask_idx in zipped_targets_masks:
        filled_sentence = sentence.split()
        filled_sentence[mask_idx] = target
        filled_sentence = ' '.join(filled_sentence)
        filled_sentences.append(filled_sentence)

    else:
      # In the case of multi mask, we have a single filled candidate. 
      zipped_targets_masks = zip(targets, mask_indices)
      filled_sentence = sentence.split()
      for target, mask_idx in zipped_targets_masks:
        filled_sentence[mask_idx] = target
      filled_sentence = ' '.join(filled_sentence)
      filled_sentences.append(filled_sentence)

    return filled_sentences
