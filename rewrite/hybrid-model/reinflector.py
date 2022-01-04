from utils.data_utils import Dataset, Candidate, OutputExample
from collections import defaultdict
from cbr import CBR, build_ngrams
from ranker import Ranker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenderReinflector:

    def __init__(self, cbr_model, morph_reinflector, rbr_model, neural_model,
                 ranker=None,
                 first_person_only=False):
        """
        Args:
            - cbr_model (default dict): The cbr model where the
                keys are (source_word, target_word_gender) and vals
                are target_words.
            - morph_reinflector (MorphReinflctor obj): The morph reinflector.
            - rbr_model (default dict): The rbr model where the keys are
               (target_word_gender, source_pattern) and vals are dicts
               of target_patterns
            - neural_model ():
            - ranker (Ranker object): The ranker.
            - first_person_only (bool): To do *only* first person gender
                                        reinfelction.
        """
        self.cbr_model = cbr_model
        self.morph_reinflector = morph_reinflector
        self.ranker = ranker
        self.rbr_model = rbr_model
        self.neural_model = neural_model
        self.first_person_only = first_person_only

    def get_base_and_clitic_target_gender(self, tag, speaker_gender,
                                          listener_gender):

        base_word_gender, clitic_gender = tag.split('+')
        target_word_gender, target_clitic_gender = base_word_gender, clitic_gender

        if base_word_gender == '1M' and speaker_gender == '1F':
            target_word_gender = '1F'
        elif base_word_gender == '1F' and speaker_gender == '1M':
            target_word_gender = '1M'
        elif base_word_gender == '2M' and listener_gender == '2F':
            target_word_gender = '2F'
        elif base_word_gender == '2F' and listener_gender == '2M':
            target_word_gender = '2M'

        if clitic_gender == '1M' and speaker_gender == '1F':
            target_clitic_gender = '1F'
        elif clitic_gender == '1F' and speaker_gender == '1M':
            target_clitic_gender = '1M'
        elif clitic_gender == '2M' and listener_gender == '2F':
            target_clitic_gender = '2F'
        elif clitic_gender == '2F' and listener_gender == '2M':
            target_clitic_gender = '2M'

        return f'{target_word_gender}+{target_clitic_gender}'

    def reinflect(self, dataset, speaker_gender, listener_gender=None,
                  use_cbr=True, pick_top_mle=True, use_morph=True,
                  use_rbr=True, use_neural=True):

        """
        Args:
            - dataset (Dataset object): contains list the input examples.
                                        The input examples *must* contain bert
                                        predicted gender tags.
            - speaker_gender (str): M or F.
            - listener_gender (str): M or F or None (if we're doing 1st per
                                                     gender reinflection only).
            - use_cbr (bool): to use cbr model or not.
            - use_morph (bool): to use morphological analyzer and reinflector
                                or not.
            - use_rbr (bool): to use rbr model or not.
            - use_neural (bool): to use the neural model or not.
        Returns:
            - candidates (list): candidate objects.
        """
        candidates = []
        inf_analysis_stats = defaultdict(int)
        oov_stats = defaultdict(int)

        if not self.first_person_only:
            # Adding person annotations to be compatible with token annoations
            speaker_gender = '1' + speaker_gender
            listener_gender = '2' + listener_gender
        else:
            speaker_gender = '1' + speaker_gender
            assert listener_gender == None

        for i, example in enumerate(dataset):
            src_tokens = example.src_tokens
            pred_tags = example.src_bert_tags

            # building ngrams for the CBR model if needed
            if use_cbr:
                tokens_ngrams = build_ngrams(src_tokens,
                                             ngrams=self.cbr_model.ngrams,
                                             pad_left=True)

            candidate_sentence = []
            candidate_targets = []
            proposed_by = []

            for j, (token, tag) in enumerate(zip(src_tokens, pred_tags)):
                # if the token tag is B+B or if it matches
                # the speaker gender tag or the listener gender, pass the 
                # token as it is
                if (tag == 'B+B' or
                    tag == f'B+{listener_gender}' or
                    tag == f'{listener_gender}+B' or
                    tag == f'B+{speaker_gender}' or
                    tag == f'{speaker_gender}+B' or
                    tag == f'{speaker_gender}+{listener_gender}' or
                    tag == f'{listener_gender}+{speaker_gender}' or
                    tag == f'{listener_gender}+{listener_gender}' or
                    tag == f'{speaker_gender}+{speaker_gender}'):

                    candidate_sentence.append(token)
                    proposed_by.append('NA')
                    inf_analysis_stats['reg_passes'] += 1

                else:
                    # Getting the target gender based on the provided
                    # user preferences and predicted token tag
                    if self.first_person_only:
                        target_gender = self.get_base_and_clitic_target_gender(tag,
                                                                               listener_gender=None,
                                                                               speaker_gender=speaker_gender)
                    else:
                        target_gender = self.get_base_and_clitic_target_gender(tag,
                                                                               listener_gender=listener_gender,
                                                                               speaker_gender=speaker_gender)

                    is_oov = True

                    # use CBR model to get reinflections 
                    if use_cbr:
                        cbr_candidates = self.cbr_model[(tokens_ngrams[j],
                                                        target_gender)]
                        inf_analysis_stats['cbr_triggers'] += 1

                        if cbr_candidates:
                            # if there are multiple
                            # reinflections and pick_top_mle, get the most
                            # probable reinflection. Otherwise, create a mask
                            # sentence and expand targets
                            is_oov = False
                            if len(cbr_candidates) > 1:
                                if pick_top_mle:
                                    reinflected_token = max(cbr_candidates.items(),
                                                        key=lambda x: x[1])[0]
                                    candidate_sentence.append(reinflected_token)
                                    proposed_by.append('CBR')
                                else:
                                    candidate_sentence.append('[MASK]')
                                    token_targets = list(cbr_candidates.keys())
                                    # removing the token itself if it appears
                                    # within the targets
                                    if token in token_targets:
                                        token_targets.remove(token)
                                    candidate_targets.append(token_targets)
                                    proposed_by.append('CBR')
                            else:
                                # if there is a single reinflection, return it.
                                # but if the generated option is equal
                                # to the input token, don't return it and 
                                # consider the token to be an OOV.
                                if list(cbr_candidates.keys())[0] == token:
                                    is_oov = True
                                else:
                                    candidate_sentence.append(list(cbr_candidates.keys())[0])
                                    proposed_by.append('CBR')
                        else:
                            oov_stats['cbr_oov'] += 1

                    # use morph if CBR fail
                    # or as a stand alone model
                    if ((use_cbr and use_morph and is_oov) or
                          (use_morph and is_oov)):

                        inf_analysis_stats['morph_triggers'] += 1
                        morph_res = self.morph_reinflector.reinflect(token,
                                                                     tag,
                                                                     target_gender)

                        if morph_res:
                            reinflected_token = morph_res['reinflected_token']
                            token_candidates = morph_res['proposals']
                            proposal_src = morph_res['proposed_by']
                            is_oov = False
                            candidate_sentence.append(reinflected_token)
                            proposed_by.append(proposal_src)
                            # If there are multiple targets, expand targets
                            if len(token_candidates) != 0:
                                assert reinflected_token == '[MASK]'
                                candidate_targets.append(token_candidates)
                        else:
                            oov_stats['morph_oov'] += 1

                    # use RBR if morph or CBR (or both) fail
                    # or as a stand alone model
                    if ((use_cbr and use_morph and use_rbr and is_oov) or
                        (use_rbr and is_oov)):

                        rbr_candidates = self.rbr_model[(target_gender, token,
                                                         tag)]

                        if rbr_candidates:
                            inf_analysis_stats['rbr_triggers'] += 1
                            is_oov = False
                            if len(rbr_candidates) > 1:
                                candidate_sentence.append('[MASK]')
                                candidate_targets.append(rbr_candidates)
                                proposed_by.append('RBR')
                            else:
                                reinflected_token = rbr_candidates[0]
                                candidate_sentence.append(reinflected_token)
                                proposed_by.append('RBR')
                        else:
                            oov_stats['rbr_oov'] += 1

                    # use seq2seq if RBR or morph or CBR (or all) fail
                    # or as a stand alone model
                    if ((use_cbr and use_morph and use_rbr and use_neural 
                         and is_oov) or (use_neural and is_oov)):

                        neural_candidates = self.neural_model.reinflect(token=token,
                                                                        target_gender=target_gender)
                        inf_analysis_stats['neural_triggers'] += 1
                        is_oov = False

                        if len(neural_candidates) > 1:
                            candidate_sentence.append('[MASK]')
                            candidate_targets.append(neural_candidates)
                            proposed_by.append('seq2seq')
                        else:
                            reinflected_token = neural_candidates[0]
                            candidate_sentence.append(reinflected_token)
                            proposed_by.append('seq2seq')

                    if is_oov:
                        # If everything fails, pass the oov word as it is
                        candidate_sentence.append(token)
                        proposed_by.append('OOV')

            candidates.append(Candidate(masked_sentence=candidate_sentence,
                              targets=candidate_targets,
                              proposed_by=proposed_by))


        for i, candidate in enumerate(candidates):
            if candidate.targets:
                inf_analysis_stats['selection_triggers'] += 1

        logger.info(f"CBR triggers: {inf_analysis_stats['cbr_triggers']}")
        logger.info(f"Morph triggers: {inf_analysis_stats['morph_triggers']}")
        logger.info(f"RBR triggers: {inf_analysis_stats['rbr_triggers']}")
        logger.info(f"Neural triggers: {inf_analysis_stats['neural_triggers']}")
        logger.info(f"Regular passes: {inf_analysis_stats['reg_passes']}")
        logger.info(f"Selection triggers: {inf_analysis_stats['selection_triggers']}")
        logger.info("===========================")
        logger.info(f"CBR OOV: {oov_stats['cbr_oov']}")
        logger.info(f"Morph OOV: {oov_stats['morph_oov']}")
        logger.info(f"RBR OOV: {oov_stats['rbr_oov']}")
        logger.info("===========================")

        return candidates


    def select(self, candidates):
        """
        Args:
            - candidates (list): candidate objects.

        Returns:
            - reinflections (list): output example objects.
        """
        reinflections = []
        for i, candidate in enumerate(candidates):
            sentence = ' '.join(candidate.masked_sentence)
            if candidate.targets:
                scored = self.ranker.fill_and_rank(sentence=sentence,
                                                   targets=candidate.targets)

                reinflections.append(OutputExample(sentence=scored[0].sentence,
                                                   scored_candidates=scored[1:],
                                                   proposed_by=candidate.proposed_by))
            else:
                reinflections.append(OutputExample(sentence=sentence,
                                                   scored_candidates=None,
                                                   proposed_by=candidate.proposed_by))

        return reinflections

