from camel_tools.morphology.reinflector import Reinflector
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.utils.dediac import dediac_ar
from camel_tools.disambig.mle import MLEDisambiguator
from utils.data_utils import Dataset, Candidate, OutputExample
from cbr import CBR, build_ngrams
import editdistance
import operator
from ranker import Ranker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CBR_OOV = 0
MORPH_AMBIG = 0
GENDER_FILTERING = 0
CLITIC_FILTERING = 0

# def closest_ana(analyses, token):
#     """
#     Args:
#         - analyses (list): list of analyses
#         - token (str): word.

#     Returns:
#         - min_idx (int): The idx of the analysis that has a diac
#                          with the minimum edit distance to token.
#     """
#     min_dist = 1e10
#     min_idx = 0
#     filtered_analyses = []
#     for i, ana in enumerate(analyses):
#         edit_dis = editdistance.eval(dediac_ar(ana['diac']), token)
#         if edit_dis < min_dist:
#             min_dist = edit_dis
#             min_idx = i
#     return min_idx

class GenderReinflector:

    def __init__(self, cbr_model, morph_database, ranker=None,
                 first_person_only=False):
        """
        Args:
            - cbr_model (default dict): The cbr model where the
                keys are (source_word, target_word_gender) and vals
                are target_words.
            - morph_database (str): Path to the mophological database to use.
            - ranker (Ranker object): The ranker.
            - first_person_only (bool): To do *only* first person gender
                                        reinfelction.
        """
        self.cbr_model = cbr_model
        self.db = MorphologyDB(morph_database, flags='r')
        self.analyzer = Analyzer(self.db)
        self.reinflector = Reinflector(self.db)
        self.ranker = ranker
        self.first_person_only = first_person_only

    def get_base_and_clitic_target_gender(self, tag, speaker_gender,
                                          listener_gender):

        base_word_gender, clitic_gender = tag.split('+')
        target_word_gender, target_clitic_gender = 'B', 'B'

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

    # def morph_reinflect(self, token, tag, target_gender):
    #     """
    #     Uses morphology to do the gender reinflection

    #     Args:
    #         - token (str): the src token.
    #         - tag (str): the predicted bert src token tag.
    #         - target gender (str): the target gender.

    #     Returns:
    #         - reinflected_token, proposals, proposed by (tuple)
    #     """

    #     if self.first_person_only:
    #         gen = tag
    #     else:
    #         # Getting the gender and person info from the tag
    #         per = tag[0]
    #         gen = tag[1]

    #     # Get the analyses of the token using the analyzer
    #     analyses = self.analyzer.analyze(token)

    #     # We know the gender of the token and we know
    #     # its person. Let's filter based on the enclitic first 
    #     # then based on gender. If clitic lookup fails, use gender
    #     # to filter. This will mitigate the problem of verbs like:
    #     # أريدكن

    #     is_enclitic = False
    #     filtered_analyses_enclitic = []

    #     # we only do enclitic filtering for the multi-user case because the
    #     # enclitic gender marking only appear in 2nd person and *not* in first
    #     # Arabic doesn't show gender in first-person verb
    #     if not self.first_person_only:
    #         filtered_analyses_enclitic = [ana for ana in analyses
    #                                       if tag.lower() in ana['enc0']]

    #     if filtered_analyses_enclitic:
    #         filtered_analyses = list(filtered_analyses_enclitic)
    #         is_enclitic = True
    #         global CLITIC_FILTERING
    #         CLITIC_FILTERING += 1
    #     # do base word gender filtering if enclitic filtering fails
    #     # or if we're doing first person only refinflection
    #     else:
    #         filtered_analyses = [ana for ana in analyses
    #                              if gen.lower() == ana['gen']]
    #         global GENDER_FILTERING
    #         GENDER_FILTERING += 1

    #     anas_diacs = list(set([dediac_ar(ana['diac']) for ana
    #                             in filtered_analyses]))

    #     # If there exists more than one analysis
    #     # with different diacs, do another step of filtering
    #     # where we pick the analyses that have the diac closest
    #     # to the input token

    #     if len(anas_diacs) > 1:
    #         final_analysis = closest_ana(filtered_analyses, token)
    #         filtered_analyses = [filtered_analyses[final_analysis]]
    #         logger.info(f'{token} {anas_diacs} {gen}'
    #                      '--> EDIT DISTANCE FILTERING')

    #     # If no analysis was found, output the token as it is.
    #     # otherwise, use camel tools reinflector
    #     if len(filtered_analyses) == 0:
    #         # import pdb; pdb.set_trace()
    #         logger.info(f'{token} {gen} --> NO ANALYSIS')
    #         return token, None, 'OOV'

    #     else:
    #         # At this stage, all filtered_analyses will have 
    #         # the same diac so let's just use the diac of
    #         # the first analysis
    #         assert len(set([dediac_ar(ana['diac'])
    #                     for ana in filtered_analyses])) == 1

    #         ana = filtered_analyses[0]
    #         word = dediac_ar(ana['diac'])

    #         # if the filtering was done based on the enclitic,
    #         # generate using the target gender enclitic
    #         if is_enclitic:
    #             word_enc = ana['enc0']
    #             target_enc = word_enc.replace(gen.lower(),
    #                                           target_gender.lower()) 
    #             features = {'enc0': target_enc.lower()}

    #         else: # otherwise, use the word base gender
    #             features = {'gen': target_gender.lower()}

    #         # Pass the diac to the reinflector
    #         reinflected_analyses = self.reinflector.reinflect(word,
    #                                                     features)

    #         re_diacs = list(set([dediac_ar(ana['diac']) for ana
    #                                 in reinflected_analyses]))

    #         # If no reinflection was found due to a
    #         # bert tagger issue or a spelling error issue,
    #         # pass the token as it is
    #         if len(re_diacs) == 0:
    #             reinflected_token = token
    #             logger.info(f'{token} {gen} --> NO REINFLECTION')
    #             return reinflected_token, None, 'OOV'

    #         # If there's a single reinflection, return it.
    #         elif len(re_diacs) == 1:
    #             reinflected_token = re_diacs[0]
    #             return reinflected_token, None, 'Morph'
    #         # if the reinflector returns multiple
    #         # reinflections, add a mask and expand targets
    #         else:
    #             return '[MASK]', re_diacs, 'Morph'

    def morph_reinflect(self, token, tag, target_gender):
        """
        Uses morphology to do the gender reinflection

        Args:
            - token (str): the src token.
            - tag (str): the predicted bert src token tag.
            - target gender (str): the target gender.

        Returns:
            - reinflected_token, proposals, proposed by (tuple)
        """

        base_word_gender, clitic_gender = tag.split('+')
        target_word_gender, target_clitic_gender = target_gender.split('+')

        # Get the analyses of the token using the analyzer
        analyses = self.analyzer.analyze(token)

        if not analyses:
            print(f'No analyes found for {token}')
            return {'reinflected_token': token,
                    'proposals': [], 
                    'proposed_by': 'Analyzer_OOV'}

        # get the analyses we care about
        filtered_analyses = []
        if base_word_gender != 'B':
            filtered_analyses += [ana for ana in analyses
                                  if base_word_gender[1].lower() == ana['gen']]

        if clitic_gender != 'B':
            filtered_analyses += [ana for ana in analyses
                                  if clitic_gender[1].lower() in ana['enc0']]

        if not filtered_analyses:
            print(f'No analyses match the filtering criteria {tag} for {token}')
            return {'reinflected_token':token,
                    'proposals': [], 
                    'proposed_by': 'Analyzer_OOV'}

        # try to reinflect all analyses
        reinflections = []

        for ana in filtered_analyses:
            features = {'pos': ana['pos']}

            if target_word_gender != 'B' and 'gen' in ana:
                features['gen'] = target_word_gender[1].lower()

            if target_clitic_gender != 'B' and 'enc0' in ana:
                features['enc0'] = ana['enc0'].replace(clitic_gender[1].lower(),
                                                       target_clitic_gender[1].lower())

            reinflection_analyses = self.reinflector.reinflect(word=dediac_ar(ana['diac']),
                                                               feats=features)


        reinflections += [dediac_ar(re['diac']) for re in reinflection_analyses]
        reinflections = list(set(reinflections))

        if len(reinflections) == 0:
            print(f'No reinflections found for {token} with {features}')

            return {'reinflected_token': token,
                    'proposals': [], 
                    'proposed_by': 'Reinflector_OOV'}


        if len(reinflections) == 1:
            return {'reinflected_token': reinflections[0],
                    'proposals': [], 
                    'proposed_by': 'Reinflector'}


        elif len(reinflections) > 1:
            return {'reinflected_token': '[MASK]',
                    'proposals': reinflections, 
                    'proposed_by': 'Reinflector'}


    def reinflect(self, dataset, speaker_gender, listener_gender=None,
                  use_cbr=True, use_morph=True, pick_top_mle=True):
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
        Returns:
            - candidates (list): candidate objects.
        """
        candidates = []
        cbr_trigger = 0
        reinflection_trigger = 0
        regular_passes = 0

        if not self.first_person_only:
            # Adding person annotations to be compatible with token annoations
            speaker_gender = '1' + speaker_gender
            listener_gender = '2' + listener_gender

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
                    tag == f'{listener_gender}+{speaker_gender}'):

                    candidate_sentence.append(token)
                    proposed_by.append('NA')
                    regular_passes += 1

                else:
                    # import pdb; pdb.set_trace()
                    # Use person info in the tag to decide the target gender
                    if self.first_person_only:
                        # TODO: Currently, first person token annotations
                        # don't include gender clitics
                        target_gender = speaker_gender
                    else:
                        # target_gender = (speaker_gender[1] if '1' in tag
                        #                  else listener_gender[1])
                        target_gender = self.get_base_and_clitic_target_gender(tag,
                                                                               listener_gender=listener_gender,
                                                                               speaker_gender=speaker_gender)
                    # USE CBR to get reinflections 
                    if use_cbr:
                        cbr_candidates = self.cbr_model[(tokens_ngrams[j],
                                                        target_gender)]

                        if cbr_candidates:
                            # if there are multiple
                            # reinflections and pick_top_mle, get the most
                            # probable reinflection. Otherwise, create a mask
                            # sentence and expand targets
                            cbr_trigger += 1
                            if len(cbr_candidates) > 1:
                                if pick_top_mle:
                                    reinflected_token = max(cbr_candidates.items(),
                                                        key=operator.itemgetter(1))[0]
                                    candidate_sentence.append(reinflected_token)
                                    proposed_by.append('CBR')
                                else:
                                    candidate_sentence.append('[MASK]')
                                    token_targets = list(cbr_candidates.keys())
                                    candidate_targets.append(token_targets)
                                    proposed_by.append('CBR')
                            else:
                                # if there is a single reinflection, return it.
                                candidate_sentence.append(list(cbr_candidates.keys())[0])
                                proposed_by.append('CBR')

                        elif use_morph:
                            import pdb; pdb.set_trace()
                            global CBR_OOV
                            CBR_OOV += 1
                            # If the reinflection was not observed in the
                            # training data and use_morph, use camel tools
                            # morphology
                            reinflection_trigger += 1
                            morph_res = self.morph_reinflect(token,
                                                             tag,
                                                             target_gender)

                            reinflected_token = morph_res['reinflected_token']
                            token_candidates = morph_res['proposals']
                            proposal_src = morph_res['proposed_by']

                            candidate_sentence.append(reinflected_token)
                            proposed_by.append(proposal_src)
                            # If there are multiple targets, expand targets
                            if len(token_candidates) != 0:
                                assert reinflected_token == '[MASK]'
                                candidate_targets.append(token_candidates)

                        else:
                            # If the reinflection was not observed in the
                            # training data and we don't want to use morph nor
                            # camel tools pass the token to the output as
                            # it is
                            # global CBR_OOV
                            CBR_OOV += 1
                            candidate_sentence.append(token)
                            proposed_by.append('OOV')

                    # Use morph
                    elif use_morph:
                        import pdb; pdb.set_trace()
                        reinflection_trigger += 1
                        morph_res = self.morph_reinflect(token,
                                                         tag,
                                                         target_gender)
                        reinflected_token = morph_res['reinflected_token']
                        token_candidates = morph_res['proposals']
                        proposal_src = morph_res['proposed_by']

                        candidate_sentence.append(reinflected_token)
                        proposed_by.append(proposal_src)
                        # If there are multiple targets, expand targets
                        if len(token_candidates) != 0:
                            assert reinflected_token == '[MASK]'
                            candidate_targets.append(token_candidates)

            candidates.append(Candidate(masked_sentence=candidate_sentence,
                              targets=candidate_targets,
                              proposed_by=proposed_by))


        for i, candidate in enumerate(candidates):
            if candidate.targets:
                global MORPH_AMBIG
                MORPH_AMBIG += 1

        logger.info(f'CBR triggers: {cbr_trigger}')
        logger.info(f'Morph triggers: {reinflection_trigger}')
        logger.info(f'Regular passes: {regular_passes}')
        logger.info("===========================")
        logger.info(f'Gender Filtering: {GENDER_FILTERING}')
        logger.info(f'Clitic Filtering: {CLITIC_FILTERING}')
        logger.info(f'CBR OOV: {CBR_OOV}')
        logger.info(f'MORPH AMBIG: {MORPH_AMBIG}')
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
                # import pdb; pdb.set_trace()
                scored = self.ranker.fill_and_rank(sentence=sentence,
                                                   targets=candidate.targets)
                logger.info(scored)
                logger.info('=====================')
                reinflections.append(OutputExample(sentence=scored[0].sentence,
                                                   scored_candidates=scored[1:],
                                                   proposed_by=candidate.proposed_by))
            else:
                reinflections.append(OutputExample(sentence=sentence,
                                                   scored_candidates=None,
                                                   proposed_by=candidate.proposed_by))

        return reinflections
