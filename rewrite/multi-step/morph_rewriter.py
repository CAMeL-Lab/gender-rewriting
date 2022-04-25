from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.generator import Generator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.utils.dediac import dediac_ar
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MorphRewriter:
    def __init__(self, morph_database):
        self.db = MorphologyDB(morph_database, flags='r')
        self.analyzer = Analyzer(self.db)
        self.generator = Generator(self.db)

    def rewrite(self, token, tag, target_gender):
        """
        Uses morphology to do the gender rewriting

        Args:
            - token (str): the src token.
            - tag (str): the predicted bert src token tag.
            - target gender (str): the target gender.

        Returns:
            - rewritten_token, proposals, proposed by (tuple)
        """

        IGNORE =['diac', 'stem', 'stempos', 'stemgloss',
         'stemcat', 'caphi', 'gloss', 'bw', 'catib6', 'ud', 'root', 'pattern',
         'form_gen', 'form_num', 'rat', 'source', 'd1seg', 'd2seg', 'd3seg',
         'atbseg','atbtok','bwtok', 'pos_logprob', 'lex_logprob', 'd1tok',
         'd2tok','pos_lex_logprob', 'd3tok']

        base_word_gender, clitic_gender = tag.split('+')
        target_word_gender, target_clitic_gender = target_gender.split('+')

        # Get the analyses of the token using the analyzer
        analyses = self.analyzer.analyze(token)

        if not analyses:
            logger.info(f'No analyses found for {token}')
            return None

        # get the analyses we care about
        filtered_analyses = []
        if base_word_gender != 'B':
            filtered_analyses += [ana for ana in analyses
                                  if base_word_gender[1].lower() == ana['gen']]

        if clitic_gender != 'B':
            filtered_analyses += [ana for ana in analyses
                                  if clitic_gender[1].lower() in ana['enc0']]

        if not filtered_analyses:
            logger.info(f'No analyses match the filtering criteria {tag}'
                        f' for {token}')
            return None

        # check if the analyses only include 3rd person
        only_third_per = True if len([ana for ana in filtered_analyses
                                     if ana['per'] == '1' or ana['per'] == '2'
                                     or ana['per'] == 'na']) == 0 else False

        # try to rewrite all analyses
        gender_alts = []

        for ana in filtered_analyses:
            for x in IGNORE:
                if x in ana:
                    del ana[x]

            if target_word_gender != 'B' and 'gen' in ana:
                ana['gen'] = target_word_gender[1].lower()

            if target_clitic_gender != 'B' and 'enc0' in ana:
                ana['enc0'] = ana['enc0'].replace(clitic_gender[1].lower(),
                                                  target_clitic_gender[1].lower())


            # we would only run the generator under the following conditions:
            # 1) If the filtered analyses include 1st/2nd/3rd per verbs,
            #    we run the geneator only on 1st and 2nd per verbs
            # 2) If the filtered analyses *only* include 3rd person verbs
            # 3) If the analysis doesn't contain person info (e.g., noun)
            if ana['per'] != '3' or only_third_per or ana['per'] == 'na':
                # delete mod to get the any option in the generator
                if 'mod' in ana: del ana['mod']
                gen_analyses = self.generator.generate(lemma=ana['lex'],
                                                       feats=ana)


                gender_alts += [dediac_ar(re['diac']) for re in gen_analyses]

        gender_alts = list(set(gender_alts))

        if len(gender_alts) == 0:
            logger.info(f'No alternatives found for {token} with tag {tag}'
                        f' and target gender {target_gender}')

            return None

        elif len(gender_alts) == 1:
            return {'rewritten_token': gender_alts[0],
                    'proposals': [],
                    'proposed_by': 'MorphR'}


        elif len(gender_alts) > 1:
            return {'rewritten_token': '[MASK]',
                    'proposals': gender_alts,
                    'proposed_by': 'MorphR'}

