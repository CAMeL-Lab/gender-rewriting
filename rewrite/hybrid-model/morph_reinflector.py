from camel_tools.morphology.reinflector import Reinflector
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.generator import Generator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.utils.dediac import dediac_ar

class MorphReinflector:
    def __init__(self, morph_database):
        self.db = MorphologyDB(morph_database, flags='r')
        self.analyzer = Analyzer(self.db)
        self.reinflector = Reinflector(self.db)
        self.generator = Generator(self.db)

    def reinflect(self, token, tag, target_gender):
        """
        Uses morphology to do the gender rewriting

        Args:
            - token (str): the src token.
            - tag (str): the predicted bert src token tag.
            - target gender (str): the target gender.

        Returns:
            - reinflected_token, proposals, proposed by (tuple)
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
            logger.info(f'No analyses match the filtering criteria {tag}'
                        f'for {token}')
            return {'reinflected_token':token,
                    'proposals': [],
                    'proposed_by': 'Analyzer_OOV'}

        # check if the analyses only include 3rd person
        only_third_per = True if len([ana for ana in filtered_analyses
                                     if ana['per'] == '1' or ana['per'] == '2'
                                     or ana['per'] == 'na']) == 0 else False

        # try to reinflect all analyses
        reinflections = []

        for ana in filtered_analyses:
            for x in IGNORE:
                if x in ana:
                    del ana[x]

            # features = {'pos': ana['pos']}

            # in Arabic, only 2nd and 3rd person verbs inflect for gender
            # and we only care about 2nd person in this task.
            # if features['pos'] == 'verb' and (ana['per'] == '1' or ana['per'] == '2'):
            #     features['per'] = ana['per']

            if target_word_gender != 'B' and 'gen' in ana:
                # features['gen'] = target_word_gender[1].lower()
                ana['gen'] = target_word_gender[1].lower()

            if target_clitic_gender != 'B' and 'enc0' in ana:
                # features['enc0'] = ana['enc0'].replace(clitic_gender[1].lower(),
                #                                        target_clitic_gender[1].lower())
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
                reinflection_analyses = self.generator.generate(lemma=ana['lex'],
                                                                feats=ana)

            # reinflection_analyses = self.reinflector.reinflect(word=dediac_ar(ana['diac']),
            #                                                    feats=features)

                reinflections += [dediac_ar(re['diac']) for re in reinflection_analyses]

        reinflections = list(set(reinflections))

        if len(reinflections) == 0:
            logger.info(f'No reinflections found for {token} with tag {tag}'
                         f' and target gender {target_gender}')

            return {'reinflected_token': token,
                    'proposals': [],
                    'proposed_by': 'Reinflector_OOV'}


        elif len(reinflections) == 1:
            return {'reinflected_token': reinflections[0],
                    'proposals': [],
                    'proposed_by': 'Reinflector'}


        elif len(reinflections) > 1:
            return {'reinflected_token': '[MASK]',
                    'proposals': reinflections,
                    'proposed_by': 'Reinflector'}

