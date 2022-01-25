from utils.data_utils import Dataset
from cbr import CBR
from rbr import RBR
from morph_rewriter import MorphRewriter
from neural_rewriter.rewriter import NeuralRewriter
# from reinflector_union import GenderReinflector
from rewriter import GenderRewriter
from ranker import Ranker
from utils.error_analysis import do_error_analysis
import argparse
from argparse import Namespace
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def write_predictions(output_file_dir, output_examples):
    """
    Args:
        - output_file_dir (str)
        - output_examples (list of OutputExample objects)
    """
    with open(output_file_dir, mode='w') as f:
        for ex in output_examples:
            f.write(ex.sentence)
            f.write('\n')
        f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The data dir. Should contain the src and trg files."
    )
    parser.add_argument(
        "--src_bert_tags_dir",
        default=None,
        type=str,
        required=True,
        help="The predicted bert tags for the src tokens for dev or test sets."
    )
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained BERT model. Note: The model dir name must "
             "start with 'bert'."
    )
    parser.add_argument(
        "--first_person_only",
        action="store_true",
        help="Whether to do first person or multi-user rewriting."
    )
    parser.add_argument(
        "--use_cbr",
        action="store_true",
        help="Whether to use the CBR model."
    )
    parser.add_argument(
        "--cbr_ngram",
        default=2,
        type=int,
        help="CBR ngrams. Defaults to 2."
    )
    parser.add_argument(
        "--cbr_backoff",
        action="store_true",
        help="Whether to use CBR backoff or not during inference."
    )
    parser.add_argument(
        "--pick_top_mle",
        action="store_true",
        help="Whether to do MLE rewriting or not for CBR."
    )
    parser.add_argument(
        "--reduce_cbr_noise",
        action="store_true",
        help="Whether to ignore cbr rewrite if its the same as the input."
    )
    parser.add_argument(
        "--use_morph",
        action="store_true",
        help="Whether to use MorphR or not."
    )
    parser.add_argument(
        "--use_rbr",
        action="store_true",
        help="Whether to use the RBR model."
    )
    parser.add_argument(
        "--rbr_top_rule",
        action="store_true",
        help="Whether to use the top rule in the RBR model."
    )
    parser.add_argument(
        "--rbr_top_tgt_rule",
        action="store_true",
        help="Whether to use the top target rule in the RBR model."
    )
    parser.add_argument(
        "--use_seq2seq",
        action="store_true",
        help="Whether to use NeuralR or not."
    )
    parser.add_argument(
        "--top_n_best",
        type=int,
        help="Top n decoded sequences from the seq2seq model."
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        help="Beam width of the seq2seq model."
    )
    parser.add_argument(
        "--seq2seq_model_path",
        type=str,
        help="seq2seq pretrained model path."
    )
    parser.add_argument(
        "--use_data_augmentation",
        action="store_true",
        help="To use the augmented data for training or not."
    )
    parser.add_argument(
        "--inference_mode",
        default="dev",
        required=True,
        type=str,
        help="Whether to do inference on the dev or test sets."
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Whether to use a GPU or not."
    )
    parser.add_argument(
        "--morph_db",
        type=str,
        default=None,
        help="Path to the anaylzer and generator database."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Generated sentences output dir."
    )
    parser.add_argument(
        "--analyze_errors",
        action="store_true",
        help="To do error analysis on the outputs or not."
    )
    parser.add_argument(
        "--error_analysis_dir",
        type=str,
        default=None,
        help="error analysis output dir."
    )

    args = parser.parse_args()
    logger.info(args)

    # We will repeat the rewriting process across the various target genders
    user_genders = (['M', 'F'] if args.first_person_only else
                    ['MM', 'FM', 'MF', 'FF'])
                    # ['FF'])

    # Creating a ranker
    ranker = Ranker(model_name=args.bert_model,
                    use_gpu=args.use_gpu)

    for target_gender in user_genders:
        logger.info('\n')
        logger.info(f'######## {target_gender} Rewriting ########')
        logger.info('\n')
        # Reading training data
        # TODO: fix tokens files to remove the empty lines from the end
        # This will take care of fixing the empty line issue at the end 
        # of the preds files

        if args.use_data_augmentation:
            train_dataset = Dataset(src_path=os.path.join(args.data_dir,
                                            'augmented_data',
                                            'train.arin.tokens.augmented'),
                                    tgt_path=os.path.join(args.data_dir,
                                            'augmented_data',
                                            'CBR_filter_2+backoff+all+morph_newdb+mod_per_3rd_generator+neural_fix_augmented',
                                            'train.ar.'+target_gender+'.tokens.augmented.new'))
        else:
            train_dataset = Dataset(src_path=os.path.join(args.data_dir,
                                            'train.arin.tokens'),
                                    tgt_path=os.path.join(args.data_dir,
                                            'train.ar.'+target_gender+'.tokens'))

        logger.info(f'There are {len(train_dataset)} Training Examples')

        if args.inference_mode == "dev":
            # Reading dev data
            dev_dataset = Dataset(src_path=os.path.join(args.data_dir,
                                                        'dev.arin.tokens'),
                                  tgt_path=os.path.join(args.data_dir,
                                             'dev.ar.'+target_gender+'.tokens'),
                                  src_bert_tags_path=args.src_bert_tags_dir)

            # dev_dataset = Dataset(src_path=os.path.join(args.data_dir,
            #                                             'google_MT/dev.google.ar.tokens'),
            #                       src_bert_tags_path=args.src_bert_tags_dir)

            logger.info(f'There are {len(dev_dataset)} Dev Examples')

        elif args.inference_mode == "test":
            # Reading test data
            test_dataset = Dataset(src_path=os.path.join(args.data_dir,
                                                         'test.arin.tokens'),
                                   tgt_path=os.path.join(args.data_dir,
                                             'test.ar.'+target_gender+'.tokens'),
                                   src_bert_tags_path=args.src_bert_tags_dir)

            # test_dataset = Dataset(src_path=os.path.join(args.data_dir,
            #                                              'google_MT/test.google.ar.tokens'),
            #                        src_bert_tags_path=args.src_bert_tags_dir)


            # test_dataset = Dataset(src_path='/scratch/ba63/gender-rewriting/raw_openSub/augmentation/test.txt',
            #                        src_bert_tags_path=args.src_bert_tags_dir)


            logger.info(f'There are {len(test_dataset)} Test Examples')

        if args.use_cbr:
            logger.info(f'Training CBR model for {target_gender} target')
            cbr_model = CBR.build_model(train_dataset,
                                        ngrams=args.cbr_ngram,
                                        backoff=args.cbr_backoff)
            # import pdb; pdb.set_trace()
        else:
            cbr_model = None

        if args.use_rbr:
            logger.info(f'Training RBR model for {target_gender} target')

            rbr_model = RBR.build_model(train_dataset,
                                        pick_top_rule=args.rbr_top_rule,
                                        pick_top_tgt_rule=args.rbr_top_tgt_rule)
        else:
            rbr_model = None

        if args.use_morph:
            morphR = MorphRewriter(args.morph_db)
        else:
            morphR = None

        if args.use_seq2seq:
            logger.info(f'Loading the pretrained neural rewriter model')
            neuralR = NeuralRewriter.from_pretrained(model_path=args.seq2seq_model_path,
                                                     top_n_best=args.top_n_best,
                                                     beam_width=args.beam_width)
        else:
            neuralR = None

        # Creating a rewriter
        rewriter = GenderRewriter(cbr_model=cbr_model,
                                  morph_rewriter=morphR,
                                  rbr_model=rbr_model,
                                  neural_rewriter=neuralR,
                                  ranker=ranker,
                                  first_person_only=args.first_person_only)


        speaker_gender = target_gender[0]
        listener_gender = None if args.first_person_only else target_gender[1]


        if args.inference_mode == "dev":
            candidates = rewriter.rewrite(dataset=dev_dataset,
                                        speaker_gender=speaker_gender,
                                        listener_gender=listener_gender,
                                        use_cbr=args.use_cbr,
                                        pick_top_mle=args.pick_top_mle,
                                        reduce_cbr_noise=args.reduce_cbr_noise,
                                        use_morph=args.use_morph,
                                        use_rbr=args.use_rbr,
                                        use_neural=args.use_seq2seq)

        elif args.inference_mode == "test":
            candidates = rewriter.rewrite(dataset=test_dataset,
                                        speaker_gender=speaker_gender,
                                        listener_gender=listener_gender,
                                        use_cbr=args.use_cbr,
                                        pick_top_mle=args.pick_top_mle,
                                        reduce_cbr_noise=args.reduce_cbr_noise,
                                        use_morph=args.use_morph,
                                        use_rbr=args.use_rbr,
                                        use_neural=args.use_seq2seq)

        # Final selection
        scored_candidates = rewriter.select(candidates)

        # Writing predictions
        write_predictions(output_file_dir=os.path.join(args.output_dir,
                                          'arin.to.'+target_gender+'.preds'),
                          output_examples=scored_candidates)

        # # Auto error analysis
        if args.analyze_errors:
            if args.inference_mode == "dev":
                do_error_analysis(dataset=dev_dataset,
                                gender_alts=scored_candidates,
                                output_dir=os.path.join(args.error_analysis_dir,
                                            'arin.to.'+target_gender+'.errors'),
                                speaker_gender=speaker_gender,
                                listener_gender=listener_gender)

            elif args.inference_mode == "test":
                do_error_analysis(dataset=test_dataset,
                                gender_alts=scored_candidates,
                                output_dir=os.path.join(args.error_analysis_dir,
                                            'arin.to.'+target_gender+'.errors'),
                                speaker_gender=speaker_gender,
                                listener_gender=listener_gender)

if __name__ == "__main__":
    main()

