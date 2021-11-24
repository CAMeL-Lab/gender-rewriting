from utils.data_utils import Dataset
from cbr import CBR
from reinflector import GenderReinflector
from ranker import Ranker
from utils.error_analysis import do_error_analysis
import argparse
from argparse import Namespace
import os

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
        help="Whether to do first person or multi-user reinflection."
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
        help="Whether to do MLE reinflection or not"
    )
    parser.add_argument(
        "--use_morph",
        action="store_true",
        help="Whether to use the morphological analyzer and reinflector."
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
        required=True,
        help="Path to the anaylzer and reinflector database."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Reinflections output dir."
    )
    parser.add_argument(
        "--error_analysis_dir",
        type=str,
        default=None,
        required=True,
        help="error analysis output dir."
    )

    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    # We will repeat the reinflection process across the various target genders
    user_genders = (['M', 'F'] if args.first_person_only else
                    ['MM', 'FM', 'MF', 'FF'])

    # Creating a ranker
    ranker = Ranker(model_name=args.bert_model,
                    use_gpu=args.use_gpu)

    for target_gender in user_genders:
        # Reading training data
        train_dataset = Dataset(src_path=os.path.join(args.data_dir,
                                           'train.arin.tokens'),
                                tgt_path=os.path.join(args.data_dir,
                                           'train.ar.'+target_gender+'.tokens'))

        print(f'There are {len(train_dataset)} Training Examples')

        if args.inference_mode == "dev":
            # Reading dev data
            dev_dataset = Dataset(src_path=os.path.join(args.data_dir,
                                             'dev.arin.tokens'),
                                  tgt_path=os.path.join(args.data_dir,
                                             'dev.ar.'+target_gender+'.tokens'),
                                  src_bert_tags_path=args.src_bert_tags_dir)

            print(f'There are {len(dev_dataset)} Dev Examples')

        elif args.inference_mode == "test":
            # Reading test data
            test_dataset = Dataset(src_path=os.path.join(args.data_dir,
                                            'test.arin.tokens'),
                                   tgt_path=os.path.join(args.data_dir,
                                            'test.ar.'+target_gender+'.tokens'),
                                   src_bert_tags_path=args.src_bert_tags_dir)

            print(f'There are {len(test_dataset)} Test Examples')

        # import pdb; pdb.set_trace()
        if args.use_cbr:
            print(f'Training CBR model')
            cbr_model = CBR.build_model(train_dataset,
                                        ngrams=args.cbr_ngram,
                                        backoff=args.cbr_backoff)
        else:
            cbr_model = None

        # Creating a reinflector
        reinflector = GenderReinflector(cbr_model=cbr_model,
                                       morph_database=args.morph_db,
                                       ranker=ranker,
                                       first_person_only=args.first_person_only)


        speaker_gender = target_gender[0]
        listener_gender = None if args.first_person_only else target_gender[1]

        if args.inference_mode == "dev":
            candidates = reinflector.reinflect(dataset=dev_dataset,
                                               speaker_gender=speaker_gender,
                                               listener_gender=listener_gender,
                                               use_cbr=args.use_cbr,
                                               pick_top_mle=args.pick_top_mle,
                                               use_morph=args.use_morph)

        elif args.inference_mode == "test":
            candidates = reinflector.reinflect(dataset=test_dataset,
                                               speaker_gender=speaker_gender,
                                               listener_gender=listener_gender,
                                               use_cbr=args.use_cbr,
                                               pick_top_mle=args.pick_top_mle,
                                               use_morph=args.use_morph)

        # Final selection
        scored_candidates = reinflector.select(candidates)

        # Writing predictions
        write_predictions(output_file_dir=os.path.join(args.output_dir,
                                          'arin.to.'+target_gender+'.preds'),
                          output_examples=scored_candidates)

        # Auto error analysis
        if args.inference_mode == "dev":
            do_error_analysis(dataset=dev_dataset,
                              reinflections=scored_candidates,
                              output_dir=os.path.join(args.error_analysis_dir,
                                           'arin.to.'+target_gender+'.errors'),
                              speaker_gender=speaker_gender,
                              listener_gender=listener_gender)

        elif args.inference_mode == "test":
            do_error_analysis(dataset=test_dataset,
                              reinflections=scored_candidates,
                              output_dir=os.path.join(args.error_analysis_dir,
                                           'arin.to.'+target_gender+'.errors'),
                              speaker_gender=speaker_gender,
                              listener_gender=listener_gender)

if __name__ == "__main__":
    main()
