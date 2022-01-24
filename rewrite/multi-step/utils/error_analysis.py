from utils.data_utils import Dataset
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar
)
from collections import Counter

def do_error_analysis(dataset, reinflections,
                      output_dir,
                      speaker_gender='NA',
                      listener_gender='NA'):
    """
    Args:
        - dataset (Dataset object)
        - reinflections (list of OutputExample objects)
        - output_dir (str)
    """
    out_file = open(output_dir, mode='w')
    proposed_by_counts = []
    for i, ex in enumerate(dataset):
        output_example = reinflections[i]

        pred_trg_sentence = output_example.sentence
        proposed_by = ' '.join(output_example.proposed_by)
        scored_candidates = output_example.scored_candidates

        gold_src_sentence = ' '.join(ex.src_tokens)
        gold_src_tags = ' '.join(ex.src_tags)
        gold_trg_sentence = ' '.join(ex.tgt_tokens)
        gold_trg_tags = ' '.join(ex.tgt_tags)
        src_bert_tags = ' '.join(ex.src_bert_tags)

        # We only care about the error analysis if there's a mistake
        # The mistake could be due to four possibilities:
        # 1) tagging error; 2) reinflection proposal error (morph or CBR);
        # 3) reinflection selection error (mlm scorer);
        # 4) normalization
        if pred_trg_sentence != gold_trg_sentence:
            if gold_src_tags != src_bert_tags:
                error_type = 'Tagging'
            else:
                # Check for normalization
                pred_trg_sentence_norm = normalize(pred_trg_sentence)
                gold_trg_sentence_norm = normalize(gold_trg_sentence)
                if pred_trg_sentence_norm == gold_trg_sentence_norm:
                    error_type = 'Normalization'
                else:
                    # Check for selection or proposal errors
                    error_type = 'Proposal'
                    if scored_candidates:
                        for candidate in scored_candidates:
                            if candidate.sentence == gold_trg_sentence:
                                error_type = 'Selection'

            # getting word-level error stats
            if error_type == "Proposal":
                pred_tokens = pred_trg_sentence.split()
                
                proposed_by_counts += [p for i, p
                                       in enumerate(output_example.proposed_by)
                                       if pred_tokens[i] != ex.tgt_tokens[i]]

            out_file.write(f"SRC:\t\t{gold_src_sentence}\n")
            out_file.write(f"TRG:\t\t{gold_trg_sentence}\n")
            out_file.write(f"PRED: \t\t{pred_trg_sentence}\n")
            out_file.write(f"SRC TAGS:\t{gold_src_tags}\n")
            out_file.write(f"TRG TAGS:\t{gold_trg_tags}\n")
            out_file.write(f"PRED SRC TAGS:\t{src_bert_tags}\n")
            out_file.write(f"PROPOSED BY:\t{proposed_by}\n")
            out_file.write(f"ERROR Type:\t{error_type}\n")
            out_file.write(f"SPEAKER TRG GEN:\t{speaker_gender}\n")
            out_file.write(f"LISTENER TRG Type:\t{listener_gender}")
            out_file.write('\n\n')

    out_file.write("==========================\n")
    out_file.write(Counter(proposed_by_counts).__str__())
    out_file.close()

def normalize(sent):
    sent = normalize_alef_ar(sent)
    sent = normalize_alef_maksura_ar(sent)
    sent = normalize_teh_marbuta_ar(sent)
    return sent