from .model.seq2seq import Seq2Seq
from .utils.data_utils import Vectorizer
from .model.beam_decoder import BeamSampler
import json
import torch
import os

class Seq2Seq_Reinflector:
    def __init__(self, model, vectorizer, beam_width, top_n_best):
        self.model = model
        self.vectorizer = vectorizer
        self.beam_decoder = BeamSampler(model=self.model,
                                 src_vocab_char=self.vectorizer.src_vocab_char,
                                 src_vocab_word=self.vectorizer.src_vocab_word,
                                 trg_vocab_char=self.vectorizer.trg_vocab_char,
                                 trg_gender_vocab=self.vectorizer.trg_gender_vocab,
                                 beam_width=beam_width,
                                 topk=top_n_best)


    @classmethod
    def from_pretrained(cls, model_path, beam_width=10, top_n_best=5):
        # loading the model's config
        with open(os.path.join(model_path, 'joint.config.json')) as f:
            model_config = json.load(f)

        model = Seq2Seq(**model_config)
        # TODO: Fix decoding device issue (move data to the right device)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        # loading the model's state dict
        model.load_state_dict(torch.load(os.path.join(model_path, 'joint.pt'),
                                         map_location=device))
        model.eval()
        model = model.to(device)

        # loading the vectorizer
        # with open(os.path.join(model_path,'vectorizer.no_morph.json')) as f:
        with open(os.path.join(model_path,'vectorizer.json')) as f:
            vectorizer = Vectorizer.from_serializable(json.load(f))

        return cls(model, vectorizer, beam_width, top_n_best)

    def reinflect(self, token, target_gender):
        """
        Uses the pretrained char-level seq2seq model to do the gender rewriting

        Args:
            - token (str): the src token.
            - target gender (str): the target gender.

        Returns:
            - reinflected_token, proposals, proposed by (tuple)
        """
        # adding target gender (side constraint) to the token
        token_sc = f'<{target_gender}>{token}'

        reinflections = self.beam_decoder.beam_decode(token=token_sc,
                                                      add_side_constraints=True,
                                                      max_len=512)

        return reinflections