# Based on: https://github.com/microsoft/protein-sequence-models

import torch
from sequence_models.pretrained import load_model_and_alphabet

from plm_models.plm_super import PLM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CARP_PLM(PLM):
    def __init__(self, small=False):
        model_version = "carp_600k" if small else"carp_640M" 
        model, self.collater = load_model_and_alphabet(model_version)

        super().__init__("CARP", model)

    def generate_raw_embeddings(self, sequences, per_residue=False):

        # If it is a single sequence, put it in a singleton list
        if isinstance(sequences, str):
            sequences = [sequences]

        # # Wrap each sequence into a singleton list (needed for CARP)
        # sequences = [[element] for element in sequences]

        with torch.no_grad():
            x = self.collater(sequences)[0]
            rep = self.model(x)["representations"][56].to(device)

            if per_residue:
                return rep
            
            return rep.mean(dim=1)


    def get_embedding_size(self):
        """ Returns the size of the dimension of the embeddings"""
        return 1280