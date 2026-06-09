# Based on: https://github.com/ElArkk/jax-unirep/blob/master/scripts/testdrive_sampling.py
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import torch
from jax_unirep import get_reps
from jax_unirep.evotuning import load_params

from plm_models.plm_super import PLM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class UniRep_PLM(PLM):
    def __init__(self, small=False):

        super().__init__("UniRep", None)

    def generate_raw_embeddings(self, sequences, per_residue=False):

        # If it is a single sequence, put it in a singleton list
        if isinstance(sequences, str):
            sequences = [sequences]

        with torch.no_grad():
            avg_hidden_states, _, _ = get_reps(sequences)
            return torch.tensor(avg_hidden_states).to(device)

    def get_embedding_size(self):
        """ Returns the size of the dimension of the embeddings"""
        return 1900
