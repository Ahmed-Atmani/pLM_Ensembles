# Based on https://github.com/ShenLab/SeqDance/blob/main/notebook/zero_shot_mutation.ipynb


import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Import ESMDance/SeqDance wrapper library
import sys
sys.path.insert(0, "./plm_models/esmdance_lib/model")
from model import ESMwrap

from plm_models.plm_super import PLM



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ESM_Dynamics_family_PLM(PLM):

    def __init__(self, model_version: str, small=False):
        self.model_version = model_version.replace(" ", "").replace("-", "").lower()

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        esm2_select = 'model_35M'

        dance_model = ESMwrap(esm2_select, self.model_version)

        match self.model_version:

            case "esmdance":
                model = dance_model.from_pretrained("ChaoHou/ESMDance")

            case "seqdance":
                model = dance_model.from_pretrained("ChaoHou/SeqDance")

            case _:
                raise Exception(f'ESM (Dynamics) model version "{model_version}" not found')

        model = model.to(device)
        model.eval()
        super().__init__(model_version, model)


    def generate_raw_embeddings(self, sequences, per_residue=False):

        if isinstance(sequences, str):
            sequences = [sequences]

        input = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            output_dict = self.model(input, return_res_pred=True, return_pair_pred=False)

            embs = []
            for key, data in output_dict.items():
                if per_residue:
                    embs.append(data)
                else:
                    # Pooling, otherwise embeddings too large
                    pooled = data.mean(dim=1, keepdim=True).reshape(data.size(0), -1)
                    embs.append(pooled)

            # All tensors are now (1, F_x), so concatenation works perfectly
            return torch.cat(embs, dim=-1)


    def get_embedding_size(self):
        return 325 # Hard coded because it always stays the same, regardless of ESM-2's version used


class ESMDance_PLM(ESM_Dynamics_family_PLM):
    def __init__(self, small=False):
        super().__init__(model_version="ESMDance", small=small)

class SeqDance_PLM(ESM_Dynamics_family_PLM):
    def __init__(self, small=False):
        super().__init__(model_version="SeqDance", small=small)