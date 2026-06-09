# Based on: https://github.com/facebookresearch/esm (ESM-1x/2) and https://github.com/evolutionaryscale/esm (ESM C)

import torch
import sys
import os

# Point to the parent directory of the renamed folder
# sys.path.append(os.path.abspath("./esmc_lib"))
sys.path.append(os.path.abspath("./plm_models/esmc_lib"))

# Import older facebook ESM and newer EvolutionaryScale ESM packages
import esm
import esmc
from esmc.models.esmc import ESMC
from esmc.sdk.api import ESMProtein, LogitsConfig

from plm_models.plm_super import PLM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ESM_family_PLM(PLM):

    def __init__(self, model_version: str, small=False):
        self.model_version = model_version.replace(" ", "").replace("-", "").lower()

        match self.model_version:

            case "esm1b":
                model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
                self.batch_converter = self.alphabet.get_batch_converter()

            case "esm1v":
                model, self.alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
                self.batch_converter = self.alphabet.get_batch_converter()

            case "esm2":
                if small:        
                    model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                else:
                    model, self.alphabet = esm.pretrained.esm2_t48_15B_UR50D() # Largest version (to be used on Hydra)
                    # model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D() # Second largest version

                self.batch_converter = self.alphabet.get_batch_converter()

            case "esmc":
                # Large: "esmc_600m", "esmc_6b"
                if small:
                    model = ESMC.from_pretrained("esmc_300m")
                else:
                    model = ESMC.from_pretrained("esmc_600m")
                    
                self.alphabet = None
                self.batch_converter = None

            case _:
                raise Exception(f'ESM model version "{model_version}" not found')

        model = model.to(device)
        model.eval()
        super().__init__(model_version, model)


    def generate_raw_embeddings(self, sequences, per_residue=False):

        if isinstance(sequences, str):
            sequences = [sequences]

        if self.model_version == "esmc":
            return self._generate_embeddings_esmc(sequences, per_residue)
        else:
            return self._generate_embeddings_esm2(sequences, per_residue)


    def _generate_embeddings_esm2(self, sequences, per_residue):

        data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.model.num_layers],
                return_contacts=False
            )

        token_representations = results["representations"][self.model.num_layers]
        per_residue_embeddings = token_representations[:, 1:-1, :]

        if per_residue:
            return per_residue_embeddings

        mask = (batch_tokens != self.alphabet.padding_idx)[:, 1:-1].unsqueeze(-1)
        seq_emb = (per_residue_embeddings * mask).sum(dim=1) / mask.sum(dim=1)

        return seq_emb


    def _generate_embeddings_esmc(self, sequences, per_residue):

        embeddings = []

        with torch.no_grad():
            for seq in sequences:

                protein = ESMProtein(sequence=seq)
                protein_tensor = self.model.encode(protein)

                output = self.model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )

                emb = output.embeddings.squeeze(0)  # (L, D)

                if per_residue:
                    embeddings.append(emb)
                else:
                    embeddings.append(emb.mean(dim=0))

        return torch.stack(embeddings)


    def get_embedding_size(self):

        match self.model_version:
            case "esmc":
                return self.model.embed.weight.shape[1]
            
            case "esm1v" | "esm1b":
                return self.model.embed_tokens.weight.shape[1]

            case _:
                return self.model.embed_dim


class ESM1b_PLM(ESM_family_PLM):
    def __init__(self, small=False):
        super().__init__(model_version="ESM-1b", small=small)

class ESM1v_PLM(ESM_family_PLM):
    def __init__(self, small=False):
        super().__init__(model_version="ESM-1v", small=small)

class ESM2_PLM(ESM_family_PLM):
    def __init__(self, small=False):
        super().__init__(model_version="ESM-2", small=small)

class ESMC_PLM(ESM_family_PLM):
    def __init__(self, small=False):
        super().__init__(model_version="ESM C", small=small)