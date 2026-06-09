# Based on: https://github.com/facebookresearch/esm (ESM-1x/2) and https://github.com/evolutionaryscale/esm (ESM C)

import torch
import esm

from plm_models.plm_super import PLM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ESM2_PLM(PLM):

    def __init__(self):
        model, self.alphabet = esm.pretrained.esm2_t48_15B_UR50D() # Largest version (to be used on Hydra)
        model = model.to(device).half()
        model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        super().__init__("ESM2", model)


    def generate_raw_embeddings(self, sequences, per_residue=False):

        if isinstance(sequences, str):
            sequences = [sequences]

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

    def get_embedding_size(self):
        return self.model.embed_dim