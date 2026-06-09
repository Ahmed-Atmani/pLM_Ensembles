# Based on: https://github.com/jeffreyruffolo/AntiBERTy

from antiberty import AntiBERTyRunner
import torch

from plm_models.plm_super import PLM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AntiBERTy_PLM(PLM):
    def __init__(self, small=False):
        model = AntiBERTyRunner()  

        super().__init__("AntiBERTy", model)

    def generate_raw_embeddings(self, sequences, per_residue=False):

        if isinstance(sequences, str):
            sequences = [sequences]
            
        with torch.no_grad():
            per_residue_embeddings = torch.stack(self.model.embed(sequences), dim=0)[:, 1:-1, :]

            if per_residue:
                return per_residue_embeddings
            
            return per_residue_embeddings.mean(dim=1)

    def get_embedding_size(self):
        """ Returns the size of the dimension of the embeddings"""
        return self.model.config.hidden_size