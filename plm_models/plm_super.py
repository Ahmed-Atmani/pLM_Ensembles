import torch


def normalize(emb_batch):
    """
    Normalizes a batch of embeddings (such that all features have mean 0 stdev 1). 
    To be used after appending all the embeddings.
    """
    mean = emb_batch.mean(1, keepdim=True)
    std = emb_batch.std(1, keepdim=True)

    std = torch.where(torch.isclose(std, torch.zeros_like(std)), 
                      torch.ones_like(std),
                      std)
    return (emb_batch - mean) / std


class PLM:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def generate_raw_embeddings(self, seq, per_residue=False):
        """
        1. Converts the input to the model's appropriate input type. 
        2. Generates the embeddings using the model.
        3. Converts the model's output into a simple tensor.
        """
        pass

    def get_embedding_size(self):
        """ Returns the size of the dimension of the embeddings"""
        pass

    def __call__(self, raw_seqs, per_residue=False):
        """
        raw_seqs: list of sequences
        pool: averages per-residue embeddings over residues
        """
        raw_embs = self.generate_raw_embeddings(raw_seqs, per_residue=per_residue)
        embs = normalize(raw_embs)
        return embs
