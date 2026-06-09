import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class EnsembleModel:
    """
    Subclasses must implement the following methods:
    - _compute_model_output
    - get_parameters
    - save_model
    - train
    - eval

    Parameters:
        - plms: list of plm models or list of h5 file wrappers
        - mut_combination: method of combining mutation sequence embeddings ["append" | "difference"]
    """
    def __init__(self, name, plms, mut_combination="append", task_type=2):
        self.name = name
        self.plms = plms
        self.mut_combination = mut_combination
        self.task_type = task_type

    """
    Returns the combined (seq + mut_seq) embeddings of a plm
    seqs: a list of sequences of amino acids
    mut_seqs: a list of sequences of amino acids after mutation
    """
    def _compute_plm_embeddings(self, plm_name, prot_ids, subset):
        plm = self.plms[plm_name]
        embs, mut_embs = plm(prot_ids=prot_ids, subset=subset)

        match self.mut_combination:
            case "append" | "concat":
                combined_embs = torch.cat([embs, mut_embs], dim=1)
            case "difference" | "subtract" | "delta":
                combined_embs = embs - mut_embs
            case "hybrid":
                diffs = embs - mut_embs
                combined_embs = torch.cat([embs, diffs], dim=1)

            case _:
                raise Exception(f"Mutation combination method not found: {self.mut_combination}")

        
        return combined_embs.to(device)

    """
    Returns the total embeddings size of the ensemble model
    """
    def _get_total_embedding_size(self):
        cnt = 0
        for plm in self.plms.values():
            cnt += plm.get_embedding_size()
        return cnt

    """
    Returns the prediction
    seqs: a list of sequences of amino acids
    mut_seqs: a list of sequences of amino acids after mutation
    """
    def _compute_model_output(self, prot_ids, subset):
        pass

    """
    Returns the trainable parameters
    """
    def get_parameters(self):
        return torch.tensor([])

    """
    Persists the model
    """
    def save_model(self, path):
        pass

    """
    Sets the trainable models into training mode
    """
    def train(self):
        pass

    """
    Sets the trainable models into evaluation mode
    """
    def eval(self):
        pass

    """
    Returns the current state of the MLPs
    """
    def state_dict(self):
        pass

    """
    Loads current state of the MLPs
    """
    def load_state_dict(self, state_dict):
        pass

    def __call__(self, prot_ids, subset):
        return self._compute_model_output(prot_ids, subset)
    
    