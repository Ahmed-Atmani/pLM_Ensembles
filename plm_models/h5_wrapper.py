import h5py
import torch
import numpy as np

from plm_models.plm_super import PLM


class PLM_Wrapper(PLM):
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.h5_file = h5py.File(path, "r")

    """
    Retrieves the embeddings given the sequence_id (prot-id_variant) and the subset of the dataset ("DRGN" | "ALL" | "DEL" | "NEUT").
    Returns a tuple: (original_sequence_embeddings, mutated_sequence_embeddings)
    """
    def generate_raw_embeddings(self, prot_ids, subset):
        embs = []
        mut_embs = []
        for prot_id in prot_ids:
            embs.append(self.h5_file[self.name][subset][f"{prot_id}_original"][:])
            mut_embs.append(self.h5_file[self.name][subset][f"{prot_id}_mutated"][:])

        # Squeeze redundant dimension and convert potential float16 to float32
        return torch.tensor(np.array(embs)).squeeze(1).float(), torch.tensor(np.array(mut_embs)).squeeze(1).float() 
    

    """ Returns the size of the dimension of the embeddings"""
    def get_embedding_size(self):
        first_key = next(iter(self.h5_file[self.name]["DRGN"]))
        return self.h5_file[self.name]["DRGN"][first_key].shape[-1]

    """
    Retrieves the embeddings given the sequence_id (prot-id_variant) and the subset of the dataset ("DRGN" | "ALL" | "DEL" | "NEUT")
    """
    def __call__(self, prot_ids, subset):
        raw_embs = self.generate_raw_embeddings(prot_ids, subset)
        # embs = normalize(raw_embs) # Already normalized in the files
        return raw_embs

# Stupid relative path dict instead of just automating it, no time to fix
_file_paths = {
    "Ankh3": "datasets/driver_mutation/embeddings/Ankh3_embeddings.h5",
    "AntiBERTy": "datasets/driver_mutation/embeddings/AntiBERTy_embeddings.h5",
    "CARP": "datasets/driver_mutation/embeddings/CARP_embeddings.h5",
    "ESM1b": "datasets/driver_mutation/embeddings/ESM1b_embeddings.h5",
    "ESM1v": "datasets/driver_mutation/embeddings/ESM1v_embeddings.h5",
    "ESM2": "datasets/driver_mutation/embeddings/ESM2_embeddings.h5",
    "ESM2_small": "datasets/driver_mutation/embeddings/ESM2_small_embeddings.h5",
    "ESMC": "datasets/driver_mutation/embeddings/ESMC_embeddings.h5",
    "ESMDance": "datasets/driver_mutation/embeddings/ESMDance_embeddings.h5",
    "ProGen2": "datasets/driver_mutation/embeddings/ProGen2_embeddings.h5",
    "ProtT5": "datasets/driver_mutation/embeddings/ProtT5_embeddings.h5",
    "SeqDance": "datasets/driver_mutation/embeddings/SeqDance_embeddings.h5",
    "UniRep": "datasets/driver_mutation/embeddings/UniRep_embeddings.h5",
}

def load_wrapped_plms(plm_names):
    """ Loads and returns pLMs

    Args:
        plm_names: list of plm names (e.g. ["ESM2", "ProtT5"])

    Returns:
        A dictionary mapping pLM names to their pLM wrapper object
    """
    plms = {}
    for name in plm_names:
        plms[name] = PLM_Wrapper(name, _file_paths[name])
    return plms