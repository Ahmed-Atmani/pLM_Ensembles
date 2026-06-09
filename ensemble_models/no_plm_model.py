from ensemble_models.representation_fusion_model import RepresentationFusionEnsembleModel
from ensemble_models.learner_model import learner_factory
from datasets.driver_mutation.drgn_dataset import _load_dataset_file_dict
from plm_models.h5_wrapper import load_wrapped_plms

import torch
import torch.nn.functional as F
import os
import pickle as pkl
import esm


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


train_dict = _load_dataset_file_dict("DRGN")
test_dict = _load_dataset_file_dict("ALL")

def get_prot_data(prot_id, subset):
    if subset == "DRGN":
        dataset = train_dict
    elif subset == "ALL":
        dataset = test_dict
    else:
        raise Exception("Subset not found")

    return dataset[prot_id]


class NoPlmModel(RepresentationFusionEnsembleModel):
    def __init__(self, mut_combination="difference"):
        super().__init__(plms={}, mut_combination=mut_combination, task_type=2, combine_method="append", dropout=0.3)
        
        print("Loading ESM2 tokenizer...")
        _, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D() # Second largest version, only use the tokenizer
        self.batch_converter = self.alphabet.get_batch_converter()
        print("Done loading ESM2 tokenizer!")

    def _get_total_embedding_size(self):
        return 1024
    
    def tokenize_sequences(self, prot_ids, subset):
        # Fetch sequences from dataset
        seqs = []
        mut_seqs = []
        for prot_id in prot_ids:
            prot_data = get_prot_data(prot_id, subset)
            original_seq = prot_data["sequence_cropped"]

            # Compute mut_seq first (same code as compute_embeddings script)
            variant = prot_data["mutation"]
            var_new_aa = variant[-1]
            var_idx = int("".join([ele for ele in variant if ele.isdigit()])) - 1 # -1 because indices in dataset start counting with 1
            crop_offset = prot_data["crop_offset"]
            var_idx -= crop_offset
            mutated_seq = original_seq[:var_idx] + var_new_aa + original_seq[var_idx + 1:] 

            seqs.append(original_seq)
            mut_seqs.append(mutated_seq)

        # Tokenize sequences
        # def tokenize(sequences):
        #     data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        #     _, _, batch_tokens = self.batch_converter(data)
        #     return batch_tokens.to(device)  
        # Tokenize sequences and pad up to the 1024 dataset limit
        def tokenize(sequences, target_len=1024):
            data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
            _, _, batch_tokens = self.batch_converter(data)
            
            # Cast to float for linear layers
            batch_tokens = batch_tokens.float().to(device)
            current_len = batch_tokens.size(1)

            if current_len < target_len:
                pad_size = target_len - current_len
                batch_tokens = F.pad(batch_tokens, (0, pad_size), value=self.alphabet.padding_idx)
                
            return batch_tokens

        t_seqs = tokenize(seqs).float()
        t_mut_seqs = tokenize(mut_seqs).float()

        match self.mut_combination:
            case "append" | "concat":
                combined_embs = torch.cat([t_seqs, t_mut_seqs], dim=1)
            case "difference" | "subtract" | "delta":
                combined_embs = t_seqs - t_mut_seqs
            case "hybrid":
                diffs = t_seqs - t_mut_seqs
                combined_embs = torch.cat([t_seqs, diffs], dim=1)

            case _:
                raise Exception(f"Mutation combination method not found: {self.mut_combination}")
            
        return combined_embs


    def _generate_subresults(self, prot_ids, subset):
        return  [self.tokenize_sequences(prot_ids=prot_ids, subset=subset)]


def get_no_plm_factory(mut_combination="difference"):
    def temp():
        return NoPlmModel(mut_combination=mut_combination)
    return temp