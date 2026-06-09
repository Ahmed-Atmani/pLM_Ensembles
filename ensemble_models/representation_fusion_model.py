from ensemble_models.ensemble_model import EnsembleModel
from ensemble_models.learner_model import learner_factory
from plm_models.model_loader import load_plms
from plm_models.h5_wrapper import load_wrapped_plms

import torch
import os
import pickle as pkl


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RepresentationFusionEnsembleModel(EnsembleModel):
    """
    Parameters:
        - task_type: type of the downstream task ("regression" | <integer> for classification)
        - combine_method: method of combining the embeddings ("append", "avg")
        - plms: dict of plms (plm_name->plm) or dict of h5 plm wrappers (plm_name->wrapper)
    """
    def __init__(self, plms, mut_combination="append", task_type=2, combine_method="append", dropout=0.3):
        super().__init__(name="RF", plms=plms, mut_combination=mut_combination, task_type=task_type)
        self.combine_method = combine_method
        self.task_model = learner_factory(task_type, self._get_total_embedding_size(), mut_combination, dropout=dropout)

    def _generate_subresults(self, prot_ids, subset):
        per_plm_embeddings = []

        for plm_name in self.plms.keys():
            combined_embs = self._compute_plm_embeddings(plm_name=plm_name, prot_ids=prot_ids, subset=subset)
            per_plm_embeddings.append(combined_embs)
        
        return per_plm_embeddings

    def _combine_subresults(self, subresults):
        match self.combine_method:
            case "append":
                return torch.cat(subresults, dim=1)
            case "avg":
                return subresults.mean(dim=1)  
            
    def get_parameters(self):
        return self.task_model.parameters()

    """
    Returns the combined embeddings of the representation fusion model.
    """
    def _get_fusion_embeddings(self, prot_ids, subset):
        combined_embs = self._generate_subresults(prot_ids, subset)

        combined_embeddings = self._combine_subresults(combined_embs) # Combines all plm embeddings
        return combined_embeddings
    
    def _compute_model_output(self, prot_ids, subset):
        embs = self._get_fusion_embeddings(prot_ids, subset=subset).to(device)
        output = self.task_model(embs)

        return output.squeeze(-1)

    def train(self):
        self.task_model.train()

    def eval(self):
        self.task_model.eval()

    def state_dict(self):
        state = {
            "task_model": self.task_model.state_dict()
        }
        return state

    def load_state_dict(self, state_dict):
        self.task_model.load_state_dict(state_dict["task_model"])

    def save_model(self, path):

        # Create folder if nonexistant
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save metadata
        metadata = {}
        metadata["plm_names"] = [plm.name for plm in self.plms.values()]
        # metadata["plm_sizes"] = [plm.get_embedding_size() for plm in self.plms]
        metadata["combine_method"] = self.combine_method
        metadata["mut_combination"] = self.mut_combination
        metadata["task_type"] = self.task_type
        metadata_file_name = os.path.join(path, "RF.metadata")

        with open(metadata_file_name, "wb") as meta_file:
            pkl.dump(metadata, meta_file)

        # Save task learner
        mlp_file_name = os.path.join(path, "MLP.pth")
        torch.save(self.task_model, mlp_file_name)


def load_RF_model(path, use_wrapper=True):
    metadata_file_name = os.path.join(path, "RF.metadata")
    with open(metadata_file_name, "rb") as meta_file:
        metadata = pkl.load(meta_file)

    # Load ensemble model class    
    plm_names = metadata["plm_names"]
    task_type = metadata["task_type"]
    combine_method = metadata["combine_method"]
    mut_combination = metadata["mut_combination"]
    if use_wrapper:
        loaded_plms = load_wrapped_plms(plm_names)
    else:
        loaded_plms = load_plms(plm_names)
    model = RepresentationFusionEnsembleModel(plms=loaded_plms, task_type=task_type, combine_method=combine_method, mut_combination=mut_combination)

    # Load task learner
    mlp_file_name = os.path.join(path, "MLP.pth")
    mlp = torch.load(mlp_file_name, weights_only=False)
    model.task_model = mlp

    return model


def get_representation_fusion_factory(plm_names, combine_method="append", mut_combination="append", task_type=2, dropout=0.3):
    def temp():
        return RepresentationFusionEnsembleModel(plms=load_wrapped_plms(plm_names), 
                                                     combine_method=combine_method, 
                                                     mut_combination=mut_combination, 
                                                     task_type=task_type,
                                                     dropout=dropout)
    return temp