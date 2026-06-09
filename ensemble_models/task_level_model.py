from ensemble_models.ensemble_model import EnsembleModel
from ensemble_models.learner_model import learner_factory
from plm_models.model_loader import load_plms
from plm_models.h5_wrapper import load_wrapped_plms

import torch
import os
import pickle as pkl
from collections import Counter


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TaskLevelEnsembleModel(EnsembleModel):
    """
    Parameters:
        - task_type: type of the downstream task ("regression" | <integer> for classification)
        - combine_method: method of choosing the final output ("vote", "avg")
        - plms: dict of plms (plm_name->plm) or dict of h5 plm wrappers (plm_name->wrapper)
    """
    def __init__(self, plms, mut_combination="append", task_type=2, combine_method="avg", dropout=0.3):
        super().__init__(name="TL", plms=plms, mut_combination=mut_combination, task_type=task_type)
        self.combine_method = combine_method
        self.task_type = task_type
        self.task_models = [learner_factory(task_type, plm.get_embedding_size(), mut_combination, dropout=dropout) for plm in self.plms.values()]

    
    def _combine_outputs(self, outputs):
        match self.combine_method:
            case "avg":
                return outputs.mean(dim=0)
            case "vote":
                # return Counter(outputs).most_common()[0][0]
                raise Exception("Not implemented 'vote' combination for the task-level ensemble model.")

    def _get_output(self, prot_ids, subset):
        model_outputs = []

        for plm, task_model in zip(self.plms.values(), self.task_models):
            # Compute and append embeddings of seq and mut_seq
            combined_embs = self._compute_plm_embeddings(plm_name=plm.name, prot_ids=prot_ids, subset=subset)

            # Pass embeddings to task model
            task_output = task_model(combined_embs)

            model_outputs.append(task_output)
            
        outputs = torch.stack(model_outputs, dim=0).squeeze(-1) # Make sure that shape is (batch_size, #plms)
        
        return outputs

    def get_parameters(self):
        parameters = []
        for model in self.task_models:
            for param in model.parameters():
                parameters.append(param)
        return parameters
    
    def _compute_model_output(self, prot_ids, subset):
        outputs = self._get_output(prot_ids, subset)
        combined = self._combine_outputs(outputs)
        return combined

    def train(self):
        for model in self.task_models:
            model.train()

    def eval(self):
        for model in self.task_models:
            model.eval()


    def state_dict(self):
        state = {
            "task_models": [mlp.state_dict() for mlp in self.task_models]
        }
        return state

    def load_state_dict(self, state_dict):
        for mlp, weights in zip(self.task_models, state_dict["task_models"]):
            mlp.load_state_dict(weights)

    def save_model(self, path):

        # Create folder if nonexistant
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save metadata
        metadata = {}
        metadata["plm_names"] = [plm.name for plm in self.plms.values()]
        metadata["combine_method"] = self.combine_method
        metadata["mut_combination"] = self.mut_combination
        metadata["task_type"] = self.task_type
        metadata_file_name = os.path.join(path, "TL.metadata")

        with open(metadata_file_name, "wb") as meta_file:
            pkl.dump(metadata, meta_file)

        # Save the task learners
        for id, mlp in enumerate(self.task_models):    
            mlp_file_name = os.path.join(path, f"MLP{id}.pth")
            torch.save(mlp, mlp_file_name)


def load_TL_model(path, use_wrapper=True):
    metadata_file_name = os.path.join(path, "TL.metadata")
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
    model = TaskLevelEnsembleModel(plms=loaded_plms, task_type=task_type, combine_method=combine_method, mut_combination=mut_combination)

    # Load the task learners
    count = len(plm_names)
    task_models = []
    for id in range(count):
        mlp_file_name = os.path.join(path, f"MLP{id}.pth")
        mlp = torch.load(mlp_file_name, weights_only=False)
        task_models.append(mlp)
    
    model.task_models = task_models
    return model

def get_task_level_factory(plm_names, mut_combination="append", task_type=2, dropout=0.3):
    def temp():
        return TaskLevelEnsembleModel(  plms=load_wrapped_plms(plm_names), 
                                        mut_combination=mut_combination, 
                                        task_type=task_type,
                                        dropout=dropout)
    return temp