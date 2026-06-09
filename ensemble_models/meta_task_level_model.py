from ensemble_models.task_level_model import TaskLevelEnsembleModel
from ensemble_models.learner_model import learner_factory
from plm_models.model_loader import load_plms
from plm_models.h5_wrapper import load_wrapped_plms

import torch
import os
import pickle as pkl
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MetaTaskLevelEnsembleModel(TaskLevelEnsembleModel):
    """
    Parameters:
        - plms: dict of plms (plm_name->plm) or dict of h5 plm wrappers (plm_name->wrapper)
        - task_type: type of the downstream task ("regression" | <integer> for classification)
    """
    def __init__(self, plms, mut_combination="append", task_type=2, dropout=0.3, meta_dropout=0.1):
        super().__init__(plms=plms, mut_combination=mut_combination, task_type=task_type, dropout=dropout)
        self.name = "ML"
        self.meta_learner = learner_factory(task_type=task_type, 
                                            num_inputs=len(plms), 
                                            mut_combination="difference", # Always "difference" because output of task learner always 1 (so no doubling)
                                            num_hidden=1, 
                                            layer_size=32, 
                                            dropout=meta_dropout)
        self.is_first_phase = True
    
    # def get_first_phase_output(self, prot_ids, subset):
    #     return super()(prot_ids, subset)
    
    def get_first_phase_parameters(self):
        return super().get_parameters()
    
    def switch_to_second_phase(self):
        super().eval()
        self.meta_learner.train()
        self.is_first_phase = False
    
    def get_second_phase_parameters(self):
        return self.meta_learner.parameters()  
     
    # def get_second_phase_output(self, prot_ids, subset):
    #     outputs = self._get_output(prot_ids, subset)
    #     combined = self._combine_outputs(outputs)
    #     return combined
    
    def __call__(self, prot_ids, subset):
        outputs = super()._get_output(prot_ids, subset)
        if self.is_first_phase:
            return super()._combine_outputs(outputs)
        return self._combine_outputs(outputs)

    def _combine_outputs(self, outputs):
        outputs = outputs.t() # Transpose to get [batch_len, #plms]
        return self.meta_learner(outputs).squeeze(-1) # Squeeze to get shape [batch_size]

    def get_parameters(self):
        parameters = super().get_parameters()
        for param in self.meta_learner.parameters():
            parameters.append(param)
        return parameters
    
    def train(self):
        # Modify learner model
        super().train()

        # Modify meta-learner
        self.meta_learner.train()

    def eval(self):
        # Modify learner model
        super().eval()

        # Modify meta-learner
        self.meta_learner.eval()

    def state_dict(self):
        state = super().state_dict()
        state["meta_learner"] = self.meta_learner.state_dict()
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.meta_learner.load_state_dict(state_dict["meta_learner"])

    def save_model(self, path):

        # Create folder if nonexistant
        if not os.path.exists(path):
            os.makedirs(path)

        # Save metadata
        metadata = {}
        metadata["plm_names"] = [plm.name for plm in self.plms.values()]
        metadata["mut_combination"] = self.mut_combination
        metadata["task_type"] = self.task_type
        metadata["is_first_phase"] = self.is_first_phase
        metadata_file_name = os.path.join(path, "ML.metadata")

        with open(metadata_file_name, "wb") as meta_file:
            pkl.dump(metadata, meta_file)

        # Save the intermediate task learners
        for id, mlp in enumerate(self.task_models):    
            mlp_file_name = os.path.join(path, f"MLP{id}.pth")
            torch.save(mlp, mlp_file_name)

        # Save the final task learner
        final_mlp_file_name = os.path.join(path, f"MLP_final.pth")
        torch.save(self.meta_learner, final_mlp_file_name)


def load_ML_model(path, use_wrapper=True):
    metadata_file_name = os.path.join(path, "ML.metadata")
    with open(metadata_file_name, "rb") as meta_file:
        metadata = pkl.load(meta_file)

    # Load ensemble model class    
    plm_names = metadata["plm_names"]
    task_type = metadata["task_type"]
    is_first_phase = metadata["is_first_phase"]
    mut_combination = metadata["mut_combination"]
    if use_wrapper:
        loaded_plms = load_wrapped_plms(plm_names)
    else:
        loaded_plms = load_plms(plm_names)
    model = MetaTaskLevelEnsembleModel(plms=loaded_plms, task_type=task_type, mut_combination=mut_combination)
    model.is_first_phase = is_first_phase

    # Load the intermediate task learners
    count = len(plm_names)
    task_models = []
    for id in range(count):
        mlp_file_name = os.path.join(path, f"MLP{id}.pth")
        mlp = torch.load(mlp_file_name, weights_only=False)
        if is_first_phase:
            mlp.eval()
        else:
            mlp.train()
        task_models.append(mlp)
    
    model.task_models = task_models

    # Load final task learners
    final_mlp_file_name = os.path.join(path, f"MLP_final.pth")
    final_mlp = torch.load(final_mlp_file_name, weights_only=False)
    model.meta_learner = final_mlp

    return model

def get_meta_task_factory(plm_names, mut_combination="append", task_type=2, dropout=0.3, meta_dropout=0.1):
    def temp():
        return MetaTaskLevelEnsembleModel(  plms=load_wrapped_plms(plm_names), 
                                            mut_combination=mut_combination, 
                                            task_type=task_type,
                                            dropout=dropout,
                                            meta_dropout=meta_dropout)
    return temp