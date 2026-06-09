from ensemble_models.ensemble_model import EnsembleModel
from ensemble_models.learner_model import learner_factory
from plm_models.model_loader import load_plms
from plm_models.h5_wrapper import load_wrapped_plms

import torch
import os
import pickle as pkl


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SequentialEnsembleModel(EnsembleModel):
    """
    Subclasses must implement the following methods:
    - _compute_model_output
    - get_parameters
    - save_model

    Parameters:
        - plms: dict of plms (plm_name->plm) or dict of h5 plm wrappers (plm_name->wrapper)
        - task_type: type of the downstream task ("regression" | <integer> for classification)
    """
    def __init__(self, plms, mut_combination="append", task_type=2, dropout=0.3, eta=0.01):
        super().__init__(name="GB", plms=plms, mut_combination=mut_combination, task_type=task_type)
        self.task_type = task_type
        # First task depends on task type
        self.task_models = [learner_factory(task_type, list(plms.values())[0].get_embedding_size(), mut_combination, dropout=dropout)]
        # Other MLPs learn regression
        self.task_models += [learner_factory("regression", plm.get_embedding_size(), mut_combination, dropout=dropout, layer_size=128, num_hidden=1) for plm in list(self.plms.values())[1:]] 

        self.current_plm_index = 0
        self.eta = eta
        

    """
    Returns True if the first model is loaded.
    (important to select the right loss function in the pipeline)
    """
    def is_training_first_model(self):
        return self.current_plm_index == 0

    """
    Freezes the current MLP and loads the next pLM in the sequence; to be called after a training session.
    Returns: True if the next model is loaded; False if there are no models to be loaded anymore.
    """
    def load_next_model(self):
        def freeze_mlp(mlp):
            mlp.eval() # Disable dropout
            for param in mlp.parameters():
                param.requires_grad = False
                
        def unfreeze_mlp(mlp):
            mlp.train() # Enable dropout
            for param in mlp.parameters():
                param.requires_grad = True

        # Freeze current MLP
        freeze_mlp(self.task_models[self.current_plm_index])

        # Increment counter
        self.current_plm_index += 1

        # Check if there are any plms left
        if self.current_plm_index >= len(self.plms.values()):
            return False
            
        # Unfreeze MLP
        unfreeze_mlp(self.task_models[self.current_plm_index])

        return True
    
    """
    Returns the output of the frozen MLPs to be subtracted to the loss function (boosting)
    """
    def get_frozen_terms(self, prot_ids, subset):
        plm_names = list(self.plms.keys())

        # Add to term for each frozen MLP
        total = torch.tensor([0]).to(device)
        for i in range(self.current_plm_index):
            plm_name = plm_names[i]
            mlp = self.task_models[i]
            embeddings = self._compute_plm_embeddings(plm_name, prot_ids, subset=subset)
            with torch.no_grad():
                output = mlp(embeddings)
            total = total + output

        return total.squeeze(-1)
    
    def get_current_trainable_parameters(self):
        return self.task_models[self.current_plm_index].parameters()

    """
    Returns the prediction
    prot_ids: a list of protein ids
    """
    # def _compute_model_output(self, prot_ids, subset):
    #     result = 0
    #     for i, plm in enumerate(self.plms.values()):
    #         embs = self._compute_plm_embeddings(plm.name, prot_ids, subset=subset)
    #         mlp = self.task_models[i]
    #         result += mlp(embs)

    #     return torch.clamp(result, min=0, max=1).squeeze(-1)
    def _compute_model_output(self, prot_ids, subset):
        num_models = min(self.current_plm_index + 1, len(self.task_models)) # To account for last iteration
        result = 0
        for i in range(num_models):
            plm_name = list(self.plms.keys())[i]
            embs = self._compute_plm_embeddings(plm_name, prot_ids, subset=subset)
            mlp = self.task_models[i]
            if i == 0:
                result += mlp(embs)
            else:
                result += self.eta * mlp(embs)

        return result.squeeze(-1)

    """
    Returns the trainable parameters
    """
    def get_parameters(self):
        parameters = []
        for model in self.task_models:
            for param in model.parameters():
                parameters.append(param)
        return parameters

    def train(self):
        # Only set current MLP to train
        mlp = self.task_models[self.current_plm_index]
        mlp.train()
    
    def eval(self):
        # Set all MLPs to eval
        for mlp in self.task_models:
            mlp.eval()

    def state_dict(self):
        state = {
            "task_models": [mlp.state_dict() for mlp in self.task_models],
            "current_plm_index": self.current_plm_index
        }
        return state

    def load_state_dict(self, state_dict):
        for mlp, weights in zip(self.task_models, state_dict["task_models"]):
            mlp.load_state_dict(weights)
        
        # Sync the execution phase tracking variable
        self.current_plm_index = state_dict["current_plm_index"]

    def save_model(self, path):

        # Create folder if nonexistant
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save metadata
        metadata = {}
        metadata["plm_names"] = [plm.name for plm in self.plms.values()]
        metadata["task_type"] = self.task_type
        metadata["current_plm_index"] = self.current_plm_index
        metadata_file_name = os.path.join(path, "SM.metadata")

        with open(metadata_file_name, "wb") as meta_file:
            pkl.dump(metadata, meta_file)

        # Save the task learners
        for id, mlp in enumerate(self.task_models):    
            mlp_file_name = os.path.join(path, f"MLP{id}.pth")
            torch.save(mlp, mlp_file_name)


def load_SM_model(path, use_wrapper=True):
    metadata_file_name = os.path.join(path, "SM.metadata")
    with open(metadata_file_name, "rb") as meta_file:
        metadata = pkl.load(meta_file)

    # Load ensemble model class    
    plm_names = metadata["plm_names"]
    task_type = metadata["task_type"]
    current_plm_index = metadata["current_plm_index"]
    if use_wrapper:
        loaded_plms = load_wrapped_plms(plm_names)
    else:
        loaded_plms = load_plms(plm_names)
    model = SequentialEnsembleModel(plms=loaded_plms, task_type=task_type)

    # Load the task learners
    count = len(plm_names)
    task_models = []
    for id in range(count):
        mlp_file_name = os.path.join(path, f"MLP{id}.pth")
        mlp = torch.load(mlp_file_name)
        task_models.append(mlp)

    # Freeze MLPs that are trained
    def freeze_mlp(mlp):
        for param in mlp.parameters():
            param.requires_grad = False
    for i in range(current_plm_index):
        freeze_mlp(model.task_models[i])
    
    model.task_models = task_models
    return model


def get_sequential_factory(plm_names, mut_combination="append", task_type=2, dropout=0.3):
    def temp():
        return SequentialEnsembleModel(  plms=load_wrapped_plms(plm_names), 
                                        mut_combination=mut_combination, 
                                        task_type=task_type,
                                        dropout=dropout)
    return temp