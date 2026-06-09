from ensemble_models.representation_fusion_model import RepresentationFusionEnsembleModel
from pipelines.pipeline_super import EnsembleModelPipeline
from plm_models.h5_wrapper import load_wrapped_plms

import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RepresentationFusionEnsembleModelPipeline(EnsembleModelPipeline):
    def __init__(self, representation_fusion_model: RepresentationFusionEnsembleModel,
                 fold,
                 train_dataloader, 
                 validation_dataloader, 
                 test_dataloader, 
                 max_num_runs=100, 
                 early_stopping_window=5,
                 learning_rate=1.0e-4):
        super().__init__(model=representation_fusion_model, 
                        model_name="RF", 
                        fold=fold,
                        train_dataloader=train_dataloader, 
                        validation_dataloader=validation_dataloader,
                        test_dataloader=test_dataloader,
                        max_num_runs=max_num_runs,
                        early_stopping_window=early_stopping_window,
                        learning_rate=learning_rate)
