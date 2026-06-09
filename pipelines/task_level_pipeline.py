from ensemble_models.task_level_model import TaskLevelEnsembleModel
from pipelines.pipeline_super import EnsembleModelPipeline

import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TaskLevelEnsembleModelPipeline(EnsembleModelPipeline):
    def __init__(self, task_level_model: TaskLevelEnsembleModel,
                 fold,
                 train_dataloader, 
                 validation_dataloader, 
                 test_dataloader, 
                 max_num_runs=100, 
                 early_stopping_window=5,
                 learning_rate=1.0e-4):
        super().__init__(model=task_level_model, 
                        model_name="TL", 
                        fold=fold,
                        train_dataloader=train_dataloader, 
                        validation_dataloader=validation_dataloader,
                        test_dataloader=test_dataloader,
                        max_num_runs=max_num_runs,
                        early_stopping_window=early_stopping_window,
                        learning_rate=learning_rate)
        