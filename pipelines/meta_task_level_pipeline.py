from ensemble_models.meta_task_level_model import MetaTaskLevelEnsembleModel
from pipelines.pipeline_super import EnsembleModelPipeline
from plm_models.h5_wrapper import load_wrapped_plms

import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MetaTaskLevelEnsembleModelPipeline(EnsembleModelPipeline):
    def __init__(self, meta_task_level_model: MetaTaskLevelEnsembleModel,
                 fold,
                 train_dataloader, 
                 validation_dataloader, 
                 test_dataloader, 
                 max_num_runs=100, 
                 early_stopping_window=5,
                 learning_rate=1.0e-4):
        super().__init__(model=meta_task_level_model, 
                        model_name="ML", 
                        fold=fold,
                        train_dataloader=train_dataloader, 
                        validation_dataloader=validation_dataloader,
                        test_dataloader=test_dataloader,
                        max_num_runs=max_num_runs,
                        early_stopping_window=early_stopping_window,
                        learning_rate=learning_rate)
    
    def run(self, fold):
        # First phase
        self.optimizer = torch.optim.AdamW(self.model.get_first_phase_parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        print(f"=====> Starting ML first phase!")
        self.run_training()

        # second phase
        self.model.switch_to_second_phase()
        self.optimizer = torch.optim.AdamW(self.model.get_second_phase_parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        print(f"=====> Starting ML second phase!")
        self.run_training()
        
        print(f"=====> Finished ML entirely!")
        self.run_test()
        print(f"=====> Finished ML testing fold {fold}")
