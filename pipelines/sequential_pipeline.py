from ensemble_models.sequential_model import SequentialEnsembleModel
from pipelines.pipeline_super import EnsembleModelPipeline
from plm_models.h5_wrapper import load_wrapped_plms

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SequentialEnsembleModelPipeline(EnsembleModelPipeline):
    def __init__(self, sequential_model: SequentialEnsembleModel,
                 fold,
                 train_dataloader, 
                 validation_dataloader, 
                 test_dataloader, 
                 max_num_runs=100, 
                 early_stopping_window=5,
                 learning_rate=1.0e-4):
        super().__init__(model=sequential_model, 
                        model_name="GB", 
                        fold=fold,
                        train_dataloader=train_dataloader, 
                        validation_dataloader=validation_dataloader,
                        test_dataloader=test_dataloader,
                        max_num_runs=max_num_runs,
                        early_stopping_window=early_stopping_window,
                        learning_rate=learning_rate)

    def run(self, fold):
        plm_counter = 0
        plm_loaded = True

        total_plm_count = len(self.model.plms)
        plm_names = list(self.model.plms.keys())

        while plm_loaded:
            print(f"==> started with {plm_names[plm_counter]} ({plm_counter + 1}/{total_plm_count})")
            self.optimizer = torch.optim.AdamW(self.model.get_current_trainable_parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            # 1. Run training
            self.run_training()
            self.run_test()

            # 2. freeze plm and load next (until all models are trained)
            plm_loaded = self.model.load_next_model()
            plm_counter += 1

        print(f"=====> Finished SM entirely!")

        self.run_test()

        print(f"=====> Finished SE testing fold {fold}")

    def get_targets(self, prot_ids, labels, subset, run_name):
        # Only add frozen terms when training
        if (run_name == "TRAINING") and not self.model.is_training_first_model():
            return labels - self.model.get_frozen_terms(prot_ids, subset)
        
        # Not when evaluating or testing
        return labels.float()
    
    def get_loss_function(self, run_name):
        # First model learns binary classification task
        if self.model.is_training_first_model():
            return nn.BCELoss()
        
        # Validation and test evaluates the binary classification task
        # And the next 2-last models learn to predict with frozen term subtraction
        return nn.MSELoss()
    
    # Method made in assistance with GenAI (Gemini)
    def save_measurements(self, result_counter, run, predictions, targets, loss_func, batch_context):
        """
        Overridden for Sequential Boosting. Logs overall classification metrics 
        by computing the total ensemble prediction against the original binary labels.
        """
        # If it's the first model, it behaves exactly like a standard classification model
        if self.model.is_training_first_model():
            result_counter.add_values(run, predictions, targets, loss_func)
            return

        # --- Sub-model Boosting Path (Model 2+) ---
        # Extract metadata needed to look up the complete ensemble state
        prot_ids = batch_context["prot_ids"]
        subset = batch_context["subset"]
        original_labels = batch_context["original_binary_labels"]

        with torch.no_grad():
            # Fetch the combined prediction across the entire active ensemble cascade
            ensemble_probs = self.model._compute_model_output(prot_ids, subset)

        # Initialize the run inside the targeted result_counter (train, val, or test)
        result_counter.init_run(run)
        if run not in result_counter.errors:
            result_counter.errors[run] = []
            
        # Extract the scalar loss value from the batch execution step
        current_batch_loss = loss_func(predictions, targets).item()

        # Stream values directly into whichever counter object was passed down
        for prob, true_label in zip(ensemble_probs, original_labels):
            prob_val = prob.item() if hasattr(prob, 'item') else float(prob)
            target_val = true_label.item() if hasattr(true_label, 'item') else int(true_label)
            
            result_counter.probabilities[run].append(prob_val)
            result_counter.targets[run].append(target_val)

            # Standard binary metric counting thresholded on the aggregate ensemble
            predicted_target = 1 if prob_val >= 0.5 else 0
            if predicted_target == target_val:
                if predicted_target == 1:
                    result_counter.TP[run] += 1
                else:
                    result_counter.TN[run] += 1
            else:
                if predicted_target == 1:
                    result_counter.FP[run] += 1
                else:
                    result_counter.FN[run] += 1
                    
        # Append the true mathematical residual loss so your loss curves remain accurate
        result_counter.errors[run].append(current_batch_loss)