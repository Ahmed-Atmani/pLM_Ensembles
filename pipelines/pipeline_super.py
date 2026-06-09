from ensemble_models.ensemble_model import EnsembleModel
from pipelines.output_counter import BCCounter
from datasets.driver_mutation.drgn_dataset import get_test_set_dataloader, get_train_set_dataset

import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import time
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
import datetime


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class EnsembleModelPipeline:
    def __init__(self, model, model_name, fold, train_dataloader, validation_dataloader, test_dataloader, max_num_runs=40, early_stopping_window=5, learning_rate=1.0e-5, weight_decay=1.0e-3, patience=20):

        # Dataset
        self.train_set = train_dataloader
        self.val_set = validation_dataloader
        self.test_set = test_dataloader

        # Model
        self.model = model
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(self.model.get_parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Early stopping
        self.early_stopping_window = early_stopping_window
        self.patience = patience
        self.max_num_runs = max_num_runs
        self.last_run = 0

        # Logging
        self.train_measurements = BCCounter(ensemble_type="sequential", plm_names=list(self.model.plms.keys()), dataset="train", fold=fold)
        self.validation_measurements = BCCounter(ensemble_type="sequential", plm_names=list(self.model.plms.keys()), dataset="validation", fold=fold)
        self.test_measurements = BCCounter(ensemble_type="sequential", plm_names=list(self.model.plms.keys()), dataset="test", fold=fold)
        
    def run_test(self):
        return self._single_run(dataloader=self.test_set, run_name="TEST", run=1, total_runs=1, result_counter=self.test_measurements, with_grad=False)

    def run_validation(self, run):
        return self._single_run(dataloader=self.val_set, run_name="VALIDATION", run=run, total_runs=1, result_counter=self.validation_measurements, with_grad=False)
    
    def run_training(self):
        run_counter = 0
        early_stopping_counter = 0
        early_stopping_current_best_error = float("+inf")
        best_model = None
        best_model_run = -1
        patience_counter = 0

        while (early_stopping_counter < self.early_stopping_window) and (run_counter < self.max_num_runs):
            train_error = self._single_run(dataloader=self.train_set, run_name="TRAINING", run=run_counter, total_runs=self.max_num_runs, result_counter=self.train_measurements, with_grad=True)
            val_score = self.run_validation(run_counter)

            # Patience
            if patience_counter < self.patience:
                patience_counter += 1

            else:
                # Model improved
                if val_score < early_stopping_current_best_error:
                    early_stopping_current_best_error = val_score
                    early_stopping_counter = 0
                    best_model = copy.deepcopy(self.model.state_dict())
                    best_model_run = run_counter

                # Model did not improve
                else: 
                    early_stopping_counter += 1
                    print(f"incremented early stopping counter to {early_stopping_counter}/{self.early_stopping_window}")

            run_counter += 1

        print(f"    ==> Finished training after {run_counter} runs (ES={early_stopping_counter})")

        self.model.load_state_dict(best_model)
        print(f"best state loaded: run {best_model_run}")

        return train_error

    def run(self, fold):
        last_train_error = self.run_training()
        test_error = self.run_test()
        print(f"=====> Finished {self.model_name} fold {fold}. train={last_train_error}; test={test_error}")
        
    def get_targets(self, prot_ids, labels, subset, run_name):
        return labels.float().flatten()
    
    def get_loss_function(self, run_name):
        return nn.BCELoss()
        
    def save_measurements(self, result_counter, run, predictions, targets, loss_func, batch_context):
        result_counter.add_values(run=run, predictions=predictions, targets=targets, loss_func=loss_func)


    def _single_run(self, dataloader, run_name, run, total_runs, result_counter, with_grad=True):
        
        loss_func = self.get_loss_function(run_name)

        if with_grad:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(with_grad):
            # Logging
            self.last_run = run
            avg_error = 0
            num_batches = 0

            pbar = tqdm(
                dataloader,
                total=len(dataloader),
                desc=f"[{run_name}] Run {run}/{total_runs}",
                leave=True
            )

            for _, batch in enumerate(pbar):
                prot_ids = batch["prot_ids"]
                labels = batch["labels"].to(device)
                
                # Forward
                predictions = self.model(prot_ids, dataloader.subset)
                targets = self.get_targets(prot_ids, labels, dataloader.subset, run_name)
                # print(f"====> pred: {predictions.shape}; targets: {targets.shape}")
                # print(f"====> pred: {predictions};\n targets: {targets}")
                loss = loss_func(predictions, targets) 

                # Log
                avg_error += loss.item()
                num_batches += 1
                running_avg = avg_error / num_batches

                batch_context = {
                    "prot_ids": prot_ids,
                    "subset": dataloader.subset,
                    "original_binary_labels": labels
                }
                self.save_measurements(result_counter, run, predictions, targets, loss_func, batch_context)
                
                # Backward
                if with_grad:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # tqdm update
                pbar.set_postfix(
                    # loss=f"{loss.item():.4f}",
                    loss=f"{running_avg:.4f}",
                    acc=f"{result_counter.accuracy(run=run):.4f}",
                    prec=f"{result_counter.precision(run=run):.4f}",
                    rec=f"{result_counter.recall(run=run):.4f}",
                    f1=f"{result_counter.f1(run=run):.4f}",
                )

            # Final average for logging
            if num_batches > 0:
                avg_error /= num_batches
            else:
                avg_error = float("nan")

            return avg_error
            # return result_counter.f1(run=run)


def ensemble_pipeline_kfold_runner(model_factory, pipeline_class, id, model_name, num_folds=5, batch_size=64, max_num_runs=100, early_stopping_window=5, learning_rate=1.0e-4, save=True, single_fold=False, path_time=True):
    # ===Load dataset
    train_val_dataset = get_train_set_dataset()
    test_dataloader = get_test_set_dataloader()

    # === Make K Folds for cross-validation
    # Exctract proteins as groups for stratified KFold 
    all_keys = train_val_dataset.prot_ids
    groups = [k.rsplit('_', 1)[0] for k in all_keys] # e.g. P54278_S46N --> P54278 
    labels = train_val_dataset.get_all_labels() # For stratification
    kfold = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=42)

    t = datetime.datetime.now()
    if path_time:
        path = f"./results/{id}_{model_name}_{t.day}-{t.month}-{t.year}_{t.hour}-{t.minute}-{t.second}"
    else:
        path = f"./results/{id}_{model_name}"

    if not os.path.exists(path):
        os.makedirs(path)

    start_time = time.perf_counter()

    # === For each fold
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_val_dataset, y=labels, groups=groups)):
        # Create Subsets for this specific fold
        train_sub = Subset(train_val_dataset, train_ids)
        val_sub = Subset(train_val_dataset, val_ids)

        # Create DataLoaders
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        # Add (dataset) "subset" property directly to uniformize retrieval in the pipeline
        train_loader.subset = train_loader.dataset.dataset.subset
        val_loader.subset = val_loader.dataset.dataset.subset
        test_dataloader.subset = test_dataloader.dataset.subset

        # Create new model and pipeline
        model = model_factory()
        pipeline = pipeline_class(  model,
                                    fold,
                                    train_dataloader=train_loader,
                                    validation_dataloader=val_loader,
                                    test_dataloader=test_dataloader,
                                    early_stopping_window=early_stopping_window, 
                                    max_num_runs=max_num_runs,
                                    learning_rate=learning_rate)
        
        # Run the pipeline
        print(f"===> Starting fold {fold + 1}")
        pipeline.run(fold=fold + 1)
        print(pipeline.test_measurements.get_summary())
        print(f"TP: {pipeline.test_measurements.TP}")
        print(f"TN: {pipeline.test_measurements.TN}")
        print(f"FP: {pipeline.test_measurements.FP}")
        print(f"FN: {pipeline.test_measurements.FN}")
        print()

 
        # Save the first trained model
        if save and (fold == 0):
            model.save_model(path)

        # Save the data of the current model
        pipeline.train_measurements.persist_data(path)
        pipeline.validation_measurements.persist_data(path)
        pipeline.test_measurements.persist_data(path)
        print()

        if single_fold:
            break

    end_time = time.perf_counter()
    runtime = end_time - start_time

    with open(os.path.join(path, "runtime.txt"), "w+") as f:
        f.write(str(runtime))