import pickle
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


"""
Counts the true/false positives/negatives.
(made to avoid bloat in the pipeline code)
"""
class BCCounter:
    def __init__(self, ensemble_type, plm_names, dataset, fold):
        self.TP = {}
        self.TN = {}
        self.FP = {}
        self.FN = {}
        self.errors = {}
        self.probabilities = {}
        self.targets = {}
        self.metadata = {
            "ensemble_type": ensemble_type,
            "plm_names": plm_names,
            "dataset": dataset, 
            "fold": fold,
        }
        self.last_run = -1
        
    def init_run(self, run):
        self.last_run = max(self.last_run, run)
        self.TP[run] = self.TP[run] if run in self.TP else 0
        self.TN[run] = self.TN[run] if run in self.TN else 0
        self.FP[run] = self.FP[run] if run in self.FP else 0
        self.FN[run] = self.FN[run] if run in self.FN else 0
        if run not in self.probabilities: 
            self.probabilities[run] = []
        if run not in self.targets: 
            self.targets[run] = []

    def add_value(self, run, prediction, target, loss_func):
        # Check if new run
        self.init_run(run)

        # Save loss
        if run not in self.errors: 
            self.errors[run] = []
        self.errors[run].append(loss_func(prediction, target).item())

        # Save probs (while checking if float or singleton tensor)
        prob_val = prediction.item() if hasattr(prediction, 'item') else float(prediction)
        target_val = target.item() if hasattr(target, 'item') else int(target)
        self.probabilities[run].append(prob_val)
        self.targets[run].append(target_val)

        # Save prediction
        predicted_target = 1 if prediction >= 0.5 else 0

        if predicted_target == target:
            if predicted_target == 1:
                self.TP[run] += 1
            else:
                self.TN[run] += 1
        else:
            if predicted_target == 1:
                self.FP[run] += 1
            else:
                self.FN[run] += 1

    def accuracy(self, run):
        self.init_run(run) # Check if new run

        TP = self.TP[run]
        TN = self.TN[run]
        FP = self.FP[run]
        FN = self.FN[run]

        if (TP + TN) == 0:
            return 0
        
        if (TP + TN + FP + FN) == 0:
            return 0

        return (TP + TN) / (TP + TN + FP + FN)
    
    def precision(self, run):
        self.init_run(run) # Check if new run

        TP = self.TP[run]
        FP = self.FP[run]

        if TP == 0:
            return 0
        
        if (TP + FP) == 0:
            return 0

        return TP / (TP + FP) 
    
    def recall(self, run):
        self.init_run(run) # Check if new run

        TP = self.TP[run]
        FN = self.FN[run]

        if TP == 0:
            return 0
        
        if (TP + FN) == 0:
            return 0

        return TP / (TP + FN) 
    
    def f1(self, run):
        self.init_run(run) # Check if new run
        
        precision = self.precision(run)
        recall = self.recall(run)

        if (2 * precision * recall) == 0:
            return 0
        
        if (precision + recall) == 0:
            return 0

        return (2 * precision * recall) / (precision + recall)
    
    def get_summary(self, run=None):
        if run is None:
            run = self.last_run
        return f"[{self.metadata['dataset']}] acc={self.accuracy(run)}; prec={self.precision(run)}; rec={self.recall(run)}; f1={self.f1(run)}"

    def add_values(self, run, predictions, targets, loss_func):
        for (prediction, target) in zip(predictions, targets):
            self.add_value(run, prediction, target, loss_func)

    def plot_ROC(self, run=None, ax=None):
        if run is None: 
            run = self.last_run
        
        y_true = self.targets[run]
        y_probs = self.probabilities[run]
        
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f"ROC Curve - {self.metadata['plm_names']} ({self.metadata['dataset']})")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        return ax
    
    def plot_loss_trajectory(self, val_counter=None):
        # Sort keys to ensure chronologically correct x-axis
        train_runs = sorted(self.errors.keys())
        train_loss_means = [np.mean(self.errors[r]) for r in train_runs]
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(train_runs, train_loss_means, label='Train Loss', color='royalblue', lw=2)
        
        if val_counter is not None:
            val_runs = sorted(val_counter.errors.keys())
            val_loss_means = [np.mean(val_counter.errors[r]) for r in val_runs]
            ax.plot(val_runs, val_loss_means, label='Validation Loss', color='crimson', lw=2)
            
        ax.set_xlabel('Run / Epoch ID')
        ax.set_ylabel('Mean Loss Value')
        ax.set_title(f"Training Convergence - {self.metadata['plm_names']}")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        return ax

    def persist_data(self, path):
        filepath = os.path.join(path, f"results_{self.metadata['dataset']}_fold{self.metadata['fold']}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Successfully serialized tracking object to {filepath}")


def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
