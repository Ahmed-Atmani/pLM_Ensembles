import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def identity(x):
    return x

def softmax(x):
    return torch.softmax(x, dim=1)


class LearnerModel(nn.Module):
    """
    Parameters:
    - num_inputs: number of inputs (= pLM embedding dimensionality; or #pLMs for meta-learner)
    - task_type: type of the downstream task ("regression" | <integer> for classification)
    """
    def __init__(self, num_inputs, task_type="regression", num_hidden=3, layer_size=1024, dropout=0.3):
        super().__init__()

        if task_type == "regression":
            self.task_type = "regression"
            self.end_function = identity
        elif task_type == 2:
            self.task_type = "binary"
            self.end_function = torch.sigmoid
        else:
            if not isinstance(task_type, int) or task_type < 2:
                raise ValueError(f"Invalid task_type: {task_type}")
            self.task_type = "multiclass"
            self.num_classes = task_type
            self.end_function = softmax

        self.num_inputs = num_inputs
        self.num_layers = num_hidden
        self.layer_size = layer_size

        self.dropout = nn.Dropout(dropout)

        self.fc_first = nn.Linear(in_features=self.num_inputs, out_features=self.layer_size)

        # Module list becase regular list will be put in CPU instead of CUDA device
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(in_features=self.layer_size, out_features=self.layer_size) for _ in range(num_hidden)]
        )
        
        out_features = 1 if self.task_type in ["regression", "binary"] else self.num_classes
        self.fc_last = nn.Linear(in_features=self.layer_size, out_features=out_features)
    
    def forward(self, embs):
        # First layer
        x = self.fc_first(embs)
        x = F.relu(x)
        x = self.dropout(x)

        # Hidden layers
        for hidden in self.fc_hidden:
            x = hidden(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Last layer
        x = self.fc_last(x)
        x = self.end_function(x) # Apply Sigmoid/Softmax/identity depending on the type of task
        
        return x

    # def __call__(self, input):
    #     return self.forward(input)
    

class RegressionLearner(LearnerModel):
    def __init__(self, num_inputs, num_hidden=3, layer_size=1024, dropout=0.3):
        super().__init__(num_inputs, task_type="regression", num_hidden=num_hidden, layer_size=layer_size, dropout=dropout)

class ClassificationLearner(LearnerModel):
    def __init__(self, num_inputs, num_classes, num_hidden=3, layer_size=1024, dropout=0.3):
        super().__init__(num_inputs, task_type=num_classes, num_hidden=num_hidden, layer_size=layer_size, dropout=dropout)

class BinaryClassificationLearner(LearnerModel):
    def __init__(self, num_inputs, num_hidden=3, layer_size=1024, dropout=0.3):
        super().__init__(num_inputs, task_type=2, num_hidden=num_hidden, layer_size=layer_size, dropout=dropout)

def learner_factory(task_type, num_inputs, mut_combination, num_hidden=3, layer_size=1024, dropout=0.3):
    if (mut_combination == "append") or (mut_combination == "hybrid"):
        num_inputs *= 2

    match task_type:
        case "regression":
            return RegressionLearner(num_inputs=num_inputs, num_hidden=num_hidden, layer_size=layer_size, dropout=dropout).to(device)
        case 2:
            return BinaryClassificationLearner(num_inputs=num_inputs, num_hidden=num_hidden, layer_size=layer_size, dropout=dropout).to(device)
        case _:
            return ClassificationLearner(num_classes=task_type, num_inputs=num_inputs, num_hidden=num_hidden, layer_size=layer_size, dropout=dropout).to(device)
