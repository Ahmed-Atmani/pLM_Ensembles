import torch
from torch.utils.data import Dataset, DataLoader
import pickle as pkl


def _load_dataset_file_dict(subset_name):
    path = f"datasets/driver_mutation/{subset_name}.pkl"
    with open(path, "rb") as f:
        data_dict = pkl.load(f)
        return data_dict

class DriverMutationDataset(Dataset):
    def __init__(self, subset_name):
        self.subset = subset_name
        self.data = _load_dataset_file_dict(subset_name)
        self.prot_ids = list(self.data.keys())
        self.subset

    def get_all_labels(self):
        values = self.data.values()
        return torch.tensor(list(map(lambda item: item["label"], values)))

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, idx):
        prot_ids = self.prot_ids[idx]
        items = self.data[prot_ids]
        
        return  {
                    "prot_ids": prot_ids,
                    "labels": torch.tensor(items["label"], dtype=torch.float32)
                }

"""
Returns the training set data loader.
"""    
def get_train_set_dataset():
    return DriverMutationDataset(subset_name="DRGN")

"""
Returns the test set data loader.
"""
def get_test_set_dataset():
    return DriverMutationDataset(subset_name="ALL")

"""
Returns the DEL and NEUT (= partitions of the full test set) data loaders separately in a tuple.
"""
def get_split_test_sets_dataset():
    dataset_del = DriverMutationDataset(subset_name="DEL"),
    dataset_neut = DriverMutationDataset(subset_name="NEUT"),
    return dataset_del, dataset_neut

"""
Returns the training set data loader.
"""    
def get_train_set_dataloader(batch_size=64):
    dataset = DriverMutationDataset(subset_name="DRGN")
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

"""
Returns the test set data loader.
"""
def get_test_set_dataloader(batch_size=64):
    dataset = DriverMutationDataset(subset_name="ALL")
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

"""
Returns the DEL and NEUT (= partitions of the full test set) data loaders separately in a tuple.
"""
def get_split_test_sets_dataloader(batch_size=64):
    dataset_del = DriverMutationDataset(subset_name="DEL"),
    dataset_neut = DriverMutationDataset(subset_name="NEUT"),
    dataloader_del = DataLoader(dataset=dataset_del, batch_size=batch_size, shuffle=True)
    dataloader_neut = DataLoader(dataset=dataset_neut, batch_size=batch_size, shuffle=True)
    return dataloader_del, dataloader_neut