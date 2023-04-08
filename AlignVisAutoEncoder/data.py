import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PairedDataset(Dataset):
    def __init__(self, tensor_X, tensor_Y):
        assert tensor_X.size(0) == tensor_Y.size(0), "X and Y tensors should have the same number of samples"
        self.tensor_X = tensor_X
        self.tensor_Y = tensor_Y

    def __getitem__(self, index):
        return self.tensor_X[index], self.tensor_Y[index]

    def __len__(self):
        return self.tensor_X.size(0)
    
class DataLoaderInit():
    def __init__(self, X, Y):
        self.tensor_X = torch.Tensor(X)
        self.tensor_Y = torch.Tensor(Y) 
        return 
    def get_data_loader(self):
        paired_dataset = PairedDataset(self.tensor_X, self.tensor_Y)
        dataloader = DataLoader(paired_dataset, batch_size=32, shuffle=True)    
        return dataloader
