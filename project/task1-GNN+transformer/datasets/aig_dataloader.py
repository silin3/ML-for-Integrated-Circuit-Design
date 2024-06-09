import os
import torch

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from datasets.aig_dataset import AIGDataset

os.sys.path.append('../')


def load_data(batch_size=32, shuffle=True, aig_tensor_path='data/origin/aig_tensor.pt', train_data_path='data/alu2.pt'):
    aig_tensor = torch.load(aig_tensor_path)
    full_data = torch.load(train_data_path)

    train_size = int(0.8 * len(full_data))
    valid_size = (len(full_data) - train_size) // 2
    test_size = len(full_data) - train_size - valid_size

    train_data, valid_data, test_data = random_split(
        full_data, [train_size, valid_size, test_size])

    train_dataset = AIGDataset(aig_tensor, train_data)
    valid_dataset = AIGDataset(aig_tensor, valid_data)
    test_dataset = AIGDataset(aig_tensor, test_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
