import os
import torch
import numpy as np
import scipy.io as io

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def transform_loader2array(train_loader, test_loader):

    train_data, train_label, test_data, test_label = [], [], [], []

    for batch_data, batch_labels in train_loader:
        train_data.append(batch_data)
        train_label.append(batch_labels)
    for batch_data, batch_labels in test_loader:
        test_data.append(batch_data)
        test_label.append(batch_labels)
    
    return (
        torch.cat(train_data, dim=0).numpy(),
        torch.cat(train_label, dim=0).numpy(),
        torch.cat(test_data, dim=0).numpy(),
        torch.cat(test_label, dim=0).numpy(),
    )


def train_test_split(inliers, outliers):
    
    num_split = len(inliers) // 2
    train_data = inliers[:num_split]
    train_label = np.zeros(num_split)
    test_data = np.concatenate([inliers[num_split:], outliers], 0)

    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:] = 1
    return train_data, train_label, test_data, test_label


def train_test_split_mode2(inliers, outliers):
    
    num_split1 = len(inliers) // 2
    num_split2 = len(outliers) // 2

    train_data = np.concatenate([inliers[:num_split1], outliers[:num_split2]], 0)
    train_label = np.array([0] * num_split1 + [1] * num_split2)

    test_data = np.concatenate([inliers[num_split1:], outliers[num_split2:]], 0)
    test_label = np.zeros(test_data.shape[0])
    test_label[num_split1:] = 1
    return train_data, train_label, test_data, test_label


class NpzDataset(Dataset):
    def __init__(self, path: str, mode: str = "train", scaler=None, type="1"):
        
        super(NpzDataset, self).__init__()
        data = np.load(path)
        samples = data["X"]
        labels = ((data["y"]).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        if type == "1":
            train_data, train_label, test_data, test_label = train_test_split(
                inliers, outliers
            )
        else:
            train_data, train_label, test_data, test_label = train_test_split_mode2(
                inliers, outliers
            )

        if scaler is not None:
            scaler = MinMaxScaler() if scaler == "minmax" else StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)

        if mode == "train":
            self.data = torch.Tensor(train_data)
            self.targets = torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)


class MatDataset(Dataset):
    def __init__(self, path: str, mode: str = "train", scaler=None, type="1"):
        super(MatDataset, self).__init__()
        data = io.loadmat(path)
        samples = data["X"]
        labels = ((data["y"]).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        if type == "1":
            train_data, train_label, test_data, test_label = train_test_split(
                inliers, outliers
            )
        else:
            train_data, train_label, test_data, test_label = train_test_split_mode2(
                inliers, outliers
            )
        if scaler is not None:
            scaler = MinMaxScaler() if scaler == "minmax" else StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)
        if mode == "train":
            self.data = torch.Tensor(train_data)
            self.targets = torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)


def get_dataloader(path, batch_size=512, scaler=None, type="1"):

    if path.endswith(".mat"):
        train_set = MatDataset(path, mode="train", scaler=scaler, type=type)
        test_set = MatDataset(path, mode="eval", scaler=scaler, type=type)
    elif path.endswith(".npz"):
        train_set = NpzDataset(path, mode="train", scaler=scaler, type=type)
        test_set = NpzDataset(path, mode="eval", scaler=scaler, type=type)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader