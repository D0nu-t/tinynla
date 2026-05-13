import os
import torch

from torch.utils.data import Dataset


class ActivationDataset(Dataset):
    def __init__(self, path):
        self.samples = torch.load(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        return {
            "description": item["description"],
            "activation": item["activation"]
        }


def save_dataset(samples, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save(samples, output_path)