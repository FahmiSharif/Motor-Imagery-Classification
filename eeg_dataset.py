import numpy as np
from torch.utils.data import Dataset
import torch
from augmentations import apply_augmentations  # Optional

class EEGNPZDataset(Dataset):
    def __init__(self, npz_path, augment=False, augmentations=[]):
        loaded = np.load(npz_path, allow_pickle=True)
        self.data = loaded["data"].astype(np.float32)
        self.labels = loaded["labels"].astype(np.int64)
        self.subjects = loaded["subject_ids"]  # Now using 'subject_ids' from merged file

        self.augment = augment
        self.augmentations = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        s = self.subjects[idx]

        if isinstance(s, bytes):  # In case subject ID is saved as bytes
            s = s.decode()

        if self.augment:
            x = apply_augmentations(x, self.augmentations)

        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.long)

        return x, y, s  # s is a subject string like 'S001'