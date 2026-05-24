import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataGen import generate_normal_signal, generate_anomaly_signal


class SyntheticSignalDataset(Dataset):
    def __init__(self, num_samples=1000, signal_length=100):
        self.num_samples = num_samples
        self.signal_length = signal_length

        signals = []
        labels = []

        half = num_samples // 2

        for _ in range(half):
            signal = generate_normal_signal(signal_length)
            signals.append(signal)
            labels.append(0)

        for _ in range(num_samples - half):
            signal = generate_anomaly_signal(signal_length)
            signals.append(signal)
            labels.append(1)

        signals = np.array(signals, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        indices = np.random.permutation(num_samples)

        self.signals = torch.tensor(signals[indices], dtype=torch.float32)
        self.labels = torch.tensor(labels[indices], dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.signals[index], self.labels[index]