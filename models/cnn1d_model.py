import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """
    1D CNN classifier for sensor signal anomaly detection.

    Input shape:
        [batch_size, signal_length]
        or
        [batch_size, 1, signal_length]

    Output shape:
        [batch_size, 2]
    """

    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # If input is [batch_size, signal_length],
        # convert it to [batch_size, 1, signal_length]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        features = self.feature_extractor(x)

        # Convert [batch_size, 64, 1] to [batch_size, 64]
        features = features.squeeze(-1)

        output = self.classifier(features)

        return output