import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.synthetic_dataset import SyntheticSignalDataset
from models.cnn1d_model import CNN1DClassifier


def train():
    dataset = SyntheticSignalDataset(num_samples=2000, signal_length=100)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    model = CNN1DClassifier()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for signals, labels in dataloader:
            outputs = model(signals)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        average_loss = total_loss / len(dataloader)
        accuracy = correct / total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {average_loss:.4f} "
            f"Accuracy: {accuracy:.4f}"
        )

    torch.save(model.state_dict(), "models/cnn1d_classifier.pth")
    print("Model saved to models/cnn1d_classifier.pth")


if __name__ == "__main__":
    train()