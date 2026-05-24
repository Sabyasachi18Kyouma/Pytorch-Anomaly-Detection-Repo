from data.synthetic_dataset import SyntheticSignalDataset


dataset = SyntheticSignalDataset(num_samples=1000, signal_length=100)

x, y = dataset[0]

print("One signal:")
print(x)

print("\nOne label:")
print(y)

print("\nSignal shape:")
print(x.shape)

print("\nDataset length:")
print(len(dataset))