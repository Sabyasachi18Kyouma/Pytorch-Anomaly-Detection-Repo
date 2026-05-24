import torch

from data.dataGen import generate_normal_signal, generate_anomaly_signal
from models.simply_model import SimpleClassifier


def predict_signal(model, signal):
    model.eval()

    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    signal_tensor = signal_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(signal_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities.squeeze()


def main():
    model = SimpleClassifier()
    model.load_state_dict(torch.load("models/simple_classifier.pth"))

    normal_signal = generate_normal_signal(length=100)
    anomaly_signal = generate_anomaly_signal(length=100)

    normal_prediction, normal_probs = predict_signal(model, normal_signal)
    anomaly_prediction, anomaly_probs = predict_signal(model, anomaly_signal)

    print("Normal Signal Prediction")
    print("Predicted class:", normal_prediction)
    print("Probabilities:", normal_probs)

    print("\nAnomaly Signal Prediction")
    print("Predicted class:", anomaly_prediction)
    print("Probabilities:", anomaly_probs)


if __name__ == "__main__":
    main()