import time
import torch

from models.simply_model import SimpleClassifier
from realtime.Signal_stream import stream_signals
from automation.state_machine import decide_state


def load_model(model_path="models/simple_classifier.pth"):
    """
    Load trained PyTorch model from disk.
    """

    model = SimpleClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model


def predict_signal(model, signal):
    """
    Run inference on one incoming signal window.
    """

    signal_tensor = torch.tensor(signal, dtype=torch.float32)

    # Add batch dimension: [100] -> [1, 100]
    signal_tensor = signal_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(signal_tensor)

        probabilities = torch.softmax(output, dim=1)

        normal_probability = probabilities[0, 0].item()
        anomaly_probability = probabilities[0, 1].item()

        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, normal_probability, anomaly_probability


def main():
    model = load_model()

    print("Starting real-time anomaly detection...\n")

    for step, signal, true_label in stream_signals(
        num_steps=30,
        anomaly_rate=0.25,
        signal_length=100,
        delay=0.2
    ):
        start_time = time.perf_counter()

        predicted_class, normal_prob, anomaly_prob = predict_signal(model, signal)

        state, action = decide_state(anomaly_prob)

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        true_label_name = "ANOMALY" if true_label == 1 else "NORMAL"
        pred_label_name = "ANOMALY" if predicted_class == 1 else "NORMAL"

        print(
            f"Step: {step:03d} | "
            f"True: {true_label_name:7s} | "
            f"Pred: {pred_label_name:7s} | "
            f"Anomaly Prob: {anomaly_prob:.3f} | "
            f"State: {state:9s} | "
            f"Latency: {latency_ms:.3f} ms | "
            f"Action: {action}"
        )


if __name__ == "__main__":
    main()