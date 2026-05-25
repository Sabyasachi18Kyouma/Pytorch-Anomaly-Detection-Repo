import torch

from models.simply_model import SimpleClassifier
from models.cnn1d_model import CNN1DClassifier


def load_model(model_type="cnn1d"):
    """
    Load a trained model for real-time inference.

    Parameters
    ----------
    model_type : str
        "mlp" or "cnn1d"

    Returns
    -------
    model : torch.nn.Module
        Trained PyTorch model in evaluation mode.
    """

    if model_type == "mlp":
        model = SimpleClassifier()
        model_path = "models/simple_classifier.pth"

    elif model_type == "cnn1d":
        model = CNN1DClassifier()
        model_path = "models/cnn1d_classifier.pth"

    else:
        raise ValueError("model_type must be either 'mlp' or 'cnn1d'")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model


def predict_signal(model, signal):
    """
    Run inference on one signal window.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    signal : numpy.ndarray
        Input signal of shape [100].

    Returns
    -------
    predicted_class : int
        0 = normal, 1 = anomaly.
    normal_probability : float
        Probability of normal class.
    anomaly_probability : float
        Probability of anomaly class.
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