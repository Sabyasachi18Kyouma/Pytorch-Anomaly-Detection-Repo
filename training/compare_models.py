import os
import time

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data.synthetic_dataset import SyntheticSignalDataset
from models.simply_model import SimpleClassifier
from models.cnn1d_model import CNN1DClassifier


def load_trained_model(model_class, model_path):
    """
    Load a trained PyTorch model from disk.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Train the model first before running comparison."
        )

    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model


def evaluate_model(model, dataloader):
    """
    Evaluate one model on the same test dataset.
    """

    all_labels = []
    all_predictions = []

    total_inference_time = 0.0
    total_batches = 0
    total_samples = 0

    with torch.no_grad():
        for signals, labels in dataloader:
            start_time = time.perf_counter()

            outputs = model(signals)

            end_time = time.perf_counter()

            inference_time = end_time - start_time

            total_inference_time += inference_time
            total_batches += 1
            total_samples += labels.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    conf_matrix = confusion_matrix(all_labels, all_predictions)

    avg_batch_latency_ms = (total_inference_time / total_batches) * 1000
    avg_sample_latency_ms = (total_inference_time / total_samples) * 1000

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix,
        "avg_batch_latency_ms": avg_batch_latency_ms,
        "avg_sample_latency_ms": avg_sample_latency_ms,
    }


def print_results(model_name, results):
    """
    Print model evaluation results.
    """

    print(f"\n{model_name} Results")
    print("-" * 40)
    print(f"Accuracy:              {results['accuracy']:.4f}")
    print(f"Precision:             {results['precision']:.4f}")
    print(f"Recall:                {results['recall']:.4f}")
    print(f"F1-score:              {results['f1']:.4f}")
    print(f"Avg batch latency:     {results['avg_batch_latency_ms']:.4f} ms")
    print(f"Avg sample latency:    {results['avg_sample_latency_ms']:.6f} ms")
    print("Confusion matrix:")
    print(results["confusion_matrix"])


def main():
    test_dataset = SyntheticSignalDataset(
        num_samples=1000,
        signal_length=100
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    models_to_compare = {
        "MLP Baseline": {
            "class": SimpleClassifier,
            "path": "models/simple_classifier.pth",
        },
        "1D CNN": {
            "class": CNN1DClassifier,
            "path": "models/cnn1d_classifier.pth",
        },
    }

    print("Comparing models on the same synthetic test dataset...")
    print(f"Test samples: {len(test_dataset)}")

    comparison_results = {}

    for model_name, model_info in models_to_compare.items():
        model = load_trained_model(
            model_class=model_info["class"],
            model_path=model_info["path"],
        )

        results = evaluate_model(model, test_loader)
        comparison_results[model_name] = results

        print_results(model_name, results)

    print("\nSummary")
    print("-" * 40)
    print(
        f"{'Model':15s} | "
        f"{'Accuracy':>8s} | "
        f"{'Recall':>8s} | "
        f"{'F1':>8s} | "
        f"{'Sample Latency (ms)':>20s}"
    )
    print("-" * 75)

    for model_name, results in comparison_results.items():
        print(
            f"{model_name:15s} | "
            f"{results['accuracy']:8.4f} | "
            f"{results['recall']:8.4f} | "
            f"{results['f1']:8.4f} | "
            f"{results['avg_sample_latency_ms']:20.6f}"
        )


if __name__ == "__main__":
    main()