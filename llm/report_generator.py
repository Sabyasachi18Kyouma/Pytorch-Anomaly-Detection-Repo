import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from llm.prompt_templates import build_incident_report_prompt


def load_log_rows(log_path):
    """
    Load real-time runtime log CSV.
    """

    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Log file not found: {log_path}\n"
            f"Run the real-time demo first:\n"
            f"python main.py"
        )

    with open(log_path, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("Log file is empty.")

    return rows


def compute_summary(rows):
    """
    Compute runtime statistics from log rows.
    """

    total_samples = len(rows)

    true_labels = [int(row["true_label"]) for row in rows]
    predicted_labels = [int(row["predicted_label"]) for row in rows]

    anomaly_probs = [float(row["anomaly_probability"]) for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]
    states = [row["state"] for row in rows]

    correct_predictions = sum(
        1 for true, pred in zip(true_labels, predicted_labels)
        if true == pred
    )

    actual_anomalies = sum(1 for label in true_labels if label == 1)
    predicted_anomalies = sum(1 for label in predicted_labels if label == 1)

    detected_anomalies = sum(
        1 for true, pred in zip(true_labels, predicted_labels)
        if true == 1 and pred == 1
    )

    normal_samples = sum(1 for label in true_labels if label == 0)

    false_alarms = sum(
        1 for true, pred in zip(true_labels, predicted_labels)
        if true == 0 and pred == 1
    )

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    anomaly_recall = (
        detected_anomalies / actual_anomalies
        if actual_anomalies > 0
        else 0.0
    )

    false_alarm_rate = (
        false_alarms / normal_samples
        if normal_samples > 0
        else 0.0
    )

    summary = {
        "total_samples": total_samples,
        "actual_anomalies": actual_anomalies,
        "predicted_anomalies": predicted_anomalies,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "anomaly_recall": anomaly_recall,
        "false_alarm_rate": false_alarm_rate,
        "safe_mode_count": states.count("SAFE_MODE"),
        "critical_count": states.count("CRITICAL"),
        "warning_count": states.count("WARNING"),
        "normal_count": states.count("NORMAL"),
        "max_anomaly_probability": max(anomaly_probs),
        "avg_anomaly_probability": sum(anomaly_probs) / len(anomaly_probs),
        "avg_latency_ms": sum(latencies) / len(latencies),
        "max_latency_ms": max(latencies),
    }

    return summary


def generate_llm_report(summary):
    """
    Generate incident report using OpenAI.
    """

    load_dotenv()

    model_name = os.getenv("OPENAI_MODEL", "gpt-5.5")

    client = OpenAI()

    prompt = build_incident_report_prompt(summary)

    response = client.responses.create(
        model=model_name,
        input=prompt,
    )

    return response.output_text


def save_report(report_text, output_path):
    """
    Save report as Markdown file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode="w", encoding="utf-8") as file:
        file.write(report_text)

    print(f"Report saved to: {output_path}")


def main():
    log_path = "logs/realtime_log_cnn1d.csv"
    output_path = "reports/incident_report.md"

    rows = load_log_rows(log_path)
    summary = compute_summary(rows)

    print("Computed log summary:")
    print(summary)

    report_text = generate_llm_report(summary)

    save_report(report_text, output_path)


if __name__ == "__main__":
    main()