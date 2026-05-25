import csv
import os
import time
from datetime import datetime

from realtime.inference import load_model, predict_signal
from realtime.Signal_stream import stream_signals
from automation.state_machine import decide_state



def main():
    model = load_model(model_type="cnn1d")

    log_path = "logs/realtime_log.csv"
    os.makedirs("logs", exist_ok=True)

    total = 0
    correct = 0

    actual_anomalies = 0
    detected_anomalies = 0
    normal_samples = 0
    false_alarms = 0

    latencies = []

    print("Starting real-time anomaly detection...\n")

    with open(log_path, mode="w", newline="") as log_file:
        fieldnames = [
            "timestamp",
            "step",
            "true_label",
            "predicted_label",
            "normal_probability",
            "anomaly_probability",
            "state",
            "action",
            "latency_ms",
        ]

        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        for step, signal, true_label in stream_signals(
            num_steps=30,
            anomaly_rate=0.25,
            signal_length=100,
            delay=0.2,
        ):
            start_time = time.perf_counter()

            predicted_class, normal_prob, anomaly_prob = predict_signal(model, signal)

            state, action = decide_state(anomaly_prob)

            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            total += 1

            if predicted_class == true_label:
                correct += 1

            if true_label == 1:
                actual_anomalies += 1

                if predicted_class == 1:
                    detected_anomalies += 1

            else:
                normal_samples += 1

                if predicted_class == 1:
                    false_alarms += 1

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

            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "step": step,
                    "true_label": true_label,
                    "predicted_label": predicted_class,
                    "normal_probability": normal_prob,
                    "anomaly_probability": anomaly_prob,
                    "state": state,
                    "action": action,
                    "latency_ms": latency_ms,
                }
            )

    accuracy = correct / total if total > 0 else 0.0
    anomaly_recall = detected_anomalies / actual_anomalies if actual_anomalies > 0 else 0.0
    false_alarm_rate = false_alarms / normal_samples if normal_samples > 0 else 0.0
    average_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("\nReal-time run summary")
    print("---------------------")
    print(f"Total samples:        {total}")
    print(f"Accuracy:             {accuracy:.3f}")
    print(f"Anomaly recall:       {anomaly_recall:.3f}")
    print(f"False alarm rate:     {false_alarm_rate:.3f}")
    print(f"Average latency:      {average_latency:.3f} ms")
    print(f"Log saved to:         {log_path}")


if __name__ == "__main__":
    main()