def build_incident_report_prompt(summary):
    """
    Build a prompt for generating a human-readable incident report
    from real-time anomaly detection logs.
    """

    prompt = f"""

The system is a PyTorch-based real-time anomaly detection and automation system.
It monitors synthetic sensor signal windows, predicts anomaly probability,
assigns automation states, and logs runtime metrics.

Generate a concise engineering incident report in Markdown format.

Use the following runtime summary:

Total samples processed: {summary["total_samples"]}
Actual anomalies: {summary["actual_anomalies"]}
Predicted anomalies: {summary["predicted_anomalies"]}
Correct predictions: {summary["correct_predictions"]}
Accuracy: {summary["accuracy"]:.3f}
Anomaly recall: {summary["anomaly_recall"]:.3f}
False alarm rate: {summary["false_alarm_rate"]:.3f}
SAFE_MODE count: {summary["safe_mode_count"]}
CRITICAL count: {summary["critical_count"]}
WARNING count: {summary["warning_count"]}
NORMAL count: {summary["normal_count"]}
Maximum anomaly probability: {summary["max_anomaly_probability"]:.3f}
Average anomaly probability: {summary["avg_anomaly_probability"]:.3f}
Average latency: {summary["avg_latency_ms"]:.3f} ms
Maximum latency: {summary["max_latency_ms"]:.3f} ms

Write the report with these sections:

1. Run Summary
2. Anomaly Detection Behavior
3. Automation State Analysis
4. Runtime Performance
5. Recommended Engineering Actions


"""

    return prompt