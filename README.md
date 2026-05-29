# PyTorch-Based Real-Time Anomaly Detection and Automation System

A practical deep learning and software engineering project for real-time anomaly detection using PyTorch.

This project starts with synthetic sensor signals, trains anomaly detection models, compares a baseline MLP with a 1D CNN, and connects the trained model to a real-time inference and automation pipeline.

---

## Project Goal

The goal of this project is to build a complete learning-focused AI system that goes beyond a basic training notebook.

The system is designed to:

- Generate synthetic sensor-like signals
- Train PyTorch models for anomaly detection
- Compare a baseline MLP with a 1D CNN
- Simulate real-time sensor streaming
- Perform live inference
- Trigger automation states based on anomaly probability
- Log runtime results and latency metrics

---

## Current Status

### Completed

- Synthetic normal signal generation
- Synthetic anomalous signal generation
- PyTorch `Dataset` and `DataLoader`
- Baseline MLP classifier
- 1D CNN classifier for signal data
- Model training scripts
- Model evaluation scripts
- MLP vs CNN comparison script
- Real-time signal stream simulator
- Live inference pipeline
- Automation state machine
- Runtime logging to CSV
- Latency measurement
- Final demo entry point using `main.py`

## Local LLM Incident Report Generation

This project includes an optional local LLM reporting layer using Ollama.

After running the real-time anomaly detection demo:

```bash
python main.py

generate a local incident report with the command:

```bash
 python -m llm.local_report_generator

This generates a report inside the report folder.


### Planned Later

- Real-world motor/bearing fault dataset
- More advanced preprocessing
- LSTM/GRU sequence model
- Better visualization
- Unit tests
- Optional dashboard

---

## System Architecture

The project is organized as a modular real-time anomaly detection and automation pipeline with an optional local LLM-assisted reporting layer.

```text
Synthetic Signal Generator
        ↓
PyTorch Dataset / DataLoader
        ↓
Model Training
        ├── Baseline MLP Classifier
        └── 1D CNN Signal Classifier
        ↓
Saved Model Weights
        ↓
Real-Time Signal Stream
        ↓
Live PyTorch Inference
        ↓
Anomaly Probability Estimation
        ↓
Automation State Machine
        ├── NORMAL
        ├── WARNING
        ├── CRITICAL
        └── SAFE_MODE
        ↓
CSV Runtime Logging
        ↓
Local LLM Incident Report Generator
        ↓
Markdown Engineering Report
```

### Runtime Flow

During the real-time demo, the system performs the following steps:

```text
python main.py
        ↓
Load trained CNN model
        ↓
Generate synthetic live signal windows
        ↓
Run real-time anomaly prediction
        ↓
Compute anomaly probability
        ↓
Trigger automation state
        ↓
Log predictions, states, actions, and latency
        ↓
Save runtime log to CSV
```

The runtime log is saved as:

```text
logs/realtime_log_cnn1d.csv
```

### Local LLM Reporting Flow

After the real-time demo, the optional local LLM layer can generate a human-readable incident report.

```text
python -m llm.local_report_generator
        ↓
Read runtime CSV log
        ↓
Compute summary statistics
        ↓
Send structured summary to local Ollama model
        ↓
Generate Markdown incident report
        ↓
Save report locally
```

The generated report is saved as:

```text
reports/local_incident_report.md
```

This LLM reporting layer runs locally using Ollama and does not require a paid API key.

## Demo Output

Starting real-time anomaly detection
------------------------------------
Model:        cnn1d
Steps:        30
Anomaly rate: 0.25
Signal size:  100
------------------------------------

Step: 000 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 4.582 ms | Action: Continue normal operation
Step: 001 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.802 ms | Action: Continue normal operation
Step: 002 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.850 ms | Action: Continue normal operation
Step: 003 | True: ANOMALY | Pred: ANOMALY | Anomaly Prob: 1.000 | State: SAFE_MODE | Latency: 1.015 ms | Action: Trigger safety shutdown procedure
Step: 004 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.779 ms | Action: Continue normal operation
Step: 005 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.885 ms | Action: Continue normal operation
Step: 006 | True: ANOMALY | Pred: ANOMALY | Anomaly Prob: 1.000 | State: SAFE_MODE | Latency: 0.761 ms | Action: Trigger safety shutdown procedure
Step: 007 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 1.198 ms | Action: Continue normal operation
Step: 008 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 1.174 ms | Action: Continue normal operation
Step: 009 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.870 ms | Action: Continue normal operation
Step: 010 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.958 ms | Action: Continue normal operation
Step: 011 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 1.326 ms | Action: Continue normal operation
Step: 012 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.840 ms | Action: Continue normal operation
Step: 013 | True: ANOMALY | Pred: ANOMALY | Anomaly Prob: 1.000 | State: SAFE_MODE | Latency: 1.126 ms | Action: Trigger safety shutdown procedure
Step: 014 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 1.012 ms | Action: Continue normal operation
Step: 015 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.824 ms | Action: Continue normal operation
Step: 016 | True: ANOMALY | Pred: ANOMALY | Anomaly Prob: 1.000 | State: SAFE_MODE | Latency: 0.834 ms | Action: Trigger safety shutdown procedure
Step: 017 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.857 ms | Action: Continue normal operation
Step: 018 | True: ANOMALY | Pred: ANOMALY | Anomaly Prob: 1.000 | State: SAFE_MODE | Latency: 1.319 ms | Action: Trigger safety shutdown procedure
Step: 019 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.976 ms | Action: Continue normal operation
Step: 020 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.914 ms | Action: Continue normal operation
Step: 021 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 1.103 ms | Action: Continue normal operation
Step: 022 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 1.067 ms | Action: Continue normal operation
Step: 023 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.806 ms | Action: Continue normal operation
Step: 024 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.783 ms | Action: Continue normal operation
Step: 025 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 1.177 ms | Action: Continue normal operation
Step: 026 | True: ANOMALY | Pred: ANOMALY | Anomaly Prob: 1.000 | State: SAFE_MODE | Latency: 1.097 ms | Action: Trigger safety shutdown procedure
Step: 027 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.771 ms | Action: Continue normal operation
Step: 028 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.958 ms | Action: Continue normal operation
Step: 029 | True: NORMAL  | Pred: NORMAL  | Anomaly Prob: 0.000 | State: NORMAL    | Latency: 0.976 ms | Action: Continue normal operation

Real-time run summary
---------------------
Total samples:        30
Accuracy:             1.000
Anomaly recall:       1.000
False alarm rate:     0.000
Average latency:      1.088 ms