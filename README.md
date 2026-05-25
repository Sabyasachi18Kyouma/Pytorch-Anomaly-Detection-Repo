# PyTorch-Based Real-Time Anomaly Detection and Automation System

A learning-focused deep learning and software engineering project for building an anomaly detection system using PyTorch.

The current version trains a neural network classifier on synthetic sensor signals and evaluates whether a new signal is normal or anomalous. The next development stage is to extend this into a real-time inference and automation system.

---

## Project Goal

The goal of this project is to build a practical PyTorch-based anomaly detection pipeline that gradually evolves from a simple training script into a real-time AI monitoring system.

This project focuses on:

- Deep learning with PyTorch
- Synthetic sensor signal generation
- Time-series anomaly detection
- Model training and evaluation
- Real-time inference architecture
- Automation logic for system response
- Clean software engineering practices in Python

---

## Current Project Status

### Completed

- Synthetic normal signal generation
- Synthetic anomalous signal generation
- PyTorch `Dataset` implementation
- PyTorch `DataLoader` training pipeline
- Simple neural network classifier
- Training loop with loss and accuracy tracking
- Model saving using `.pth` weights
- Model evaluation on unseen generated signals
- GitHub project setup

### In Progress / Planned

- Real-time signal stream simulator
- Live model inference loop
- Automation state machine
- Logging of predictions and system states
- 1D CNN model for time-series signals
- Improved evaluation metrics such as precision, recall, F1-score, and confusion matrix

---

## Project Structure

```text
Pytorch-Anomaly-Detection-Repo/
│
├── data/
│   ├── __init__.py
│   ├── generate_data.py
│   └── synthetic_dataset.py
│
├── models/
│   ├── __init__.py
│   └── simple_model.py
│
├── training/
│   ├── __init__.py
│   ├── train_simple.py
│   └── evaluate_simple.py
│
├── notebooks/
│   ├── __init__.py
│   ├── tensor_basics.py
│   ├── test_model.py
│   └── test_dataset.py
│
├── realtime/
│   └── __init__.py
│
├── automation/
│   └── __init__.py
│
├── utils/
├── tests/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
