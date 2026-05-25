from realtime.Signal_stream import stream_signals


for step, signal, label in stream_signals(
    num_steps=10,
    anomaly_rate=0.3,
    signal_length=100,
    delay=0.1
):
    label_name = "ANOMALY" if label == 1 else "NORMAL"

    print(
        f"Step: {step:03d} | "
        f"Label: {label_name:7s} | "
        f"Signal shape: {signal.shape}"
    )