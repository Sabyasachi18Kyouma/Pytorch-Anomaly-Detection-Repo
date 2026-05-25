import time
import numpy as np

from data.dataGen import generate_normal_signal, generate_anomaly_signal


def stream_signals(num_steps=50, anomaly_rate=0.2, signal_length=100, delay=0.2):
    """
    Simulate a real-time stream of sensor signal windows.

    Each step returns:
    - step number
    - signal as a NumPy array
    - true label: 0 = normal, 1 = anomaly
    """

    for step in range(num_steps):

        random_value = np.random.rand()

        if random_value < anomaly_rate:
            signal = generate_anomaly_signal(length=signal_length)
            true_label = 1
        else:
            signal = generate_normal_signal(length=signal_length)
            true_label = 0

        yield step, signal, true_label

        time.sleep(delay)