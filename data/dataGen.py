import numpy as np


def generate_normal_signal(length=100):
    """
    Generate a normal sensor signal.
    """

    signal = np.random.normal(0, 1, length)

    return signal


def generate_anomaly_signal(length=100):
    """
    Generate an anomalous sensor signal.
    """

    signal = np.random.normal(0, 1, length)

    spike_position = np.random.randint(0, length - 5)

    signal[spike_position:spike_position + 5] += 6

    return signal


if __name__ == "__main__":

    normal_signal = generate_normal_signal()

    anomaly_signal = generate_anomaly_signal()

    print("Normal Signal:")
    print(normal_signal)

    print("\nAnomaly Signal:")
    print(anomaly_signal)

    print("\nNormal signal Data Type:")
    print(type(normal_signal))
    print("\nAnomaly Signal Data type:")
    print(type(anomaly_signal))