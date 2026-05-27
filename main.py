import argparse

from realtime.run_realtime import run_realtime_detection


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PyTorch-Based Real-Time Anomaly Detection and Automation System"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="cnn1d",
        choices=["cnn1d", "mlp"],
        help="Model type to use for real-time inference.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of real-time signal windows to process.",
    )

    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.25,
        help="Probability of generating an anomalous signal.",
    )

    parser.add_argument(
        "--signal-length",
        type=int,
        default=100,
        help="Length of each signal window.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between streamed signals in seconds.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        run_realtime_detection(
            model_type=args.model,
            num_steps=args.steps,
            anomaly_rate=args.anomaly_rate,
            signal_length=args.signal_length,
            delay=args.delay,
        )

    except FileNotFoundError:
        print("\nModel weights were not found.")
        print("--------------------------------")
        print("Train the required model first:")

        if args.model == "cnn1d":
            print("python -m training.train_cnn1d")

        elif args.model == "mlp":
            print("python -m training.train_simple")

        print("\nThen run the demo again:")
        print("python main.py")


if __name__ == "__main__":
    main()