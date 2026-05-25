def decide_state(anomaly_probability):
    """
    Decide system state and action based on anomaly probability.

    Parameters
    ----------
    anomaly_probability : float
        Probability that the current signal is anomalous.

    Returns
    -------
    state : str
        System state.
    action : str
        Recommended system action.
    """

    if anomaly_probability < 0.50:
        state = "NORMAL"
        action = "Continue normal operation"

    elif anomaly_probability < 0.75:
        state = "WARNING"
        action = "Monitor system closely"

    elif anomaly_probability < 0.90:
        state = "CRITICAL"
        action = "Reduce load and inspect system"

    else:
        state = "SAFE_MODE"
        action = "Trigger safety shutdown procedure"

    return state, action