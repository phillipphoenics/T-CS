import math

import numpy as np

from model_setup import Array, Model, ModelConfig, decode_accuracy, decoded_stream


def get_pyplot():
    """Import matplotlib only when plotting is actually needed."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with "
            "`pip install matplotlib`."
        ) from exc

    return plt


def plot_input_stream(onehot: Array, stream: Array, config: ModelConfig):
    """Plot the generated input as values and one-hot activity."""
    plt = get_pyplot()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    omit = int(config.burnIn / config.dt)

    ax[0].plot(np.arange(len(stream) - omit) * config.dt, stream[omit:])
    ax[0].set_xlabel("time (ms)")
    ax[0].set_ylabel("input value")

    ax[1].imshow(onehot[:, omit:], aspect="auto")
    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("input one-hot encoding")

    fig.tight_layout()
    return fig


def plot_weight_matrix_and_spectrum(model: Model, show_count: int = 50):
    """Plot a sample of J and its eigenvalue spectrum."""
    plt = get_pyplot()
    eigenvalues, _ = np.linalg.eig(model.J)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sample = ax[0].imshow(model.J[:show_count, :show_count])
    ax[0].set_title("Sample from weight matrix J")
    ax[0].set_xlabel("presynaptic neuron")
    ax[0].set_ylabel("postsynaptic neuron")
    plt.colorbar(sample, ax=ax[0])

    circle = np.linspace(0, 2 * math.pi, 100)
    ax[1].plot(np.real(eigenvalues), np.imag(eigenvalues), ".")
    ax[1].plot(np.sin(circle), np.cos(circle))
    ax[1].set_title("Eigenvalue spectrum of J")
    ax[1].set_xlabel("real component")
    ax[1].set_ylabel("imaginary component")

    fig.tight_layout()
    return fig


def plot_simulation(firing_rates: Array, input_stream: Array, config: ModelConfig):
    """Plot the input stream and simulated neural activity."""
    plt = get_pyplot()
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    simulation_time = np.arange(len(input_stream)) * config.dt - config.burnIn

    ax[0].plot(simulation_time, input_stream)
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("Input value")

    extents = [simulation_time[0], simulation_time[-1], 0, config.N]
    ax[1].imshow(firing_rates, aspect="auto", extent=extents)
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("Neurons")

    fig.tight_layout()
    return fig


def plot_decoder_accuracy(
    decoded_scores: Array,
    target_stream: Array,
    config: ModelConfig,
    mask: Array | None = None,
    window_ms: float = 5.0,
):
    """Plot decoder performance over time.

    The upper panel compares the true input labels with the decoder's predicted
    labels. The lower panel shows a rolling accuracy trace, which is easier to
    read than single-time-step correctness when many samples are plotted.

    Parameters
    ----------
    decoded_scores:
        Decoder output with shape (classes, time).
    target_stream:
        Integer class labels with one value per decoded time step.
    config:
        Model configuration used to convert samples to milliseconds.
    mask:
        Optional boolean array selecting which time points should count toward
        the reported and rolling accuracy. Use ``target_stream > 0`` to focus on
        time points where an input is present.
    window_ms:
        Width of the rolling accuracy window in milliseconds.
    """
    plt = get_pyplot()

    if decoded_scores.shape[1] != target_stream.size:
        raise ValueError("decoded_scores and target_stream must have matching time axes.")

    predicted_stream = decoded_stream(decoded_scores)
    correct = predicted_stream == target_stream

    if mask is None:
        mask = np.ones(target_stream.shape, dtype=bool)
        title_suffix = "all time points"
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != target_stream.shape:
            raise ValueError("mask must have the same shape as target_stream.")
        title_suffix = "masked time points"

    accuracy = decode_accuracy(decoded_scores, target_stream, mask)

    # Convert the requested time window to samples and keep at least one sample
    # so very small windows still produce a valid rolling trace.
    window_steps = min(target_stream.size, max(1, int(round(window_ms / config.dt))))
    weights = np.ones(window_steps)

    # Rolling accuracy should ignore samples outside the mask rather than
    # treating them as wrong. Dividing by the rolling count of selected samples
    # handles windows that only partially overlap the mask.
    rolling_correct = np.convolve(correct * mask, weights, mode="same")
    rolling_count = np.convolve(mask.astype(float), weights, mode="same")
    rolling_accuracy = np.divide(
        rolling_correct,
        rolling_count,
        out=np.full_like(rolling_correct, np.nan, dtype=float),
        where=rolling_count > 0,
    )

    time = np.arange(target_stream.size) * config.dt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax[0].plot(time, target_stream, label="target", linewidth=1.5)
    ax[0].plot(time, predicted_stream, label="decoded", linewidth=1.0, alpha=0.8)
    ax[0].set_ylabel("input class")
    ax[0].set_title(f"Decoder labels ({title_suffix})")
    ax[0].legend(loc="upper right")

    ax[1].plot(time, rolling_accuracy, color="tab:green", linewidth=1.5)
    ax[1].axhline(accuracy, color="black", linestyle="--", linewidth=1.0)
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title(f"Decoder accuracy = {accuracy:.3f}")

    fig.tight_layout()
    return fig
