import math

import numpy as np

from model_setup import Array, Model, ModelConfig


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
