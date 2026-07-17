from pathlib import Path

import numpy as np

from model_setup import (
    Array,
    Model,
    ModelConfig,
    build_model,
    decode,
    decode_accuracy,
    fit_decoder,
    make_input,
    step,
)
from plotting import (
    get_pyplot,
    plot_decoder_accuracy,
    plot_input_stream,
    plot_simulation,
    plot_weight_matrix_and_spectrum,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / "plots"


def simulate(model: Model, sequence_length: int) -> tuple[Array, Array, Array]:
    """Run the recurrent network and return activity plus the generated input.

    The returned arrays share the same time axis:
    - ``firing_rates`` has shape (neurons, time),
    - ``onehot`` has shape (input channels, time),
    - ``input_stream`` stores the integer input label at each time step.
    """
    config = model.config

    # Create the stimulus sequence for this run. The one-hot version drives the
    # model, while the integer stream is easier to plot and use as decoder labels.
    onehot, input_stream = make_input(sequence_length, config)

    # Store every neuron's firing rate at every time step. The first column is
    # initialized with small random activity so the reservoir does not start from
    # a perfectly silent, symmetric state.
    firing_rates = np.zeros((config.N, len(input_stream)))
    firing_rates[:, 0] = np.random.uniform(0, 0.1, size=config.N)

    # Integrate forward one time step at a time. The activity at time t and the
    # input at time t determine the activity written into time t + 1.
    for t in range(len(input_stream) - 1):
        firing_rates[:, t + 1] = step(firing_rates[:, t], onehot[:, t], model)

    return firing_rates, onehot, input_stream


def save_svg(fig, filename: str) -> Path:
    """Save a matplotlib figure as SVG in the project plots directory."""
    # Ensure the output directory exists before writing into it.
    PLOTS_DIR.mkdir(exist_ok=True)
    output_path = PLOTS_DIR / filename
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    return output_path


def can_show_plots(plt) -> bool:
    """Return whether the current matplotlib backend can open plot windows."""
    # The Agg backend is non-interactive and is commonly used on servers or in
    # scripts that only save files. Calling plt.show() there is not useful.
    return "agg" not in plt.get_backend().lower()


def main() -> None:
    """Build the model, generate diagnostic plots, fit the decoder, and test it."""
    plt = get_pyplot()

    # Use the default model parameters and build one random reservoir instance.
    config = ModelConfig()
    model = build_model(config)

    # Plot an example input sequence to check stimulus timing and encoding.
    onehot, stream = make_input(sequence_length=50, config=config)
    input_fig = plot_input_stream(onehot, stream, config)
    input_path = save_svg(input_fig, "input_stream.svg")

    # Plot a small part of the recurrent matrix and its eigenvalue spectrum to
    # verify that the reservoir weights have the expected scale.
    weights_fig = plot_weight_matrix_and_spectrum(model)
    weights_path = save_svg(weights_fig, "weight_matrix_and_spectrum.svg")

    # Run a short simulation for visual inspection of reservoir activity.
    firing_rates, _, input_stream = simulate(model, sequence_length=10)
    simulation_fig = plot_simulation(firing_rates, input_stream, config)
    simulation_path = save_svg(simulation_fig, "simulation_activity.svg")

    # Train the decoder on an independent, longer simulation. Because each
    # integration step writes activity at t + 1 from input at t, the decoder is
    # fit on firing_rates[:, 1:] against train_stream[:-1].
    train_rates, _, train_stream = simulate(model, sequence_length=500)
    fit_decoder(model, train_rates[:, 1:], train_stream[:-1])

    # Test decoder accuracy on a fresh simulation so the reported performance is
    # measured on data that were not used to fit the readout weights.
    test_rates, _, test_stream = simulate(model, sequence_length=100)
    decoded_scores = decode(model, test_rates[:, 1:])
    decode_targets = test_stream[:-1]

    # Evaluate both all time points and only time points where an actual input is
    # present. This separates silent-period performance from stimulus decoding.
    active_input_mask = decode_targets > 0
    overall_accuracy = decode_accuracy(decoded_scores, decode_targets)
    active_accuracy = decode_accuracy(decoded_scores, decode_targets, active_input_mask)

    # Save a visual decoder diagnostic showing target labels, decoded labels, and
    # rolling accuracy over time.
    decoder_accuracy_fig = plot_decoder_accuracy(
        decoded_scores,
        decode_targets,
        config,
        mask=active_input_mask,
    )
    decoder_accuracy_path = save_svg(decoder_accuracy_fig, "decoder_accuracy.svg")

    # Report all generated files relative to the project root for readable output.
    for path in (input_path, weights_path, simulation_path, decoder_accuracy_path):
        print(f"Saved {path.relative_to(PROJECT_ROOT)}")

    print(f"Decode accuracy: {overall_accuracy:.3f}")
    print(f"Decode accuracy while input is active: {active_accuracy:.3f}")

    # Open interactive windows only when the selected backend supports it.
    if can_show_plots(plt):
        plt.show()


if __name__ == "__main__":
    # This guard lets the functions above be imported from notebooks or tests
    # without running the full simulation script immediately.
    main()
