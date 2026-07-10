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
    plot_input_stream,
    plot_simulation,
    plot_weight_matrix_and_spectrum,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / "plots"


def simulate(model: Model, sequence_length: int) -> tuple[Array, Array, Array]:
    """Run a simulation and return firing rates plus the generated input."""
    config = model.config
    onehot, input_stream = make_input(sequence_length, config)

    firing_rates = np.zeros((config.N, len(input_stream)))
    firing_rates[:, 0] = np.random.uniform(0, 0.1, size=config.N)

    for t in range(len(input_stream) - 1):
        firing_rates[:, t + 1] = step(firing_rates[:, t], onehot[:, t], model)

    return firing_rates, onehot, input_stream


def save_svg(fig, filename: str) -> Path:
    """Save a matplotlib figure as SVG in the plots directory."""
    PLOTS_DIR.mkdir(exist_ok=True)
    output_path = PLOTS_DIR / filename
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    return output_path


def can_show_plots(plt) -> bool:
    """Return whether the current matplotlib backend can show windows."""
    return "agg" not in plt.get_backend().lower()


def main() -> None:
    plt = get_pyplot()
    config = ModelConfig()
    model = build_model(config)

    onehot, stream = make_input(sequence_length=50, config=config)
    input_fig = plot_input_stream(onehot, stream, config)
    input_path = save_svg(input_fig, "input_stream.svg")

    weights_fig = plot_weight_matrix_and_spectrum(model)
    weights_path = save_svg(weights_fig, "weight_matrix_and_spectrum.svg")

    firing_rates, _, input_stream = simulate(model, sequence_length=10)
    simulation_fig = plot_simulation(firing_rates, input_stream, config)
    simulation_path = save_svg(simulation_fig, "simulation_activity.svg")

    train_rates, _, train_stream = simulate(model, sequence_length=500)
    fit_decoder(model, train_rates[:, 1:], train_stream[:-1])

    test_rates, _, test_stream = simulate(model, sequence_length=100)
    decoded_scores = decode(model, test_rates[:, 1:])
    decode_targets = test_stream[:-1]
    active_input_mask = decode_targets > 0
    overall_accuracy = decode_accuracy(decoded_scores, decode_targets)
    active_accuracy = decode_accuracy(decoded_scores, decode_targets, active_input_mask)

    for path in (input_path, weights_path, simulation_path):
        print(f"Saved {path.relative_to(PROJECT_ROOT)}")

    print(f"Decode accuracy: {overall_accuracy:.3f}")
    print(f"Decode accuracy while input is active: {active_accuracy:.3f}")

    if can_show_plots(plt):
        plt.show()


if __name__ == "__main__":
    main()
