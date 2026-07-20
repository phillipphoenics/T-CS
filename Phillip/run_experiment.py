"""End-to-end experiment: build both paradigms, train, analyze, save outputs.

Mirrors the role of `code/run_model.py` for the base project: a standalone,
reproducible script that builds the model(s), runs the full analysis suite,
saves every figure into `Phillip/plots/`, and writes a numeric summary into
`Phillip/results/summary.json`. The narrative notebook
(`working_memory_project.ipynb`) calls the same underlying functions from
`task.py`, `dynamics.py`, `reservoir_model.py`, `trained_rnn.py`, and
`analysis.py` stage by stage so it can explain and display each one; this
script just runs the whole pipeline non-interactively.
"""

import json
from pathlib import Path

import numpy as np

import analysis
import plotting
from config import ExperimentConfig
from dynamics import init_shared_weights
from reservoir_model import build_reservoir
from task import make_trial_batch
from trained_rnn import TrainingConfig, train_trained_rnn

PROJECT_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"

GLOBAL_SEED = 0

NOISE_SEVERITIES = [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0]
DEGRADATION_SEVERITIES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
SILENCING_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]


def save_fig(fig, filename: str, plots_dir: Path = PLOTS_DIR) -> Path:
    plots_dir.mkdir(exist_ok=True, parents=True)
    path = plots_dir / filename
    fig.savefig(path, format="svg", bbox_inches="tight")
    return path


def build_and_train(
    config: ExperimentConfig,
    rng: np.random.Generator,
    training: TrainingConfig | None = None,
    n_reservoir_train_trials: int = 600,
):
    """Draw the shared initialization, then build both paradigms from it."""
    J0, Jin0 = init_shared_weights(config, rng)
    reservoir = build_reservoir(config, J0.copy(), Jin0.copy(), rng, n_reservoir_train_trials)
    rnn_bundle, history = train_trained_rnn(config, J0.copy(), Jin0.copy(), rng, training)
    return reservoir, rnn_bundle, history


def run_full_analysis(config: ExperimentConfig, bundles: list, rng: np.random.Generator) -> dict:
    """Run every analysis (decoding, PCA, perturbation sweeps) on each bundle."""
    results = {"delay_curve": [], "generalization": [], "pca": [], "noise": [],
               "degradation": [], "silencing": {}, "selectivity": {}, "decoder_weight": {}}

    silencing_by_targeting = {
        "Random silencing": [],
        "Targeted: task-selectivity": [],
        "Targeted: decoder weight": [],
    }

    for bundle in bundles:
        results["delay_curve"].append(analysis.accuracy_vs_delay_length(config, bundle, rng))

        gen = analysis.temporal_generalization_matrix(config, bundle, rng)
        gen["stability_index"] = analysis.generalization_stability_index(gen)
        results["generalization"].append(gen)

        results["pca"].append(analysis.pca_trajectories(config, bundle, rng))

        fstat = analysis.delay_selectivity_fstat(config, bundle, rng)
        weight_mag = analysis.decoder_weight_magnitude(bundle)
        results["selectivity"][bundle.label] = fstat
        results["decoder_weight"][bundle.label] = weight_mag

        results["noise"].append(analysis.noise_sweep(config, bundle, rng, NOISE_SEVERITIES))
        results["degradation"].append(analysis.degradation_sweep(config, bundle, rng, DEGRADATION_SEVERITIES))

        silencing_by_targeting["Random silencing"].append(
            analysis.silencing_sweep(config, bundle, rng, SILENCING_FRACTIONS, targeting="random")
        )
        silencing_by_targeting["Targeted: task-selectivity"].append(
            analysis.silencing_sweep(
                config, bundle, rng, SILENCING_FRACTIONS, targeting="targeted", ranking_scores=fstat
            )
        )
        silencing_by_targeting["Targeted: decoder weight"].append(
            analysis.silencing_sweep(
                config, bundle, rng, SILENCING_FRACTIONS, targeting="targeted", ranking_scores=weight_mag
            )
        )

    results["silencing"] = silencing_by_targeting
    return results


def save_all_figures(config: ExperimentConfig, bundles: list, history: dict, results: dict, rng: np.random.Generator):
    example_batch = make_trial_batch(4, config, rng, delay_lengths=(config.primary_delay,))
    save_fig(plotting.plot_trial_schematic(example_batch, 0, config), "trial_schematic.svg")

    save_fig(plotting.plot_weight_spectrum_comparison(bundles), "weight_spectrum_comparison.svg")
    save_fig(plotting.plot_training_curve(history), "rnn_training_curve.svg")
    save_fig(plotting.plot_accuracy_vs_delay(results["delay_curve"]), "accuracy_vs_delay.svg")
    save_fig(plotting.plot_temporal_generalization(results["generalization"]), "temporal_generalization.svg")
    save_fig(plotting.plot_pca_trajectories(results["pca"]), "pca_trajectories.svg")
    save_fig(plotting.plot_dimensionality_summary(results["pca"]), "dimensionality_summary.svg")
    save_fig(plotting.plot_noise_sweep(results["noise"]), "perturbation_noise.svg")
    save_fig(plotting.plot_degradation_sweep(results["degradation"]), "perturbation_degradation.svg")
    save_fig(plotting.plot_silencing_comparison(results["silencing"]), "perturbation_silencing.svg")


def summarize(bundles: list, results: dict) -> dict:
    """Collapse the full result set into plain-Python scalars for summary.json."""
    summary = {}
    for i, bundle in enumerate(bundles):
        label = bundle.label
        summary[label] = {
            "accuracy_by_delay": dict(zip(
                results["delay_curve"][i]["delay_lengths"].tolist(),
                results["delay_curve"][i]["accuracy"].tolist(),
            )),
            "generalization_stability_index": results["generalization"][i]["stability_index"],
            "participation_ratio_whole_trial": results["pca"][i]["participation_ratio"],
            "participation_ratio_delay": results["pca"][i]["participation_ratio_delay"],
            "noise_severities": results["noise"][i]["severities"].tolist(),
            "noise_accuracy": results["noise"][i]["accuracy_mean"].tolist(),
            "degradation_severities": results["degradation"][i]["severities"].tolist(),
            "degradation_accuracy": results["degradation"][i]["accuracy_mean"].tolist(),
        }
        for targeting_name, sweep_list in results["silencing"].items():
            summary[label][f"silencing[{targeting_name}]_fractions"] = sweep_list[i]["fractions"].tolist()
            summary[label][f"silencing[{targeting_name}]_accuracy"] = sweep_list[i]["accuracy_mean"].tolist()
    return summary


def main() -> None:
    rng = np.random.default_rng(GLOBAL_SEED)
    config = ExperimentConfig()

    print("Building shared initialization and training both paradigms...")
    reservoir, rnn_bundle, history = build_and_train(config, rng)
    bundles = [reservoir, rnn_bundle]

    print("Reservoir readout fit. Trained RNN final eval accuracy:", history["eval_accuracy"][-1])

    print("Running full analysis suite (decoding CV + PCA + perturbation sweeps for both models; this can take ~15-20 minutes on CPU)...")
    results = run_full_analysis(config, bundles, rng)

    print("Saving figures to", PLOTS_DIR)
    save_all_figures(config, bundles, history, results, rng)

    print("Saving numeric summary to", RESULTS_DIR)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summarize(bundles, results), f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
