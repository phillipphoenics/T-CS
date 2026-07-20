"""Figure-generating functions for the reservoir-vs-trained-RNN study.

Follows the lazy-import convention from `code/plotting.py` (`get_pyplot()`),
so importing this module never requires matplotlib until a figure is
actually drawn. Every function returns the `Figure` object rather than
calling `plt.show()`, matching the base project's style.

Color usage is fixed for the whole study (see the palette dict below):
Fixed Reservoir is always blue, Trained RNN is always green, in every figure
that compares the two paradigms. Cue-identity classes (used only in the PCA
trajectory panels, one column per model) cycle through the same validated
eight-hue set in fixed order; with `config.n_classes == 8` that uses every
slot, including blue/green again for classes 0/1. That reuse is local and
harmless: each PCA panel is already titled by model (color there encodes
class identity, not model identity), and every trajectory also carries a
direct class-index label at its endpoint as a non-color-dependent cue, so
identity is never carried by hue alone. Values come from the repository's
validated categorical/sequential palette; see the `dataviz` skill for the
derivation.
"""

import numpy as np

Array = np.ndarray

# --- Palette (validated categorical + sequential ramp; see module docstring) ---
MODEL_COLORS = {
    "Fixed Reservoir": "#2a78d6",   # categorical slot 1 (blue)
    "Trained RNN": "#008300",        # categorical slot 2 (green)
}
CLASS_COLORS = [
    "#2a78d6",  # slot 1 blue
    "#008300",  # slot 2 green
    "#e87ba4",  # slot 3 magenta
    "#eda100",  # slot 4 yellow
    "#1baf7a",  # slot 5 aqua
    "#eb6834",  # slot 6 orange
    "#4a3aa7",  # slot 7 violet
    "#e34948",  # slot 8 red
]
SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def get_pyplot():
    """Import matplotlib only when plotting is actually needed."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc
    return plt


def _sequential_blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE_STEPS)


def _style_axis(ax):
    """Recessive grid/axes: light gridlines, muted ticks, no top/right spine."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.title.set_color(INK_PRIMARY)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _mark_epochs(ax, batch_like, vertical=True):
    """Dashed boundary lines + small labels at cue/delay/recall transitions.

    ``batch_like`` needs ``time_ms``, ``cue_mask``, ``delay_mask``, ``recall_mask``
    (1-D, one fixed-delay trial's worth — as returned by the analysis functions).
    """
    time_ms = batch_like["time_ms"]
    cue_end = time_ms[batch_like["cue_mask"]][-1] if np.any(batch_like["cue_mask"]) else None
    delay_end = time_ms[batch_like["delay_mask"]][-1] if np.any(batch_like["delay_mask"]) else None
    boundaries = [t for t in (cue_end, delay_end) if t is not None]
    line_fn = ax.axvline if vertical else ax.axhline
    for t in boundaries:
        line_fn(t, color=INK_MUTED, linestyle="--", linewidth=1.0, alpha=0.8)


def plot_trial_schematic(batch, trial_idx: int, config) -> "Figure":
    """Inputs and recall target for one example trial, with epoch boundaries."""
    plt = get_pyplot()
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    time_ms = batch.time_ms
    inputs = batch.inputs[trial_idx]
    targets = batch.targets[trial_idx]

    ax[0].imshow(
        inputs, aspect="auto", cmap=_sequential_blue_cmap(),
        extent=[time_ms[0], time_ms[-1], inputs.shape[0] - 0.5, -0.5],
    )
    ax[0].set_yticks(range(inputs.shape[0]))
    ax[0].set_yticklabels([f"cue {i}" for i in range(config.n_classes)] + ["go"])
    ax[0].set_title(f"Example trial (delay = {batch.delay_length_ms[trial_idx]:.0f} ms)")

    ax[1].imshow(
        targets, aspect="auto", cmap=_sequential_blue_cmap(),
        extent=[time_ms[0], time_ms[-1], targets.shape[0] - 0.5, -0.5],
    )
    ax[1].set_yticks(range(targets.shape[0]))
    ax[1].set_yticklabels([f"class {i}" for i in range(config.n_classes)])
    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("recall target")
    ax[0].set_ylabel("input channel")

    single = {
        "time_ms": time_ms,
        "cue_mask": batch.cue_mask[trial_idx],
        "delay_mask": batch.delay_mask[trial_idx],
        "recall_mask": batch.recall_mask[trial_idx],
    }
    for a in ax:
        _mark_epochs(a, single)
        a.set_facecolor(SURFACE)

    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def plot_weight_spectrum_comparison(bundles: list) -> "Figure":
    """Sample of J plus its eigenvalue spectrum, side by side per model.

    Shows directly what gradient descent did to the recurrent matrix
    relative to the shared random initialization the reservoir kept as-is.
    """
    plt = get_pyplot()
    fig, ax = plt.subplots(2, len(bundles), figsize=(6 * len(bundles), 9))
    if len(bundles) == 1:
        ax = ax[:, None]

    show_count = 50
    for col, bundle in enumerate(bundles):
        color = MODEL_COLORS.get(bundle.label, INK_PRIMARY)
        sample = ax[0, col].imshow(
            bundle.J[:show_count, :show_count], cmap=_sequential_blue_cmap()
        )
        ax[0, col].set_title(f"{bundle.label}: sample of J", color=color)
        ax[0, col].set_xlabel("presynaptic neuron")
        ax[0, col].set_ylabel("postsynaptic neuron")
        plt.colorbar(sample, ax=ax[0, col], fraction=0.046, pad=0.04)

        eigenvalues = np.linalg.eigvals(bundle.J)
        circle = np.linspace(0, 2 * np.pi, 200)
        ax[1, col].plot(np.cos(circle), np.sin(circle), color=INK_MUTED, linewidth=1.0)
        ax[1, col].scatter(
            np.real(eigenvalues), np.imag(eigenvalues), s=10, color=color, alpha=0.7
        )
        ax[1, col].set_title(f"{bundle.label}: eigenvalue spectrum", color=color)
        ax[1, col].set_xlabel("real component")
        ax[1, col].set_ylabel("imaginary component")
        ax[1, col].set_aspect("equal")
        _style_axis(ax[1, col])

    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def plot_training_curve(history: dict) -> "Figure":
    """Loss and held-out recall accuracy across training, as two separate
    axes (never dual-axis — the two quantities have unrelated scales)."""
    plt = get_pyplot()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    color = MODEL_COLORS["Trained RNN"]
    ax[0].plot(history["step"], history["loss"], color=color, linewidth=2)
    ax[0].set_xlabel("training step")
    ax[0].set_ylabel("recall-window cross-entropy loss")
    ax[0].set_title("Training loss")

    ax[1].plot(history["step"], history["eval_accuracy"], color=color, linewidth=2)
    ax[1].axhline(1.0, color=INK_MUTED, linestyle=":", linewidth=1)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("training step")
    ax[1].set_ylabel("held-out recall accuracy")
    ax[1].set_title("Training progress")

    for a in ax:
        _style_axis(a)
        a.set_facecolor(SURFACE)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def plot_accuracy_vs_delay(results: list) -> "Figure":
    """Recall accuracy vs. delay length, one line per model."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for result in results:
        color = MODEL_COLORS.get(result["label"], INK_PRIMARY)
        ax.plot(
            result["delay_lengths"], result["accuracy"], marker="o", markersize=6,
            color=color, linewidth=2, label=result["label"],
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("delay length (ms)")
    ax.set_ylabel("recall accuracy")
    ax.set_title("Working-memory accuracy across delay lengths")
    ax.legend(frameon=False, labelcolor=INK_PRIMARY)
    _style_axis(ax)
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def plot_temporal_generalization(results: list) -> "Figure":
    """Train-time x test-time decoding accuracy heatmaps, one panel per model."""
    plt = get_pyplot()
    fig, ax = plt.subplots(1, len(results), figsize=(6.5 * len(results), 5.5))
    if len(results) == 1:
        ax = [ax]

    for col, result in enumerate(results):
        time_ms = result["time_ms"]
        extent = [time_ms[0], time_ms[-1], time_ms[-1], time_ms[0]]
        im = ax[col].imshow(
            result["accuracy"], extent=extent, cmap=_sequential_blue_cmap(),
            vmin=0, vmax=1, aspect="equal",
        )
        color = MODEL_COLORS.get(result["label"], INK_PRIMARY)
        stability = result.get("stability_index")
        title = f"{result['label']}"
        if stability is not None:
            title += f"  (stability = {stability:.2f})"
        ax[col].set_title(title, color=color)
        ax[col].set_xlabel("test time (ms)")
        ax[col].set_ylabel("train time (ms)")
        single = {
            "time_ms": time_ms,
            "cue_mask": result["cue_mask"],
            "delay_mask": result["delay_mask"],
            "recall_mask": result["recall_mask"],
        }
        _mark_epochs(ax[col], single, vertical=True)
        _mark_epochs(ax[col], single, vertical=False)
        plt.colorbar(im, ax=ax[col], fraction=0.046, pad=0.04, label="decoding accuracy")

    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def plot_pca_trajectories(results: list) -> "Figure":
    """Per-class trajectories in PC1-PC2 space (top row) and PC1 over time
    (bottom row), one column per model."""
    plt = get_pyplot()
    fig, ax = plt.subplots(2, len(results), figsize=(6.5 * len(results), 10))
    if len(results) == 1:
        ax = ax[:, None]

    n_classes = results[0]["class_trajectories"].shape[0]
    for col, result in enumerate(results):
        traj = result["class_trajectories"]  # (n_classes, n_components, T)
        time_ms = result["time_ms"]
        color = MODEL_COLORS.get(result["label"], INK_PRIMARY)

        top = ax[0, col]
        for c in range(n_classes):
            class_color = CLASS_COLORS[c % len(CLASS_COLORS)]
            top.plot(traj[c, 0], traj[c, 1], color=class_color, linewidth=2, alpha=0.9)
            top.scatter(traj[c, 0, 0], traj[c, 1, 0], color=class_color, marker="o", s=40, zorder=3)
            top.scatter(traj[c, 0, -1], traj[c, 1, -1], color=class_color, marker="s", s=40, zorder=3)
            top.annotate(str(c), (traj[c, 0, -1], traj[c, 1, -1]), color=INK_PRIMARY,
                         fontsize=9, fontweight="bold", ha="center", va="center")
        top.set_title(f"{result['label']}: state-space trajectories", color=color)
        top.set_xlabel("PC1")
        top.set_ylabel("PC2")

        bottom = ax[1, col]
        for c in range(n_classes):
            class_color = CLASS_COLORS[c % len(CLASS_COLORS)]
            bottom.plot(time_ms, traj[c, 0], color=class_color, linewidth=2, alpha=0.9, label=f"class {c}")
        single = {
            "time_ms": time_ms, "cue_mask": result["cue_mask"],
            "delay_mask": result["delay_mask"], "recall_mask": result["recall_mask"],
        }
        _mark_epochs(bottom, single)
        bottom.set_xlabel("time (ms)")
        bottom.set_ylabel("PC1")
        bottom.set_title(f"{result['label']}: PC1 over time", color=color)
        if col == len(results) - 1:
            bottom.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=8, loc="upper right")

        for a in (top, bottom):
            _style_axis(a)
            a.set_facecolor(SURFACE)

    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def plot_dimensionality_summary(results: list) -> "Figure":
    """Grouped bar chart: participation ratio, whole-trial vs. delay-only, per model."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 5))

    groups = ["Whole trial", "Delay only"]
    x = np.arange(len(groups))
    width = 0.35 if len(results) == 2 else 0.6 / max(len(results), 1)

    for i, result in enumerate(results):
        color = MODEL_COLORS.get(result["label"], INK_PRIMARY)
        values = [result["participation_ratio"], result["participation_ratio_delay"]]
        offset = (i - (len(results) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, color=color, label=result["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("participation ratio (effective # of PCs)")
    ax.set_title("Dimensionality of population activity")
    ax.legend(frameon=False, labelcolor=INK_PRIMARY)
    _style_axis(ax)
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def _sweep_plot(results: list, x_key: str, xlabel: str, title: str) -> "Figure":
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for result in results:
        color = MODEL_COLORS.get(result["label"], INK_PRIMARY)
        x = result[x_key]
        mean = result["accuracy_mean"]
        std = result["accuracy_std"]
        ax.plot(x, mean, marker="o", markersize=5, color=color, linewidth=2, label=result["label"])
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("recall accuracy")
    ax.set_title(title)
    ax.legend(frameon=False, labelcolor=INK_PRIMARY)
    _style_axis(ax)
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def plot_noise_sweep(results: list) -> "Figure":
    return _sweep_plot(
        results, "severities", "recurrent noise std (added during delay)",
        "Robustness to additive recurrent noise",
    )


def plot_degradation_sweep(results: list) -> "Figure":
    return _sweep_plot(
        results, "severities", "weight-matrix degradation (x weight std, during delay)",
        "Robustness to weight-matrix degradation",
    )


def plot_silencing_comparison(sweep_by_targeting: dict) -> "Figure":
    """One panel per targeting strategy; each panel compares both models.

    ``sweep_by_targeting``: {targeting_name: [reservoir_result, rnn_result]}.
    Column order and titles come from the dict's insertion order.
    """
    plt = get_pyplot()
    n_panels = len(sweep_by_targeting)
    fig, ax = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharey=True)
    if n_panels == 1:
        ax = [ax]

    for col, (targeting_name, results) in enumerate(sweep_by_targeting.items()):
        for result in results:
            color = MODEL_COLORS.get(result["label"], INK_PRIMARY)
            x = result["fractions"]
            mean = result["accuracy_mean"]
            std = result["accuracy_std"]
            ax[col].plot(x, mean, marker="o", markersize=5, color=color, linewidth=2, label=result["label"])
            ax[col].fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax[col].set_ylim(0, 1.05)
        ax[col].set_xlabel("fraction of units silenced")
        ax[col].set_title(targeting_name)
        if col == 0:
            ax[col].set_ylabel("recall accuracy")
            ax[col].legend(frameon=False, labelcolor=INK_PRIMARY)
        _style_axis(ax[col])
        ax[col].set_facecolor(SURFACE)

    fig.suptitle("Robustness to unit silencing during the delay", color=INK_PRIMARY)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig
