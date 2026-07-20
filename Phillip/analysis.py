"""Post-hoc analyses shared by both paradigms: cross-temporal decoding,
PCA/dimensionality, lesion-target ranking, and perturbation sweeps.

Every function here takes a `WeightBundle` (either the reservoir's or the
trained RNN's) and only ever calls `dynamics.simulate_trials` on it — it has
no idea which paradigm produced the weights. That symmetry is what makes the
comparisons in this study fair: identical task, identical analysis code,
applied to both.

All time-resolved analyses (cross-temporal decoding, PCA trajectories,
perturbation sweeps) use a *single fixed delay length* per call
(`task.make_trial_batch` with a one-element `delay_lengths`), by default
`config.primary_delay` (the longest scheduled delay — the most demanding
maintenance condition). See `task.py`'s module docstring for why mixing
delay lengths would make time index `t` incomparable across trials.
"""

from typing import Optional

import numpy as np

from config import ExperimentConfig
from dynamics import Perturbation, WeightBundle, recall_accuracy, simulate_trials
from task import make_trial_batch

Array = np.ndarray


# --------------------------------------------------------------------------
# Cross-temporal ("temporal generalization") decoding
# --------------------------------------------------------------------------

def _stratified_folds(cue_labels: Array, n_folds: int, rng: np.random.Generator) -> list:
    """Boolean test-masks (over trial index) for `n_folds` class-balanced folds."""
    fold_id = np.zeros(len(cue_labels), dtype=int)
    for c in np.unique(cue_labels):
        idx = np.where(cue_labels == c)[0]
        rng.shuffle(idx)
        fold_id[idx] = np.arange(len(idx)) % n_folds
    return [fold_id == k for k in range(n_folds)]


def _fit_ridge_classifier(
    features: Array, labels: Array, n_classes: int, ridge_lambda: float
) -> Array:
    """Closed-form ridge classifier. `features`: (n_samples, N). Returns W: (n_classes, N + 1)."""
    features_b = np.hstack([features, np.ones((features.shape[0], 1))])
    targets = np.zeros((features.shape[0], n_classes))
    targets[np.arange(features.shape[0]), labels] = 1.0
    gram = features_b.T @ features_b
    gram[np.diag_indices_from(gram)] += ridge_lambda
    weights = np.linalg.solve(gram, features_b.T @ targets)  # (N + 1, n_classes)
    return weights.T


def temporal_generalization_matrix(
    config: ExperimentConfig,
    bundle: WeightBundle,
    rng: np.random.Generator,
    n_trials: int = 400,
    n_folds: int = 5,
    delay_length: Optional[float] = None,
    ridge_lambda: float = 5.0,
) -> dict:
    """Train-time x test-time cross-validated decoding accuracy (King & Dehaene, 2014).

    A ridge classifier is fit on population activity at every training time
    `t_i` (cross-validated across trials) and evaluated at every test time
    `t_j`. A code that is temporally *stable* (King & Dehaene's "stationary"
    regime) generalizes off the diagonal, producing a wide square of high
    accuracy; a purely *dynamic*, time-varying code only decodes well along
    the diagonal (t_i == t_j).
    """
    delay_length = config.primary_delay if delay_length is None else delay_length
    batch = make_trial_batch(n_trials, config, rng, delay_lengths=(delay_length,))
    R = simulate_trials(bundle, batch, rng)
    T = batch.T
    n_classes = config.n_classes

    folds = _stratified_folds(batch.cue_labels, n_folds, rng)
    acc_matrix = np.zeros((T, T))

    # Bias-augmented features at every time point, computed once: (T, n_trials, N + 1).
    features_b_all = np.concatenate(
        [R.transpose(2, 0, 1), np.ones((T, R.shape[0], 1))], axis=2
    )

    for test_mask in folds:
        train_mask = ~test_mask
        labels_train = batch.cue_labels[train_mask]
        labels_test = batch.cue_labels[test_mask]

        for t_train in range(T):
            W = _fit_ridge_classifier(
                R[train_mask, :, t_train], labels_train, n_classes, ridge_lambda
            )
            scores = np.einsum("cf,tnf->tnc", W, features_b_all[:, test_mask, :])
            predicted = np.argmax(scores, axis=2)  # (T, n_test)
            acc_matrix[t_train, :] += (predicted == labels_test[None, :]).mean(axis=1)

    acc_matrix /= len(folds)

    return {
        "accuracy": acc_matrix,
        "time_ms": batch.time_ms,
        "cue_mask": batch.cue_mask[0],
        "delay_mask": batch.delay_mask[0],
        "recall_mask": batch.recall_mask[0],
        "delay_length": delay_length,
        "label": bundle.label,
    }


def generalization_stability_index(result: dict) -> float:
    """Ratio of mean off-diagonal to mean on-diagonal decoding accuracy,
    restricted to the delay epoch.

    Near 1: a delay-period code that generalizes across time (temporally
    stable/stationary). Near 0: a code that only decodes well near its own
    training time (a genuinely time-varying/dynamic trajectory).
    """
    acc = result["accuracy"]
    delay_idx = np.where(result["delay_mask"])[0]
    if len(delay_idx) < 2:
        return float("nan")
    sub = acc[np.ix_(delay_idx, delay_idx)]
    diagonal = np.diag(sub)
    off_diagonal = sub[~np.eye(sub.shape[0], dtype=bool)]
    if diagonal.mean() <= 0:
        return float("nan")
    return float(off_diagonal.mean() / diagonal.mean())


# --------------------------------------------------------------------------
# PCA / low-dimensional trajectories
# --------------------------------------------------------------------------

def _participation_ratio(pooled_centered: Array) -> float:
    """Effective dimensionality: PR = (sum(sigma^2))^2 / sum(sigma^4).

    Equals `n_components` if variance is spread equally over that many
    directions and 1.0 if a single direction explains everything; a smooth,
    unit-free way to summarize "how many PCs does this activity really need."
    """
    singular_values = np.linalg.svd(pooled_centered, full_matrices=False, compute_uv=False)
    power = singular_values ** 2
    return float((power.sum() ** 2) / (power ** 2).sum())


def pca_trajectories(
    config: ExperimentConfig,
    bundle: WeightBundle,
    rng: np.random.Generator,
    n_trials: int = 240,
    delay_length: Optional[float] = None,
    n_components: int = 10,
) -> dict:
    """PCA on recurrent firing rates pooled across all simulated trials.

    Following the abstract's design, PCA is fit on the full population
    activity matrix pooled over every trial and time point (not on
    class-averaged activity, so genuine trial-to-trial variability shapes
    the components). For visualization, individual-trial trajectories are
    then projected into that same space and averaged within each cue class.
    """
    delay_length = config.primary_delay if delay_length is None else delay_length
    batch = make_trial_batch(n_trials, config, rng, delay_lengths=(delay_length,))
    R = simulate_trials(bundle, batch, rng)  # (n_trials, N, T)
    n_trials_actual, N, T = R.shape

    pooled = R.transpose(1, 0, 2).reshape(N, n_trials_actual * T)  # (N, n_trials * T)
    mean_ = pooled.mean(axis=1, keepdims=True)
    pooled_centered = pooled - mean_

    U, S, _ = np.linalg.svd(pooled_centered, full_matrices=False)
    components = U[:, :n_components]  # (N, n_components)

    projected = np.einsum("nk,bnt->bkt", components, R - mean_.reshape(1, N, 1))

    n_classes = config.n_classes
    class_trajectories = np.zeros((n_classes, n_components, T))
    for c in range(n_classes):
        class_trajectories[c] = projected[batch.cue_labels == c].mean(axis=0)

    variance_explained = (S ** 2) / np.sum(S ** 2)

    # Dimensionality restricted to the delay epoch specifically (the
    # maintenance period the abstract's hypothesis is about), computed from
    # its own centering rather than reusing the whole-trial one.
    delay_columns = np.tile(batch.delay_mask[0], n_trials_actual)
    pooled_delay = pooled[:, delay_columns]
    pooled_delay_centered = pooled_delay - pooled_delay.mean(axis=1, keepdims=True)

    return {
        "components": components,
        "projected_trials": projected,
        "class_trajectories": class_trajectories,
        "variance_explained": variance_explained,
        "participation_ratio": _participation_ratio(pooled_centered),
        "participation_ratio_delay": _participation_ratio(pooled_delay_centered),
        "cue_labels": batch.cue_labels,
        "time_ms": batch.time_ms,
        "cue_mask": batch.cue_mask[0],
        "delay_mask": batch.delay_mask[0],
        "recall_mask": batch.recall_mask[0],
        "delay_length": delay_length,
        "label": bundle.label,
    }


# --------------------------------------------------------------------------
# Lesion-target ranking
# --------------------------------------------------------------------------

def delay_selectivity_fstat(
    config: ExperimentConfig,
    bundle: WeightBundle,
    rng: np.random.Generator,
    n_trials: int = 300,
    delay_length: Optional[float] = None,
) -> Array:
    """One-way ANOVA F-statistic per neuron on delay-averaged firing rate.

    High F means a neuron's mean activity during the memory delay strongly
    depends on which class was cued: a classic "memory neuron" signature,
    and a natural candidate criterion for targeted lesioning.
    """
    delay_length = config.primary_delay if delay_length is None else delay_length
    batch = make_trial_batch(n_trials, config, rng, delay_lengths=(delay_length,))
    R = simulate_trials(bundle, batch, rng)
    delay_mean = R[:, :, batch.delay_mask[0]].mean(axis=2)  # (n_trials, N)

    n_classes = config.n_classes
    grand_mean = delay_mean.mean(axis=0)
    ssb = np.zeros(config.N)
    ssw = np.zeros(config.N)
    for c in range(n_classes):
        group = delay_mean[batch.cue_labels == c]
        ssb += group.shape[0] * (group.mean(axis=0) - grand_mean) ** 2
        ssw += ((group - group.mean(axis=0)) ** 2).sum(axis=0)

    df_between = n_classes - 1
    df_within = max(n_trials - n_classes, 1)
    return (ssb / df_between) / (ssw / df_within + 1e-12)


def decoder_weight_magnitude(bundle: WeightBundle) -> Array:
    """L2 norm of each neuron's column in the readout: how heavily the
    linear decoder relies on that neuron."""
    return np.linalg.norm(bundle.Wout, axis=0)


def silencing_mask_from_ranking(scores: Array, fraction: float) -> Array:
    """Boolean mask silencing the top `fraction` of units by `scores` (highest first)."""
    N = len(scores)
    k = int(round(fraction * N))
    mask = np.zeros(N, dtype=bool)
    if k > 0:
        mask[np.argsort(scores)[-k:]] = True
    return mask


def random_silencing_mask(N: int, fraction: float, rng: np.random.Generator) -> Array:
    """Boolean mask silencing a uniformly random `fraction` of units."""
    k = int(round(fraction * N))
    mask = np.zeros(N, dtype=bool)
    if k > 0:
        mask[rng.choice(N, size=k, replace=False)] = True
    return mask


# --------------------------------------------------------------------------
# Perturbation sweeps
# --------------------------------------------------------------------------

def _eval_accuracy(
    config: ExperimentConfig,
    bundle: WeightBundle,
    rng: np.random.Generator,
    perturbation: Optional[Perturbation],
    n_trials: int,
    delay_length: float,
) -> float:
    batch = make_trial_batch(n_trials, config, rng, delay_lengths=(delay_length,))
    R = simulate_trials(bundle, batch, rng, perturbation=perturbation)
    return recall_accuracy(bundle, R, batch)


def noise_sweep(
    config: ExperimentConfig,
    bundle: WeightBundle,
    rng: np.random.Generator,
    severities,
    n_trials: int = 300,
    n_repeats: int = 3,
    delay_length: Optional[float] = None,
) -> dict:
    """Recall accuracy vs. additive Gaussian noise injected into the
    delay-phase drive (severity = noise standard deviation)."""
    delay_length = config.primary_delay if delay_length is None else delay_length
    means, stds = [], []
    for severity in severities:
        accs = [
            _eval_accuracy(
                config, bundle, rng, Perturbation(kind="noise", severity=severity),
                n_trials, delay_length,
            )
            for _ in range(n_repeats)
        ]
        means.append(np.mean(accs))
        stds.append(np.std(accs))
    return {
        "severities": np.asarray(severities, dtype=float),
        "accuracy_mean": np.asarray(means),
        "accuracy_std": np.asarray(stds),
        "label": bundle.label,
    }


def degradation_sweep(
    config: ExperimentConfig,
    bundle: WeightBundle,
    rng: np.random.Generator,
    severities,
    n_trials: int = 300,
    n_repeats: int = 3,
    delay_length: Optional[float] = None,
) -> dict:
    """Recall accuracy vs. structural noise added to J during the delay
    (severity = added-noise std, in units of the existing weight std)."""
    delay_length = config.primary_delay if delay_length is None else delay_length
    weight_scale = float(np.std(bundle.J))
    means, stds = [], []
    for severity in severities:
        accs = []
        for _ in range(n_repeats):
            J_eff = bundle.J + rng.normal(0.0, severity * weight_scale, size=bundle.J.shape)
            pert = Perturbation(kind="degrade", severity=severity, J_eff=J_eff)
            accs.append(_eval_accuracy(config, bundle, rng, pert, n_trials, delay_length))
        means.append(np.mean(accs))
        stds.append(np.std(accs))
    return {
        "severities": np.asarray(severities, dtype=float),
        "accuracy_mean": np.asarray(means),
        "accuracy_std": np.asarray(stds),
        "label": bundle.label,
    }


def silencing_sweep(
    config: ExperimentConfig,
    bundle: WeightBundle,
    rng: np.random.Generator,
    fractions,
    targeting: str,
    ranking_scores: Optional[Array] = None,
    n_trials: int = 300,
    n_repeats: int = 3,
    delay_length: Optional[float] = None,
) -> dict:
    """Recall accuracy vs. fraction of units silenced during the delay.

    ``targeting="random"`` draws a fresh random mask each repeat.
    ``targeting="targeted"`` always silences the top-scoring units from
    ``ranking_scores`` (e.g. `delay_selectivity_fstat` or
    `decoder_weight_magnitude`); repeats still resample test trials and
    initial-condition jitter, so they remain informative even though the
    silenced set itself is fixed.
    """
    if targeting not in ("random", "targeted"):
        raise ValueError("targeting must be 'random' or 'targeted'")
    if targeting == "targeted" and ranking_scores is None:
        raise ValueError("targeted silencing requires ranking_scores")

    delay_length = config.primary_delay if delay_length is None else delay_length
    means, stds = [], []
    for fraction in fractions:
        accs = []
        for _ in range(n_repeats):
            mask = (
                random_silencing_mask(config.N, fraction, rng)
                if targeting == "random"
                else silencing_mask_from_ranking(ranking_scores, fraction)
            )
            pert = Perturbation(kind="silence", severity=fraction, mask=mask)
            accs.append(_eval_accuracy(config, bundle, rng, pert, n_trials, delay_length))
        means.append(np.mean(accs))
        stds.append(np.std(accs))
    return {
        "fractions": np.asarray(fractions, dtype=float),
        "accuracy_mean": np.asarray(means),
        "accuracy_std": np.asarray(stds),
        "targeting": targeting,
        "label": bundle.label,
    }


def accuracy_vs_delay_length(
    config: ExperimentConfig,
    bundle: WeightBundle,
    rng: np.random.Generator,
    n_trials: int = 300,
) -> dict:
    """Baseline (unperturbed) recall accuracy at each scheduled delay length."""
    means = []
    for delay in config.delay_lengths:
        batch = make_trial_batch(n_trials, config, rng, delay_lengths=(delay,))
        R = simulate_trials(bundle, batch, rng)
        means.append(recall_accuracy(bundle, R, batch))
    return {
        "delay_lengths": np.asarray(config.delay_lengths, dtype=float),
        "accuracy": np.asarray(means),
        "label": bundle.label,
    }
