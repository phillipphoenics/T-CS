"""Shared leaky-RNN dynamics, weight initialization, and perturbations.

Both paradigms in this study — the Fixed Reservoir Network and the Fully
Trained RNN — run through the exact same `simulate_trials` function once
their weights exist. This is deliberate: the only thing that should differ
between "reservoir" and "trained RNN" is which entries of `WeightBundle` were
shaped by gradient descent, never the simulation code itself. The recurrence
is the same Euler-discretized leaky integrator documented in
`assets/model_equations.md`:

    r(t+dt) = phi( u(t) + alpha * (r(t) - u(t)) ),   u(t) = J r(t) + Jin x(t)

with `phi = tanh` and `alpha = exp(-dt/tau)`.

Perturbations (additive noise, weight-matrix degradation, and unit
silencing) are injected only while a trial is inside its own delay epoch
(`TrialBatch.delay_mask`), so any change in recall accuracy can be
attributed specifically to disrupted *maintenance*, not to corrupted sensory
encoding or corrupted readout.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import ExperimentConfig
from task import TrialBatch

Array = np.ndarray


@dataclass
class WeightBundle:
    """Everything needed to simulate and read out one network instance."""

    config: ExperimentConfig
    J: Array          # (N, N) recurrent weights
    Jin: Array         # (N, n_channels) input weights
    Wout: Array        # (n_classes, N) readout weights
    bout: Array        # (n_classes,) readout bias
    label: str          # e.g. "Fixed Reservoir" or "Trained RNN"; used in plots


@dataclass
class Perturbation:
    """A delay-phase-only perturbation applied by `simulate_trials`.

    Exactly one mechanism is active per instance:

    kind="noise":
        Gaussian noise with std `severity` is added to the recurrent+input
        drive at every delay time step ("additive recurrent noise").
    kind="degrade":
        `J_eff` (a corrupted copy of J) replaces J for the drive computation
        at every delay time step ("weight matrix degradation").
    kind="silence":
        Every unit flagged `True` in `mask` (shape (N,)) is held at zero
        firing rate at every delay time step ("unit silencing" — the same
        boolean mask is used whether it was chosen randomly or by a
        targeting criterion; see `analysis.py`).
    """

    kind: str
    severity: float = 0.0
    J_eff: Optional[Array] = None
    mask: Optional[Array] = None


def create_weight_matrix(
    rows: int,
    columns: int,
    sparsity: float,
    gain: float,
    normalization_size: int,
    rng: np.random.Generator,
) -> Array:
    """Sparse random weight matrix with normalized variance.

    Identical scheme to `code/model_setup.py::create_weight_matrix`, just
    threaded through an explicit `Generator` instead of the numpy global
    random state, so weight initialization is reproducible independently of
    call order (important here since we draw many auxiliary random things —
    trial batches, perturbation instances — around the same model).
    """
    random_values = rng.normal(0.0, 1.0, size=(rows, columns))
    sparse_mask = rng.uniform(0.0, 1.0, size=(rows, columns)) <= sparsity
    return random_values * sparse_mask * gain / math.sqrt(normalization_size * sparsity)


def init_shared_weights(
    config: ExperimentConfig, rng: np.random.Generator
) -> tuple[Array, Array]:
    """Draw the random (J, Jin) initialization shared by BOTH paradigms.

    The reservoir keeps these forever and only trains a linear readout on
    top. The trained RNN starts from an identical copy of these same two
    matrices and then reshapes everything via gradient descent. Any
    representational difference measured later is therefore attributable to
    training, not to a different random architecture draw.
    """
    J = create_weight_matrix(
        config.N, config.N, config.sp, config.g, config.N, rng
    )
    Jin = create_weight_matrix(
        config.N, config.n_channels, config.spIn, config.gIn, config.n_channels, rng
    )
    return J, Jin


def simulate_trials(
    bundle: WeightBundle,
    batch: TrialBatch,
    rng: np.random.Generator,
    perturbation: Optional[Perturbation] = None,
) -> Array:
    """Integrate the network forward over one trial batch.

    Returns firing rates with shape (n_trials, N, T). Vectorized over the
    trial dimension; the only unavoidable python loop is over time, since the
    dynamics are recurrent.
    """
    config = bundle.config
    n_trials = batch.n_trials
    N = config.N
    T = batch.T
    alpha = math.exp(-config.dt / config.tau)

    R = np.empty((n_trials, N, T))
    R[:, :, 0] = rng.uniform(0.0, 0.1, size=(n_trials, N))

    apply_noise = perturbation is not None and perturbation.kind == "noise"
    apply_degrade = perturbation is not None and perturbation.kind == "degrade"
    apply_silence = perturbation is not None and perturbation.kind == "silence"
    J_delay = perturbation.J_eff if apply_degrade else bundle.J

    for t in range(T - 1):
        r_t = R[:, :, t]
        x_t = batch.inputs[:, :, t]
        in_delay = batch.delay_mask[:, t]

        # Clean update (used outside the delay epoch, and as the baseline
        # for trials not currently perturbed within a mixed-condition call).
        # A standing intrinsic noise floor is always present (see
        # `ExperimentConfig.intrinsic_noise_std`) so that "additive recurrent
        # noise" as a delay-phase perturbation means genuinely *extra* noise
        # on top of a baseline both paradigms were already trained under.
        drive = r_t @ bundle.J.T + x_t @ bundle.Jin.T
        drive = drive + rng.normal(0.0, config.intrinsic_noise_std, size=drive.shape)
        r_next = np.tanh(drive + (r_t - drive) * alpha)

        if perturbation is not None and np.any(in_delay):
            drive_p = r_t @ J_delay.T + x_t @ bundle.Jin.T
            drive_p = drive_p + rng.normal(0.0, config.intrinsic_noise_std, size=drive_p.shape)
            if apply_noise:
                drive_p = drive_p + rng.normal(0.0, perturbation.severity, size=drive_p.shape)
            r_next_p = np.tanh(drive_p + (r_t - drive_p) * alpha)
            if apply_silence:
                r_next_p = r_next_p.copy()
                r_next_p[:, perturbation.mask] = 0.0
            r_next = np.where(in_delay[:, None], r_next_p, r_next)

        R[:, :, t + 1] = r_next

    return R


def decode_output(bundle: WeightBundle, R: Array) -> Array:
    """Apply the linear readout at every time step.

    R has shape (n_trials, N, T); returns class scores (n_trials, n_classes, T).
    """
    return np.einsum("cj,njt->nct", bundle.Wout, R) + bundle.bout[None, :, None]


def predicted_labels(scores: Array) -> Array:
    """Argmax over the class axis. scores: (n_trials, n_classes, T) -> (n_trials, T)."""
    return np.argmax(scores, axis=1)


def recall_accuracy(
    bundle: WeightBundle, R: Array, batch: TrialBatch
) -> float:
    """Fraction of (trial, time step) pairs correctly classified within `recall_mask`."""
    scores = decode_output(bundle, R)
    predicted = predicted_labels(scores)
    correct = predicted == batch.cue_labels[:, None]
    mask = batch.recall_mask
    if not np.any(mask):
        return float("nan")
    return float(correct[mask].mean())


def recall_accuracy_per_trial(
    bundle: WeightBundle, R: Array, batch: TrialBatch
) -> Array:
    """Per-trial recall accuracy (mean over that trial's own recall window)."""
    scores = decode_output(bundle, R)
    predicted = predicted_labels(scores)
    correct = (predicted == batch.cue_labels[:, None]) & batch.recall_mask
    counts = batch.recall_mask.sum(axis=1)
    return correct.sum(axis=1) / np.maximum(counts, 1)
