"""Cue-delay-recall trial generation, shared by both network paradigms.

A trial has four epochs, laid out along the time axis:

    burn-in | cue | delay (variable length) | recall

During `cue`, one of `config.n_classes` input channels is driven high
(one-hot) to indicate which item must be remembered. During `delay`, all
inputs are silent: the network must carry the cue identity through this gap
using only its own recurrent dynamics. `recall` opens with a brief pulse on a
dedicated "go" channel, after which the network is scored against the
originally cued identity until the end of the trial.

Two ways to build a batch:

- ``delay_lengths`` with more than one value (the default, `config.delay_lengths`)
  draws an i.i.d. delay per trial from that grid. Use this for *training*, so
  the network is pushed to generalize across delay durations.
- ``delay_lengths=(x,)`` (a single value) gives every trial in the batch the
  same delay. Use this for *analysis* (cross-temporal decoding, PCA,
  perturbation sweeps): those methods compare activity across trials at a
  shared absolute time index, which is only meaningful if every trial in the
  batch shares the same epoch boundaries.
"""

import math
from dataclasses import dataclass

import numpy as np

from config import ExperimentConfig

Array = np.ndarray


def to_steps(duration_ms: float, dt: float) -> int:
    """Convert a duration in milliseconds to a (minimum 1) integer step count."""
    return max(1, int(round(duration_ms / dt)))


@dataclass
class TrialBatch:
    """A batch of cue-delay-recall trials sharing one time axis of length `T`.

    Shapes
    ------
    inputs:
        (n_trials, n_channels, T) — one-hot cue channels plus the go channel.
    targets:
        (n_trials, n_classes, T) — one-hot cued identity, held from recall
        onset through the end of the trial; all zero beforehand.
    cue_labels:
        (n_trials,) int — the cued class index for each trial.
    delay_length_ms:
        (n_trials,) float — the actual delay drawn for each trial (ms).
    cue_mask, delay_mask, recall_mask:
        (n_trials, T) bool — which time steps belong to which epoch, *per
        trial* (delay length can vary trial-to-trial even though `T` is
        shared, via padding; see module docstring).
    """

    inputs: Array
    targets: Array
    cue_labels: Array
    delay_length_ms: Array
    cue_mask: Array
    delay_mask: Array
    recall_mask: Array
    T: int
    dt: float

    @property
    def n_trials(self) -> int:
        return self.inputs.shape[0]

    @property
    def time_ms(self) -> Array:
        """Time axis in ms, with t=0 at burn-in start (matches `plotting.py`)."""
        return np.arange(self.T) * self.dt


def make_trial_batch(
    n_trials: int,
    config: ExperimentConfig,
    rng: np.random.Generator,
    delay_lengths: tuple | None = None,
) -> TrialBatch:
    """Generate a batch of cue-delay-recall trials.

    See the module docstring for when to pass a single-value `delay_lengths`
    (analysis) versus the full grid (training).
    """
    delay_lengths = config.delay_lengths if delay_lengths is None else delay_lengths
    delay_choices = np.asarray(delay_lengths, dtype=float)

    n_classes = config.n_classes
    n_channels = config.n_channels
    go_channel = n_classes

    burn_in_steps = to_steps(config.burn_in, config.dt)
    cue_steps = to_steps(config.cue_dur, config.dt)
    recall_steps = to_steps(config.recall_dur, config.dt)
    go_steps = min(to_steps(config.go_pulse_dur, config.dt), recall_steps)
    max_delay_steps = to_steps(float(delay_choices.max()), config.dt)

    T = burn_in_steps + cue_steps + max_delay_steps + recall_steps

    inputs = np.zeros((n_trials, n_channels, T))
    targets = np.zeros((n_trials, n_classes, T))
    cue_mask = np.zeros((n_trials, T), dtype=bool)
    delay_mask = np.zeros((n_trials, T), dtype=bool)
    recall_mask = np.zeros((n_trials, T), dtype=bool)

    cue_labels = rng.integers(0, n_classes, size=n_trials)
    delay_length_ms = rng.choice(delay_choices, size=n_trials)

    cue_start = burn_in_steps
    cue_end = cue_start + cue_steps

    for i in range(n_trials):
        delay_steps_i = to_steps(delay_length_ms[i], config.dt)
        delay_end = cue_end + delay_steps_i

        inputs[i, cue_labels[i], cue_start:cue_end] = 1.0
        inputs[i, go_channel, delay_end:delay_end + go_steps] = 1.0

        # Score from this trial's own recall onset through the shared end of
        # the batch. Trials with a shorter delay than the batch maximum
        # simply get a longer hold window; every trial is guaranteed at
        # least `recall_steps` of scored recall.
        targets[i, cue_labels[i], delay_end:T] = 1.0

        cue_mask[i, cue_start:cue_end] = True
        delay_mask[i, cue_end:delay_end] = True
        recall_mask[i, delay_end:T] = True

    return TrialBatch(
        inputs=inputs,
        targets=targets,
        cue_labels=cue_labels,
        delay_length_ms=delay_length_ms,
        cue_mask=cue_mask,
        delay_mask=delay_mask,
        recall_mask=recall_mask,
        T=T,
        dt=config.dt,
    )
