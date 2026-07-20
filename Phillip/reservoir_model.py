"""Fixed Reservoir Network paradigm (Jaeger, 2001).

The recurrent connectivity (J, Jin) is drawn once and never touched again;
only a linear readout is fit on top of it, in closed form via ridge
regression. This is the same readout scheme as
`code/model_setup.py::fit_decoder`, adapted from "decode the current input
class at every time step of a continuous stream" to "decode the cued class
at every recall-window time step of a cue-delay-recall trial".
"""

import numpy as np

from config import ExperimentConfig
from dynamics import WeightBundle, simulate_trials
from task import make_trial_batch

Array = np.ndarray


def fit_readout(
    config: ExperimentConfig,
    firing_rates: Array,
    targets: Array,
    recall_mask: Array,
    regularization: float | None = None,
) -> tuple[Array, Array]:
    """Ridge-regression readout fit, pooling every (trial, time) sample
    inside `recall_mask` into one least-squares problem.

    ``firing_rates``: (n_trials, N, T). ``targets``: (n_trials, n_classes, T),
    one-hot. ``recall_mask``: (n_trials, T) bool. Returns ``(Wout, bout)``
    with shapes ``(n_classes, N)`` and ``(n_classes,)``.
    """
    regularization = config.ridge_lambda if regularization is None else regularization
    N = firing_rates.shape[1]

    # Boolean-index the (trial, time) leading dims to pool every scored
    # sample into one feature matrix, regardless of each trial's own
    # (possibly ragged) recall-window length.
    features = firing_rates.transpose(0, 2, 1)[recall_mask]        # (n_samples, N)
    target_samples = targets.transpose(0, 2, 1)[recall_mask]        # (n_samples, n_classes)

    # Constant bias feature, matching add_decoder_bias's role in the base project.
    features_b = np.hstack([features, np.ones((features.shape[0], 1))])

    gram = features_b.T @ features_b
    gram[np.diag_indices_from(gram)] += regularization
    weights = np.linalg.solve(gram, features_b.T @ target_samples)   # (N + 1, n_classes)

    Wout = weights[:N, :].T
    bout = weights[N, :]
    return Wout, bout


def build_reservoir(
    config: ExperimentConfig,
    J0: Array,
    Jin0: Array,
    rng: np.random.Generator,
    n_train_trials: int = 600,
) -> WeightBundle:
    """Package the shared random initialization as a fixed reservoir and fit
    its readout.

    ``J0``/``Jin0`` must be the same arrays passed to `trained_rnn.train_trained_rnn`
    for this comparison, so the two paradigms start from an identical
    architecture (see `dynamics.init_shared_weights`). Readout training uses
    trials with mixed delay lengths (the full grid in ``config.delay_lengths``),
    so the single fitted readout has to work across every delay duration —
    mirroring how the trained RNN is optimized in `trained_rnn.py`.
    """
    bundle = WeightBundle(
        config=config,
        J=J0,
        Jin=Jin0,
        Wout=np.zeros((config.n_classes, config.N)),
        bout=np.zeros(config.n_classes),
        label="Fixed Reservoir",
    )

    train_batch = make_trial_batch(n_train_trials, config, rng)
    firing_rates = simulate_trials(bundle, train_batch, rng)
    Wout, bout = fit_readout(config, firing_rates, train_batch.targets, train_batch.recall_mask)

    bundle.Wout = Wout
    bundle.bout = bout
    return bundle
