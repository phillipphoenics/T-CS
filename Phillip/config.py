"""Shared experiment configuration for the reservoir-vs-trained-RNN study.

Both network paradigms studied here — the Fixed Reservoir Network and the
Fully Trained RNN — must see the *identical* cue-delay-recall task and start
from the *identical* random architecture (Section "Model" of the abstract).
Keeping every shared hyperparameter in one dataclass makes that guarantee
explicit: both `reservoir_model.py` and `trained_rnn.py` are built from the
same `ExperimentConfig` instance, so any representational difference we later
measure is attributable to *training*, not to a difference in network size,
task timing, or initialization statistics.

This intentionally mirrors the `ModelConfig` pattern in `code/model_setup.py`
(a dataclass of hyperparameters plus a `create_weight_matrix` normalization
scheme), extended with the trial-structure fields a cue-delay-recall task
needs that the base project's continuous random stream did not.
"""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Hyperparameters shared by the reservoir and the trained RNN.

    Network
    -------
    N:
        Number of recurrent units. Chosen smaller than the base project's
        default (1000) so that the fully-trained RNN can be optimized with
        plain backpropagation-through-time on a laptop CPU in on the order of
        ten minutes for 1500 steps, rather than requiring a GPU, while still
        being far higher-dimensional than `n_classes` so a "collapse to a
        low-dimensional manifold" is a real, measurable compression rather
        than a foregone conclusion.
    g:
        Gain of the random recurrent matrix at initialization. Both
        paradigms start from this same value (see `init_shared_weights`);
        1.2 sits just past the classic edge-of-chaos point for a tanh
        network (Sompolinsky/Sussillo & Abbott), giving the un-trained
        reservoir rich, high-dimensional transient dynamics to work with.
    sp, gIn, spIn:
        Sparsity of the recurrent matrix, and gain/sparsity of the input
        projection. Same role as in `code/model_setup.py`.
    tau, dt:
        Membrane time constant and integration step, both in ms. `dt` is
        coarser than the base project's 0.1 ms (here 1.0 ms) purely to keep
        trial lengths (and therefore BPTT unroll length) computationally
        light; `tau=20` still spans 20 integration steps, which is enough to
        resolve the leaky dynamics.

    Task (cue-delay-recall)
    ------------------------
    n_classes:
        Number of distinct cue identities the network must remember.
    burn_in, cue_dur, recall_dur:
        Epoch durations in ms. `burn_in` is silence before the cue (lets the
        network settle from its initial condition); `cue_dur` is how long
        the one-hot cue is shown; `recall_dur` is how long the network is
        scored against the remembered identity once recall begins.
    delay_lengths:
        The grid of memory-delay durations (ms) trials are drawn from. This
        is the "varied delay lengths" axis in the abstract: both models are
        trained across this whole grid, and evaluated per-length to trace
        out a working-memory-capacity curve.
    go_pulse_dur:
        Duration (ms) of an explicit "go" input pulse at the start of the
        recall window (its own input channel, index `n_classes`). This
        signals *when* to report without requiring the network to also
        solve an elapsed-time-estimation problem, which is not the question
        under study here.

    Readout
    -------
    ridge_lambda:
        Ridge penalty for the reservoir's linear readout (see
        `reservoir_model.py`). Not used by the trained RNN, whose output
        weights are optimized jointly with everything else.

    Standing dynamical noise
    ------------------------
    intrinsic_noise_std:
        Gaussian noise added to the drive at *every* time step of *every*
        simulation (training, evaluation, and analysis alike) — a standing
        biological noise floor, distinct from the delay-only experimental
        perturbation. This matters for more than realism: the Fully Trained
        RNN is optimized against this same noisy simulator, so gradient
        descent only has a reason to prefer a noise-robust solution if noise
        was actually present during training. Without it, a fragile-but-
        accurate solution scores exactly as well as a robust one, and the
        robustness comparison this study makes would have no mechanism
        behind it. The delay-phase "additive recurrent noise" perturbation
        in `dynamics.Perturbation` adds *extra* noise on top of this
        baseline, specifically during the memory delay.
    """

    # --- network ---
    N: int = 300
    g: float = 1.2
    sp: float = 0.2
    gIn: float = 2.0
    spIn: float = 0.5
    tau: float = 20.0
    dt: float = 1.0

    # --- task / trial structure ---
    n_classes: int = 8
    burn_in: float = 5.0
    cue_dur: float = 10.0
    recall_dur: float = 15.0
    delay_lengths: tuple = (20.0, 80.0, 160.0, 300.0)
    go_pulse_dur: float = 5.0

    # --- reservoir readout ---
    ridge_lambda: float = 1.0

    # --- standing dynamical noise ---
    intrinsic_noise_std: float = 0.2

    @property
    def n_channels(self) -> int:
        """Input channels: one one-hot cue identity plus one go-pulse channel."""
        return self.n_classes + 1

    @property
    def primary_delay(self) -> float:
        """The longest scheduled delay: the main "deep dive" condition.

        Cross-temporal decoding, PCA trajectories, and perturbation sweeps
        are all run at this single delay length so that every trial in an
        analysis batch shares an identical timing/phase alignment (see
        `task.make_trial_batch`). The longest delay is also the most
        demanding maintenance condition, which is where representational
        differences between paradigms should be clearest.
        """
        return max(self.delay_lengths)
