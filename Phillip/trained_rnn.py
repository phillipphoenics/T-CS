"""Fully Trained RNN paradigm: input, recurrent, and output weights are all
optimized end-to-end via gradient descent (backpropagation through time).

`LeakyRNN` below is a deliberate, literal reimplementation of the same
recurrence used by `dynamics.simulate_trials` (see that module's docstring
for the equation), so the trained network's forward dynamics are provably
identical to the reservoir's up to which weights gradients are allowed to
touch. Training starts from the *same* random `(J0, Jin0)` draw the
reservoir uses (`dynamics.init_shared_weights`, called once by the caller
and passed to both paradigms) — the only thing that can differ afterwards is
what gradient descent does to those weights.

PyTorch's autograd is used only for this training procedure. Once training
finishes, the learned weights are exported back to plain numpy arrays and
packaged into the same `WeightBundle` used by the reservoir, so every
downstream analysis (`analysis.py`) runs through the identical numpy
`simulate_trials` regardless of which paradigm produced the weights.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig
from dynamics import WeightBundle
from task import make_trial_batch

Array = np.ndarray


@dataclass
class TrainingConfig:
    """Hyperparameters of the optimization procedure itself.

    Kept separate from `ExperimentConfig`, which only holds the shared
    architecture/task substrate both paradigms must agree on. These values
    are specific to *how* the trained RNN is fit and have no reservoir
    counterpart.
    """

    n_steps: int = 1500
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    n_eval_trials: int = 300
    eval_every: int = 100
    torch_seed: int = 0


class LeakyRNN(nn.Module):
    """Trainable leaky-RNN cell plus linear readout, matching `dynamics.py`."""

    def __init__(self, config: ExperimentConfig, J0: Array, Jin0: Array):
        super().__init__()
        self.config = config
        self.alpha = math.exp(-config.dt / config.tau)
        self.J = nn.Parameter(torch.tensor(J0, dtype=torch.float32))
        self.Jin = nn.Parameter(torch.tensor(Jin0, dtype=torch.float32))
        self.Wout = nn.Parameter(torch.empty(config.n_classes, config.N))
        self.bout = nn.Parameter(torch.zeros(config.n_classes))
        nn.init.xavier_uniform_(self.Wout)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``inputs``: (n_trials, n_channels, T).

        Returns ``(rates, scores)`` with shapes ``(n_trials, N, T)`` and
        ``(n_trials, n_classes, T)``.
        """
        n_trials, _, T = inputs.shape
        r = torch.rand(n_trials, self.config.N) * 0.1
        rates = []
        for t in range(T):
            x_t = inputs[:, :, t]
            drive = r @ self.J.T + x_t @ self.Jin.T
            # Same standing noise floor as `dynamics.simulate_trials`
            # (`ExperimentConfig.intrinsic_noise_std`): training under this
            # noise is what gives gradient descent an actual reason to prefer
            # a noise-robust solution over a fragile-but-accurate one.
            if self.config.intrinsic_noise_std:
                drive = drive + torch.randn_like(drive) * self.config.intrinsic_noise_std
            r = torch.tanh(drive + (r - drive) * self.alpha)
            rates.append(r)
        rates = torch.stack(rates, dim=2)
        scores = torch.einsum("cn,bnt->bct", self.Wout, rates) + self.bout[None, :, None]
        return rates, scores


def _masked_recall_accuracy(scores: torch.Tensor, cue_labels: Array, recall_mask: Array) -> float:
    predicted = scores.argmax(dim=1).numpy()
    correct = (predicted == cue_labels[:, None]) & recall_mask
    return float(correct.sum() / max(recall_mask.sum(), 1))


def train_trained_rnn(
    config: ExperimentConfig,
    J0: Array,
    Jin0: Array,
    rng: np.random.Generator,
    training: TrainingConfig | None = None,
) -> tuple[WeightBundle, dict]:
    """Train a `LeakyRNN` end-to-end and export it as a numpy `WeightBundle`.

    The loss is the cross-entropy between the readout scores and the cued
    class, averaged over every (trial, time step) pair inside that trial's
    own recall window (``batch.recall_mask``) — the network receives no
    direct supervision during the cue or delay epochs, so any useful
    delay-period dynamics must emerge purely from gradients that had to flow
    backward through the delay to make recall possible at all. Every
    training batch is drawn fresh with mixed delay lengths from
    ``config.delay_lengths``, so the network is pushed to maintain the cue
    across the *whole* delay grid, not just one duration.
    """
    training = training or TrainingConfig()
    torch.manual_seed(training.torch_seed)

    model = LeakyRNN(config, J0, Jin0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )

    history = {"step": [], "loss": [], "eval_accuracy": []}

    for step in range(1, training.n_steps + 1):
        batch = make_trial_batch(training.batch_size, config, rng)
        inputs = torch.tensor(batch.inputs, dtype=torch.float32)
        cue_labels = torch.tensor(batch.cue_labels, dtype=torch.long)
        recall_mask = torch.tensor(batch.recall_mask)

        _, scores = model(inputs)
        target_bt = cue_labels[:, None].expand(-1, scores.shape[-1])
        loss_bt = F.cross_entropy(scores, target_bt, reduction="none")
        loss = (loss_bt * recall_mask).sum() / recall_mask.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip_norm)
        optimizer.step()

        if step % training.eval_every == 0 or step == training.n_steps:
            with torch.no_grad():
                eval_batch = make_trial_batch(training.n_eval_trials, config, rng)
                eval_inputs = torch.tensor(eval_batch.inputs, dtype=torch.float32)
                _, eval_scores = model(eval_inputs)
                acc = _masked_recall_accuracy(
                    eval_scores, eval_batch.cue_labels, eval_batch.recall_mask
                )
            history["step"].append(step)
            history["loss"].append(float(loss.item()))
            history["eval_accuracy"].append(acc)

    bundle = WeightBundle(
        config=config,
        J=model.J.detach().numpy().copy(),
        Jin=model.Jin.detach().numpy().copy(),
        Wout=model.Wout.detach().numpy().copy(),
        bout=model.bout.detach().numpy().copy(),
        label="Trained RNN",
    )
    return bundle, history
