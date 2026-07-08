import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass
class ModelConfig:
    """Configuration for the recurrent neural network model."""

    # Recurrent pool
    N: int = 1000
    g: float = 0.95
    sp: float = 0.25
    tau: float = 20.0
    dt: float = 0.1
    nonlin: Callable[[Array], Array] = np.tanh

    # Input layer
    nIn: int = 20
    gIn: float = 10.0
    spIn: float = 0.05
    burnIn: float = 10.0
    durIn: float = 1.0
    ISI: float = 0.0
    nonlinIn: Callable[[Array], Array] = lambda x: x


@dataclass
class Model:
    """Model parameters plus generated weight matrices."""

    config: ModelConfig
    J: Array
    Jin: Array


def create_weight_matrix(
    rows: int,
    columns: int,
    sparsity: float,
    gain: float,
    normalization_size: int,
) -> Array:
    """Create a sparse random weight matrix with normalized variance."""
    random_values = np.random.normal(0, 1, size=(rows, columns))
    sparse_mask = np.random.uniform(0, 1, size=(rows, columns)) <= sparsity
    return random_values * sparse_mask * gain / math.sqrt(normalization_size * sparsity)


def build_model(config: ModelConfig | None = None) -> Model:
    """Create the recurrent and input weight matrices."""
    config = config or ModelConfig()

    recurrent_weights = create_weight_matrix(
        rows=config.N,
        columns=config.N,
        sparsity=config.sp,
        gain=config.g,
        normalization_size=config.N,
    )
    input_weights = create_weight_matrix(
        rows=config.N,
        columns=config.nIn,
        sparsity=config.spIn,
        gain=config.gIn,
        normalization_size=config.nIn,
    )

    return Model(config=config, J=recurrent_weights, Jin=input_weights)


def step(firing_rates: Array, input_layer: Array, model: Model) -> Array:
    """Advance the firing rates by one Euler integration step."""
    config = model.config
    timestep = math.exp(-config.dt / config.tau)

    input_drive = model.J @ firing_rates + model.Jin @ config.nonlinIn(input_layer)
    updated_rates = config.nonlin(input_drive + (firing_rates - input_drive) * timestep)

    return updated_rates


def make_input(sequence_length: int, config: ModelConfig) -> tuple[Array, Array]:
    """Generate a random input sequence as one-hot and integer streams."""
    burn_in_steps = int(config.burnIn / config.dt)
    isi_steps = int(config.ISI / config.dt)
    input_duration_steps = int(config.durIn / config.dt)

    input_stream = [0] * burn_in_steps

    for _ in range(sequence_length):
        value = np.random.randint(0, config.nIn) + 1
        input_stream.extend([0] * isi_steps)
        input_stream.extend([value] * input_duration_steps)

    input_stream = np.array(input_stream, dtype=int)

    onehot = np.zeros((config.nIn + 1, input_stream.size))
    onehot[input_stream, np.arange(input_stream.size)] = 1.0

    return onehot[1:, :], input_stream
