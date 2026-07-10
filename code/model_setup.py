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

    # Linear decoder
    decode_regularization: float = 1.0
    decode_include_bias: bool = True


@dataclass
class Model:
    """Model parameters plus generated weight matrices."""

    config: ModelConfig
    J: Array
    Jin: Array
    Jout: Array | None = None


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


def make_decode_targets(input_stream: Array, config: ModelConfig) -> Array:
    """Create one-hot decode targets, including class 0 for no input."""
    targets = np.zeros((config.nIn + 1, input_stream.size))
    targets[input_stream, np.arange(input_stream.size)] = 1.0
    return targets


def add_decoder_bias(firing_rates: Array) -> Array:
    """Append a constant bias row to decoder features."""
    bias = np.ones((1, firing_rates.shape[1]))
    return np.vstack((firing_rates, bias))


def fit_decoder(
    model: Model,
    firing_rates: Array,
    target_stream: Array,
    regularization: float | None = None,
) -> Array:
    """Fit a linear readout from firing rates to input classes."""
    config = model.config
    regularization = (
        config.decode_regularization if regularization is None else regularization
    )

    targets = make_decode_targets(target_stream, config)
    features = (
        add_decoder_bias(firing_rates)
        if config.decode_include_bias
        else firing_rates
    )

    gram = features @ features.T
    if regularization > 0:
        gram = gram + regularization * np.eye(gram.shape[0])

    decoder_weights = np.linalg.solve(gram, (targets @ features.T).T).T
    model.Jout = decoder_weights

    return decoder_weights


def decode(model: Model, firing_rates: Array) -> Array:
    """Decode input-class scores from firing rates."""
    if model.Jout is None:
        raise ValueError("Decoder has not been fit. Call fit_decoder first.")

    features = (
        add_decoder_bias(firing_rates)
        if model.config.decode_include_bias
        else firing_rates
    )
    return model.Jout @ features


def decoded_stream(decoded_scores: Array) -> Array:
    """Convert decoder scores into predicted input labels."""
    return np.argmax(decoded_scores, axis=0).astype(int)


def decode_accuracy(
    decoded_scores: Array,
    target_stream: Array,
    mask: Array | None = None,
) -> float:
    """Measure the fraction of correctly decoded labels."""
    predicted_stream = decoded_stream(decoded_scores)

    if mask is None:
        mask = np.ones(target_stream.shape, dtype=bool)

    if not np.any(mask):
        return float("nan")

    return float(np.mean(predicted_stream[mask] == target_stream[mask]))


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
