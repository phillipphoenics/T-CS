import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass
class ModelConfig:
    """Configuration values for the recurrent neural network model.

    The model has three main parts:
    1. a recurrent reservoir of ``N`` neurons,
    2. a sparse input projection from ``nIn`` possible external inputs,
    3. an optional linear decoder that reads input labels from reservoir activity.
    
    Keeping these values in a dataclass makes it easy to run controlled
    experiments by changing only the configuration object.
    """

    # Recurrent pool
    # Number of neurons in the recurrent reservoir.
    N: int = 1000
    # Gain of the recurrent weight matrix. Values close to 1 vkeep activity rich
    # while usually avoiding unstable runaway dynamics.
    g: float = 0.95
    # Fraction of non-zero recurrent connections.
    sp: float = 0.25
    # Membrane/activity time constant in milliseconds.
    tau: float = 20.0
    # Simulation time step in milliseconds.
    dt: float = 0.1
    # Nonlinearity applied to recurrent activity after each integration step.
    nonlin: Callable[[Array], Array] = np.tanh

    # Input layer
    # Number of distinct non-zero input classes. Class 0 is reserved for silence.
    nIn: int = 20
    # Gain of the input weight matrix.
    gIn: float = 10.0
    # Fraction of non-zero input-to-reservoir connections.
    spIn: float = 0.05
    # Initial silent period before the first stimulus, in milliseconds.
    burnIn: float = 10.0
    # Duration of each presented stimulus, in milliseconds.
    durIn: float = 1.0
    # Inter-stimulus interval, in milliseconds.
    ISI: float = 0.0
    # Optional transformation of the one-hot input vector before projection.
    nonlinIn: Callable[[Array], Array] = lambda x: x

    # Linear decoder
    # Ridge penalty used when fitting the readout. This stabilizes the matrix
    # inverse when neurons are correlated or training data are limited.
    decode_regularization: float = 1.0
    # If true, append a constant feature so the decoder can learn class offsets.
    decode_include_bias: bool = True


@dataclass
class Model:
    """Model configuration together with generated weight matrices."""

    # Hyperparameters and simulation constants.
    config: ModelConfig
    # Recurrent reservoir weights with shape (N, N).
    J: Array
    # Input projection weights with shape (N, nIn).
    Jin: Array
    # Decoder/readout weights with shape (nIn + 1, N [+ bias]).
    # This is filled by fit_decoder and left as None before training.
    Jout: Array | None = None


def create_weight_matrix(
    rows: int,
    columns: int,
    sparsity: float,
    gain: float,
    normalization_size: int,
) -> Array:
    """Create a sparse random weight matrix with normalized variance.

    The random Gaussian matrix supplies the raw weights. The sparse mask then
    removes connections with probability ``1 - sparsity``. Finally, the scaling
    by ``sqrt(normalization_size * sparsity)`` keeps the expected input variance
    comparable when changing layer size or sparsity.
    """
    random_values = np.random.normal(0, 1, size=(rows, columns))
    sparse_mask = np.random.uniform(0, 1, size=(rows, columns)) <= sparsity
    return random_values * sparse_mask * gain / math.sqrt(normalization_size * sparsity)


def build_model(config: ModelConfig | None = None) -> Model:
    """Create the recurrent and input weight matrices for one model instance."""
    config = config or ModelConfig()

    # The recurrent matrix maps the reservoir state at time t back into the
    # reservoir. Its normalization uses N because each neuron receives from the
    # recurrent population.
    recurrent_weights = create_weight_matrix(
        rows=config.N,
        columns=config.N,
        sparsity=config.sp,
        gain=config.g,
        normalization_size=config.N,
    )
    # The input matrix maps the external one-hot input into the reservoir. Its
    # normalization uses nIn because each neuron receives from the input layer.
    input_weights = create_weight_matrix(
        rows=config.N,
        columns=config.nIn,
        sparsity=config.spIn,
        gain=config.gIn,
        normalization_size=config.nIn,
    )

    return Model(config=config, J=recurrent_weights, Jin=input_weights)


def make_decode_targets(input_stream: Array, config: ModelConfig) -> Array:
    """Create one-hot decoder targets, including class 0 for no input.

    ``input_stream`` stores one integer label per time step. The decoder is fit
    against a one-hot matrix because it predicts one score for every possible
    class, including the silent/no-input class.
    """
    targets = np.zeros((config.nIn + 1, input_stream.size))
    targets[input_stream, np.arange(input_stream.size)] = 1.0
    return targets


def add_decoder_bias(firing_rates: Array) -> Array:
    """Append a constant bias row to decoder features.

    Firing rates are stored as (neurons, time). Appending a row of ones gives
    the linear decoder an intercept term without changing the time dimension.
    """
    bias = np.ones((1, firing_rates.shape[1]))
    return np.vstack((firing_rates, bias))


def fit_decoder(
    model: Model,
    firing_rates: Array,
    target_stream: Array,
    regularization: float | None = None,
) -> Array:
    """Fit a ridge-regression readout from firing rates to input classes.

    The returned matrix maps reservoir features to class scores:

        decoded_scores = Jout @ features

    where ``features`` are either firing rates alone or firing rates plus a
    constant bias row. The same weights are also stored on ``model.Jout`` so
    ``decode`` can be called later on held-out simulations.
    """
    config = model.config
    regularization = (
        config.decode_regularization if regularization is None else regularization
    )

    # Convert integer labels into one-hot rows. This gives the least-squares
    # problem one target row per class.
    targets = make_decode_targets(target_stream, config)
    features = (
        add_decoder_bias(firing_rates)
        if config.decode_include_bias
        else firing_rates
    )

    # Ridge-regression normal equation:
    # Jout = targets @ features.T @ inv(features @ features.T + lambda * I)
    # The solve below avoids explicitly forming an inverse, which is numerically
    # better and usually faster.
    gram = features @ features.T
    if regularization > 0:
        gram = gram + regularization * np.eye(gram.shape[0])

    decoder_weights = np.linalg.solve(gram, (targets @ features.T).T).T
    model.Jout = decoder_weights

    return decoder_weights


def decode(model: Model, firing_rates: Array) -> Array:
    """Decode input-class scores from firing rates using the fitted readout."""
    if model.Jout is None:
        raise ValueError("Decoder has not been fit. Call fit_decoder first.")

    # Match the feature layout used during training. If a bias row was included
    # when fitting, it must also be included when decoding new activity.
    features = (
        add_decoder_bias(firing_rates)
        if model.config.decode_include_bias
        else firing_rates
    )
    return model.Jout @ features


def decoded_stream(decoded_scores: Array) -> Array:
    """Convert decoder scores into predicted input labels.

    The winning class is the row with the largest score at each time step.
    """
    return np.argmax(decoded_scores, axis=0).astype(int)


def decode_accuracy(
    decoded_scores: Array,
    target_stream: Array,
    mask: Array | None = None,
) -> float:
    """Measure the fraction of correctly decoded labels.

    ``mask`` can be used to evaluate only selected time points, for example
    periods where an input is active (``target_stream > 0``). If the mask
    excludes every time point, the function returns NaN instead of raising.
    """
    predicted_stream = decoded_stream(decoded_scores)

    if mask is None:
        mask = np.ones(target_stream.shape, dtype=bool)

    if not np.any(mask):
        return float("nan")

    return float(np.mean(predicted_stream[mask] == target_stream[mask]))


def step(firing_rates: Array, input_layer: Array, model: Model) -> Array:
    """Advance the firing rates by one Euler-style integration step.

    ``firing_rates`` is the reservoir state at one time point and ``input_layer``
    is the external input vector at the same time point. The update mixes the old
    state with the current recurrent/input drive according to the time constant,
    then applies the reservoir nonlinearity.
    """
    config = model.config
    # Exact decay factor for a first-order low-pass process over one time step.
    timestep = math.exp(-config.dt / config.tau)

    # Total synaptic drive from recurrent activity plus external input.
    input_drive = model.J @ firing_rates + model.Jin @ config.nonlinIn(input_layer)
    # Relax the old rates toward the current drive and squash them through the
    # nonlinear activation function.
    updated_rates = config.nonlin(input_drive + (firing_rates - input_drive) * timestep)

    return updated_rates


def make_input(sequence_length: int, config: ModelConfig) -> tuple[Array, Array]:
    """Generate a random input sequence as one-hot and integer streams.

    The integer stream contains class labels at every time step. Label 0 means
    no input. The returned one-hot matrix excludes row 0 because only the active
    input channels are projected into the reservoir.
    """
    burn_in_steps = int(config.burnIn / config.dt)
    isi_steps = int(config.ISI / config.dt)
    input_duration_steps = int(config.durIn / config.dt)

    # Begin with silence so the reservoir has time to settle before stimuli.
    input_stream = [0] * burn_in_steps

    for _ in range(sequence_length):
        # Input labels are 1..nIn because 0 is reserved for silence.
        value = np.random.randint(0, config.nIn) + 1
        input_stream.extend([0] * isi_steps)
        input_stream.extend([value] * input_duration_steps)

    input_stream = np.array(input_stream, dtype=int)

    # Build a full one-hot representation with a row for silence, then drop the
    # silence row before returning because the input layer has only nIn channels.
    onehot = np.zeros((config.nIn + 1, input_stream.size))
    onehot[input_stream, np.arange(input_stream.size)] = 1.0

    return onehot[1:, :], input_stream
