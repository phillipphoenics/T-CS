# Recurrent Network Model Equations

This note describes the equations implemented in `code/model_setup.py`.

## State Variables

The recurrent network contains $N$ neurons. At time $t$, their firing rates are
collected in the vector:

```math
r(t) \in \mathbb{R}^N
```

The input layer contains $n_\text{in}$ units. The input vector is:

```math
x(t) \in \mathbb{R}^{n_\text{in}}
```

In the current implementation, the input is one-hot encoded: one input unit is
active at a time, while the others are zero.

## Input Drive

The total drive into the recurrent population is:

```math
u(t)
=
J r(t)
+
J_\text{in}\,\phi_\text{in}(x(t))
```

Here:

- $J \in \mathbb{R}^{N \times N}$ is the recurrent weight matrix.
- $J_\text{in} \in \mathbb{R}^{N \times n_\text{in}}$ is the input weight matrix.
- $\phi_\text{in}$ is the input nonlinearity. In the current model,
  $\phi_\text{in}(x) = x$.

For a single neuron $i$, the same equation is:

```math
u_i(t)
=
\sum_{j=1}^{N} J_{ij} r_j(t)
+
\sum_{k=1}^{n_\text{in}} J_{\text{in},ik}\,\phi_\text{in}(x_k(t))
```

The first term is the recurrent contribution from the network itself. The
second term is the external input contribution.

## Time Update

The code computes:

```math
\alpha
=
\exp\left(-\frac{dt}{\tau}\right)
```

where $dt$ is the simulation time step and $\tau$ is the neural time constant.

The firing rates are then updated as:

```math
r(t + dt)
=
\phi\left(
u(t) + \alpha\,[r(t) - u(t)]
\right)
```

Equivalently:

```math
r(t + dt)
=
\phi\left(
\alpha r(t) + (1-\alpha)u(t)
\right)
```

In the current implementation:

```math
\phi(z) = \tanh(z)
```

This means that the next state is a mixture of the old state $r(t)$ and the new
drive $u(t)$. The parameter $\alpha$ controls the balance:

- If $\alpha$ is close to 1, the state changes slowly and retains memory.
- If $\alpha$ is close to 0, the state rapidly follows the new drive.

Because $\tanh(z)$ is bounded between $-1$ and $1$, it prevents firing rates
from growing without bound.

## Weight Matrix Scaling

The recurrent weight matrix is generated as a sparse random matrix and scaled
approximately as:

```math
J_{ij}
\sim
\frac{g}{\sqrt{N \cdot \text{sparsity}}}
```

The purpose of this normalization is to keep the overall scale of recurrent
input comparable when the network size or sparsity changes.

The parameter $g$ controls the effective strength of the recurrent dynamics.
Larger $g$ usually makes the network activity more persistent and potentially
less stable.

## Linear Decoder

The model can fit a linear decoder from recurrent activity back to the input
class. The decoder target includes the silent class $0$, so:

```math
y(t) \in \mathbb{R}^{n_\text{in} + 1}
```

where $y(t)$ is one-hot encoded from the integer input stream.

The decoder uses the recurrent firing rates as features. If a bias is enabled,
the feature vector is:

```math
\tilde r(t)
=
\begin{bmatrix}
r(t) \\
1
\end{bmatrix}
```

The fitted decoder matrix is:

```math
J_\text{out}
=
Y \tilde R^\top
\left(
\tilde R \tilde R^\top + \lambda I
\right)^{-1}
```

Here, $\tilde R$ contains decoder features across time, $Y$ contains the
one-hot target vectors, and $\lambda$ is the ridge regularization strength.

At decode time, class scores are computed as:

```math
\hat y(t)
=
J_\text{out}\tilde r(t)
```

The predicted input class is the index with the largest score:

```math
\hat c(t)
=
\operatorname*{arg\,max}_k \hat y_k(t)
```

Decode performance is measured as the fraction of time points where
$\hat c(t)$ matches the integer input stream. The runner reports both overall
accuracy and accuracy restricted to time points where the input is active.

## Eigenvalue Spectrum of $J$

An eigenvalue $\lambda$ and eigenvector $v$ of $J$ satisfy:

```math
Jv = \lambda v
```

The eigenvalue spectrum is the set of all eigenvalues of $J$. Because $J$ is
generally not symmetric, its eigenvalues can be complex:

```math
\lambda = a + bi
```

The magnitude of an eigenvalue is:

```math
|\lambda|
=
\sqrt{
\operatorname{Re}(\lambda)^2
+
\operatorname{Im}(\lambda)^2
}
```

The spectrum tells us how the recurrent matrix tends to amplify, damp, or rotate
activity patterns in the network.

## Interpretation of the Unit Circle

In the spectrum plot, the unit circle marks:

```math
|\lambda| = 1
```

Roughly:

- Eigenvalues inside the unit circle correspond to modes that tend to decay.
- Eigenvalues near the unit circle correspond to modes that can persist for a
  longer time.
- Eigenvalues outside the unit circle can correspond to unstable or strongly
  amplified dynamics.

With $g = 0.95$, the random recurrent matrix typically has a spectrum with a
radius close to $0.95$. This places the network near, but usually inside, the
stability boundary. That is useful for short-term memory: inputs can leave a
temporary trace in the network state without causing uncontrolled growth.
