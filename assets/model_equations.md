# Recurrent Network Model Equations

This note describes the equations implemented in `code/model_setup.py`.

## State Variables

The recurrent network contains \(N\) neurons. At time \(t\), their firing rates
are collected in the vector

$$
r(t) \in \mathbb{R}^N.
$$

The input layer contains \(n_\text{in}\) units. The input vector is

$$
x(t) \in \mathbb{R}^{n_\text{in}}.
$$

In the current implementation, the input is one-hot encoded: one input unit is
active at a time, while the others are zero.

## Input Drive

The total drive into the recurrent population is

$$
u(t)
=
J r(t)
+
J_\text{in}\,\phi_\text{in}(x(t)).
$$

Here:

- \(J \in \mathbb{R}^{N \times N}\) is the recurrent weight matrix.
- \(J_\text{in} \in \mathbb{R}^{N \times n_\text{in}}\) is the input weight matrix.
- \(\phi_\text{in}\) is the input nonlinearity. In the current model,
  \(\phi_\text{in}(x) = x\).

For a single neuron \(i\), the same equation is

$$
u_i(t)
=
\sum_{j=1}^{N} J_{ij} r_j(t)
+
\sum_{k=1}^{n_\text{in}} J_{\text{in},ik}\,\phi_\text{in}(x_k(t)).
$$

The first term is the recurrent contribution from the network itself. The
second term is the external input contribution.

## Time Update

The code computes

$$
\alpha
=
\exp\left(-\frac{dt}{\tau}\right),
$$

where \(dt\) is the simulation time step and \(\tau\) is the neural time
constant.

The firing rates are then updated as

$$
r(t + dt)
=
\phi\left(
u(t) + \alpha\,[r(t) - u(t)]
\right).
$$

Equivalently,

$$
r(t + dt)
=
\phi\left(
\alpha r(t) + (1-\alpha)u(t)
\right).
$$

In the current implementation,

$$
\phi(z) = \tanh(z).
$$

This means that the next state is a mixture of the old state \(r(t)\) and the
new drive \(u(t)\). The parameter \(\alpha\) controls the balance:

- If \(\alpha\) is close to 1, the state changes slowly and retains memory.
- If \(\alpha\) is close to 0, the state rapidly follows the new drive.

Because \(\tanh(z)\) is bounded between \(-1\) and \(1\), it prevents firing
rates from growing without bound.

## Weight Matrix Scaling

The recurrent weight matrix is generated as a sparse random matrix and scaled
approximately as

$$
J_{ij}
\sim
\frac{g}{\sqrt{N \cdot \text{sparsity}}}.
$$

The purpose of this normalization is to keep the overall scale of recurrent
input comparable when the network size or sparsity changes.

The parameter \(g\) controls the effective strength of the recurrent dynamics.
Larger \(g\) usually makes the network activity more persistent and potentially
less stable.

## Eigenvalue Spectrum of \(J\)

An eigenvalue \(\lambda\) and eigenvector \(v\) of \(J\) satisfy

$$
Jv = \lambda v.
$$

The eigenvalue spectrum is the set of all eigenvalues of \(J\). Because \(J\)
is generally not symmetric, its eigenvalues can be complex:

$$
\lambda = a + bi.
$$

The magnitude of an eigenvalue is

$$
|\lambda|
=
\sqrt{
\operatorname{Re}(\lambda)^2
+
\operatorname{Im}(\lambda)^2
}.
$$

The spectrum tells us how the recurrent matrix tends to amplify, damp, or rotate
activity patterns in the network.

## Interpretation of the Unit Circle

In the spectrum plot, the unit circle marks

$$
|\lambda| = 1.
$$

Roughly:

- Eigenvalues inside the unit circle correspond to modes that tend to decay.
- Eigenvalues near the unit circle correspond to modes that can persist for a
  longer time.
- Eigenvalues outside the unit circle can correspond to unstable or strongly
  amplified dynamics.

With \(g = 0.95\), the random recurrent matrix typically has a spectrum with a
radius close to \(0.95\). This places the network near, but usually inside, the
stability boundary. That is useful for short-term memory: inputs can leave a
temporary trace in the network state without causing uncontrolled growth.

