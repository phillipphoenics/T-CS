# How Network Architecture and Training Shape Working Memory Representations and Perturbation Robustness

This folder implements the full study described by the abstract of the same
name: a comparison of a **Fixed Reservoir Network** (Jaeger, 2001) against a
**Fully Trained RNN**, both solving an identical cue-delay-recall working
memory task across varied delay lengths, analyzed with cross-temporal
decoding, PCA of population trajectories, and four delay-phase perturbations.

Everything here is self-contained to this folder, per the project convention
of one participant folder per person building on the shared base code in
`../code/`.

## Files

| File | Contents |
|---|---|
| `config.py` | `ExperimentConfig` — every hyperparameter shared by *both* paradigms (network size, task timing, delay grid, standing noise floor), so both are guaranteed to see an identical task and start from an identical architecture. |
| `task.py` | Cue-delay-recall trial generation (`make_trial_batch`), with variable per-trial delay length. |
| `dynamics.py` | The shared leaky-RNN recurrence both paradigms run (`simulate_trials`), weight initialization (`init_shared_weights`), and the delay-phase-only perturbation mechanics (`Perturbation`: additive noise, weight degradation, unit silencing). |
| `reservoir_model.py` | Builds the Fixed Reservoir Network and fits its linear (ridge-regression) readout. |
| `trained_rnn.py` | Builds and trains the Fully Trained RNN end-to-end via PyTorch/backpropagation-through-time, then exports the learned weights back to plain numpy so every downstream analysis is paradigm-agnostic. |
| `analysis.py` | Cross-temporal ("temporal generalization") decoding, PCA/dimensionality (participation ratio), lesion-target ranking (task-selectivity F-statistic, decoder-weight magnitude), and the four perturbation sweeps. |
| `plotting.py` | Every figure-generating function. |
| `run_experiment.py` | Standalone, reproducible script: builds both models, runs the full analysis suite, saves every figure to `plots/` and a numeric summary to `results/summary.json`. |
| `working_memory_project.ipynb` | The narrated version of the same pipeline, with discussion tied to the hypothesis. **Start here.** |

## Running it

```bash
# from the repository root
python -m venv .venv && source .venv/bin/activate   # if not already done
pip install -r requirements.txt -r Phillip/requirements.txt
cd Phillip
python run_experiment.py          # regenerates plots/ and results/summary.json
# or open working_memory_project.ipynb for the narrated version
```

`Phillip/requirements.txt` adds PyTorch (CPU-only) on top of the repository
root's `numpy`/`matplotlib`; it is only used by `trained_rnn.py`. Everything
else — the reservoir, all analyses, all plotting — is plain numpy, matching
the style of the rest of the repository (ridge regression and PCA are both
implemented directly rather than via `scikit-learn`).

## Design choices worth knowing about

- **Shared initialization.** `dynamics.init_shared_weights` is called once;
  the same `(J0, Jin0)` pair seeds both paradigms (`reservoir_model.build_reservoir`
  and `trained_rnn.train_trained_rnn` both take them as explicit arguments,
  rather than each drawing their own). Any representational difference found
  later is therefore attributable to training, not to a different random
  architecture draw.
- **Standing intrinsic noise.** `ExperimentConfig.intrinsic_noise_std` adds a
  small Gaussian noise floor to *every* simulation step — training included.
  Without this, gradient descent would have no reason to prefer a
  noise-robust delay-period solution over a fragile-but-accurate one (both
  score identically in a noiseless world), which would make the robustness
  comparison this study is built around untestable.
- **Fixed-delay analysis batches.** Cross-temporal decoding, PCA, and the
  perturbation sweeps all use a *single* delay length per batch (by default
  `config.primary_delay`, the longest scheduled delay), because those methods
  compare activity across trials at a shared absolute time index — only
  meaningful if every trial in the batch shares the same epoch boundaries.
  Training and the delay-length capacity curve use the full mixed grid. See
  `task.py`'s module docstring.
- **Perturbations are delay-only.** Every perturbation in `dynamics.Perturbation`
  is injected exclusively while a trial is inside its own delay epoch; cue
  encoding and the recall readout always run on the unperturbed network. Any
  accuracy drop is therefore attributable specifically to disrupted
  *maintenance*.
- **Matched-fraction lesioning.** Targeted silencing (by task-selectivity or
  by decoder-weight magnitude) is always compared against random silencing
  at the *same* fraction of units, so the comparison isolates whether *which*
  units are removed matters, not just how many.

## Key findings

_Numbers below are from one run (`results/summary.json`); see
`working_memory_project.ipynb` for the full narrative, figures, and a
from-scratch reproduction. Effect sizes are large enough to be robust to the
usual run-to-run seed variance, but exact decimals will shift on a re-run._

- **Working-memory capacity.** The Fixed Reservoir starts near-ceiling at
  short delays and degrades substantially by the longest delay (300 ms). The
  Trained RNN's accuracy is instead roughly **flat across delay length** —
  optimization buys delay-length invariance, not just a higher ceiling.
- **Cross-temporal decoding.** The reservoir's temporal-generalization matrix
  shows the classic dynamic-code signature: a bright diagonal ridge that
  fades off-diagonal. The trained RNN's matrix is close to a uniform square
  block spanning the whole delay — a temporally stable code. Its
  generalization-stability index sits far closer to 1 than the reservoir's.
- **Dimensionality.** PCA on delay-period activity gives the reservoir an
  effective dimensionality (participation ratio) an order of magnitude
  higher than the trained RNN's — the trained RNN's trajectories visibly
  shoot out to and then sit at a handful of fixed points per class, while
  the reservoir's keep drifting throughout the delay.
- **Robustness — conditional on the perturbation.** Against additive
  recurrent noise, mild weight-matrix degradation, and *random* unit
  silencing, the trained RNN is dramatically more robust than the reservoir,
  consistent with a low-dimensional attractor pulling noisy states back
  toward the correct answer. Against *targeted* silencing of the specific
  units its readout relies on most, that advantage weakens or reverses: the
  same concentration of task-relevant information that makes the trained
  solution efficient and diffuse-noise-robust also makes it vulnerable to a
  surgical strike on the few units carrying that information. **The
  hypothesis holds for diffuse/unstructured perturbations but does not
  simply generalize to precisely targeted ones** — a genuine trade-off, not
  a failure of the experiment.

## References

- Wang, X.-J. (2021). 50 years of mnemonic persistence: from monkey neurons to human memory.
- Meyers, E. M. (2018). Dynamic population coding and its relationship to working memory.
- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
- King, J.-R. & Dehaene, S. (2014). Characterizing the dynamics of mental representations: the temporal generalization method.

Full PDFs for related background reading are in `../papers/`.
