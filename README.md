# Laplace autoencoders for parametric space–time surrogates

Code for *A Laplace-domain a posteriori surrogate for the parametric space–time solution of
parabolic problems* (K. Attarian, A. Ammar, F. Chinesta — preprint in preparation).

The surrogate maps physical parameters θ to the **complete** transient field U(t) in a single
evaluation. Instead of predicting one latent vector per time step (as a DL-ROM does), it predicts
K ≪ Nt Laplace-domain coefficients and recovers the trajectory with a differentiable,
Tikhonov-regularised inverse Laplace transform. The output cost no longer grows with Nt, and
one trained model serves any output time grid above a sampling floor set by its poles.

Two placements of the transform are implemented:

| Model | Data flow |
|-------|-----------|
| **SLAE** — Spatial Laplace AE | `U(t) → L → Û(s_k) → Enc → z(s_k) → Dec → L⁻¹ → U(t)` |
| **LLAE** — Latent Laplace AE  | `U(t) → Enc → z(t) → L → ẑ(s_k) → L⁻¹ → z(t) → Dec → U(t)` |
| **DL-ROM** — baseline, no Laplace ([Fresca et al. 2021](https://doi.org/10.1007/s10915-021-01462-7)) | `U(t) → Enc → z(t) → Dec → U(t)` |

All three share the same convolutional encoder/decoder (FiLM-conditioned) and the same
θ → latent surrogate network (`FreqSurrogate`), so comparisons isolate the effect of the
Laplace compression. Optional SVD / Tucker compression of the predicted latents is also provided.

Test case: transient CH₄ concentration in a parametric combustion problem,
θ = (k, A, C), 8 100 simulations × 150 time steps on a 128 × 128 grid.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python ≥ 3.13.

```bash
git clone https://github.com/keyvanatt/laplace-autoencoders.git
cd laplace-autoencoders
uv sync                 # add --group dev for notebooks and tests
```

All commands below are run from the repository root with `PYTHONPATH=src`
(`.venv/bin/python` on Linux/macOS, `.venv\Scripts\python` on Windows).

## Data and checkpoints

- **Dataset** — the simulation data is on the Hugging Face Hub (CC BY-NC 4.0):
  [keyvanatt/laplace-autoencoders-dataset](https://huggingface.co/datasets/keyvanatt/laplace-autoencoders-dataset).
  See [dataset/README.md](dataset/README.md) for the files and layout.

  ```bash
  hf download keyvanatt/laplace-autoencoders-dataset --repo-type dataset --local-dir dataset
  ```
- **Trained checkpoints** (AEs and surrogates for every configuration in the paper) are on the
  Hugging Face Hub: [keyvanatt/laplace-autoencoders-checkpoints](https://huggingface.co/keyvanatt/laplace-autoencoders-checkpoints).

  ```bash
  hf download keyvanatt/laplace-autoencoders-checkpoints --local-dir checkpoints
  ```

## Quick start: inference

```python
from laplace_surrogate.inference.pipeline import InferencePipeline

pipe = InferencePipeline.from_checkpoint('checkpoints/LLAEModel__llae_ld64_K16_g0.01__t4h2.ckpt')
U = pipe.predict([[k, A, C]])   # (B, Nt, N, N) float32, θ given in physical units
```

`from_checkpoint` reads the model type stored in the checkpoint and handles θ normalisation.
Supported: `SLAEModel`, `LLAEModel`, `SLAESVDModel`, `LLAESVDModel`, `SLAETuckerModel`,
`LLAETuckerModel`, `DLROMModel`, `CorrectionAE`.

An interactive demo is available with `streamlit run app/streamlit_app.py`
(extra requirements in `app/requirements-app.txt`).

## Training

Training is configured with [Hydra](https://hydra.cc) (`configs/`) and logged to
[Weights & Biases](https://wandb.ai). It happens in two phases.

```bash
# Phase 1 — autoencoder (model = slae | llae | dlrom)
python scripts/train_ae.py model=llae training=ae

# Phase 2 — surrogate θ → latents, trained end to end through the frozen decoder
python scripts/train_surrogate.py model=llae training=surrogate_llae training.ae_ckpt=<ae.ckpt>

# Phase 2 variants — compressed latents
python scripts/train_surrogate.py model=llae training=surrogate_llae_svd    training.ae_ckpt=<ae.ckpt>
python scripts/train_surrogate.py model=llae training=surrogate_llae_tucker training.ae_ckpt=<ae.ckpt>

# DL-ROM baseline, optionally on a subsampled time grid (Nt → ceil(150 / k))
python scripts/train_ae.py        model=dlrom training=ae data.t_stride=2
python scripts/train_surrogate.py model=dlrom training=surrogate_dlrom training.ae_ckpt=<ae.ckpt>

# Optional phase 3 — residual UNet corrector on top of an SLAE surrogate
python scripts/train_corrector.py training=corrector
```

The default output and cache locations (`save_dir`, `data.cache_dir`) point to the scratch disk
used for the paper; override them on the command line, e.g. `save_dir=checkpoints data.cache_dir=cache`.

`scripts/laplace_opti.py` optimises the Laplace poles offline, before any training.

## Evaluation and figures

```bash
python scripts/evaluate.py eval.ckpt_path=<surrogate.ckpt>
```

writes relative L² errors on the test set, a histogram and comparison GIFs to `plots/`.

The figures and tables of the paper are produced by the notebooks in `notebooks/`:

| Notebook | Content |
|----------|---------|
| `dataset_demo` | dataset snapshots, parameter distribution, dynamics |
| `eval_ae_prescribed`, `eval_ae_optimal` | autoencoder reconstruction (prescribed / optimised poles) |
| `eval_surr_prescribed`, `eval_surr_optimal` | surrogate accuracy |
| `eval_nt150`, `eval_nt_sweep` | comparison with DL-ROM, time-resolution sweep |
| `eval_pod` | linear POD baseline |
| `eval_compression_surr`, `eval_tucker_surr` | SVD / Tucker latent compression |

Notebook outputs are stripped on commit by [nbstripout](https://github.com/kynan/nbstripout);
run `nbstripout --install` once after cloning if you plan to commit notebooks.

## Repository layout

```text
src/laplace_surrogate/
  data/               TransientDataset, LightningDataModule
  laplace_transform/  forward / inverse (Tikhonov) transforms, learnable poles
  models/             SLAE, LLAE, surrogates (direct, SVD, Tucker), corrector
  lightning/          Lightning modules for every training phase
  inference/          InferencePipeline
src/dl_rom/           DL-ROM baseline
configs/              Hydra configs (model, training, data, eval)
scripts/              training, evaluation, pole optimisation
notebooks/            paper figures
app/                  Streamlit demo
tests/                unit tests (pytest)
```

## Tests

```bash
uv sync --group dev
PYTHONPATH=src python -m pytest tests -q
```

## License

MIT — see [LICENSE](LICENSE).
