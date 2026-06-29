# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Environment

The project uses a `uv` virtual environment at `.venv/`. Always run scripts with:
```bash
PYTHONPATH=src .venv/bin/python <script>
```

All scripts insert `src/` into `sys.path` themselves, but must be run from the repo root.

## Repository Layout

```text
src/laplace_surrogate/
  data/               dataset.py, datamodule.py (LightningDataModule)
  models/             slae.py, llae.py, lslae.py
                      slae_surrogate.py, llae_surrogate.py
                      corrector.py, encoder_decoder.py, surrogate_base.py, base.py
  laplace_transform/  forward.py, inverse.py, learnable.py
  lightning/          ae_module.py, slae_surrogate_module.py,
                      llae_surrogate_module.py, lslae_surrogate_module.py,
                      corrector_module.py, ckpt_utils.py
  inference/          pipeline.py (InferencePipeline.from_checkpoint)
  utils/              visualization.py, rotate.py, make_split.py

configs/
  config.yaml         (Hydra root — default: model=llae, training=surrogate_llae)
  model/              slae.yaml, llae.yaml, lslae.yaml
  training/           ae.yaml, surrogate_slae.yaml, surrogate_llae.yaml,
                      surrogate_lslae.yaml, corrector.yaml
  data/               combustion.yaml
  eval/               default.yaml

scripts/
  train_ae.py         Phase 1 — train SLAE, LLAE, or LSLAE autoencoder
  train_surrogate.py  Phase 2 — train end-to-end surrogate θ→z→U(t)
  train_corrector.py  Phase 3 — train optional residual corrector
  evaluate.py         Evaluation from checkpoint
  laplace_opti.py     Standalone Laplace pole optimisation

app/
  streamlit_app.py, requirements-app.txt

tests/
  test_transform.py, test_models.py
```

## Dataset

Target: transient CH4 concentration fields `U(t)` as a function of physical parameters `θ = (k, A, C)`. Dataset: `dataset/ch4_rotated.npy` (8 100 simulations, 150 time steps, 200×200 grids).

`src/laplace_surrogate/data/dataset.py` exposes `TransientDataset(data_path, laplace, s_list, rule, dt, interp_size)`.

- `laplace=False`: `__getitem__` returns `(theta_norm, U)` with shape `(Nt, N, N)`.
- `laplace=True`: applies the Laplace transform and returns `(theta_norm, U_laplace_norm)` with shape `(K, 2, N, N)`.
- Call `dataset.fit(train_indices)` before training to compute normalization statistics.

## AE Families

Three autoencoder families are implemented. All share the same convolutional backbone (`ConvEncoder`, `ConvDecoder` in `encoder_decoder.py`) and the same training entry point (`train_ae.py`), controlled by `model=slae|llae|lslae`.

### SLAE — Spatial Laplace AE

The Laplace transform is applied first in the physical domain, then each complex frequency frame is encoded/decoded independently with a frequency-conditioned convolutional AE.

- Class: `src/laplace_surrogate/models/slae.py` → `SLAE`
- Conditioning: sinusoidal frequency encoding + FiLM
- Data flow: `U(t) → L → Û(s_k) → Encoder → z(s_k) → Decoder → L⁻¹ → Û_rec(t)`
- Surrogate model: `SLAEModel` (`slae_surrogate.py`)

### LLAE — Latent Laplace AE

Time-domain frames are encoded into a latent sequence first, then the Laplace transform is applied in latent space.

- Class: `src/laplace_surrogate/models/llae.py` → `LLAE`
- Conditioning: time ratio `t/T`
- Data flow: `U(t) → Encoder → z(t) → L → ẑ(s_k) → L⁻¹ → z̃(t) → Decoder → Û_rec(t)`
- Surrogate model: `LLAEModel` (`llae_surrogate.py`)

### LSLAE — Latent SVD Laplace AE (variant of LLAE)

LSLAE reuses a pre-trained LLAE encoder/decoder and adds an offline SVD projection on the latent sequence to compress the surrogate input dimension from `D` to `k_svd`.

- Class: `src/laplace_surrogate/models/lslae.py` → `LSLAE` (alias `LSLAEModel = LSLAE`)
- Extends `BaseDecoder` (only the decoder is trained; encoder is frozen from LLAE)
- Data flow: `U(t) → [frozen Enc] → z(t) → V → G(t) → L → Ĝ(s_k) → L⁻¹ → G̃(t) → Vᵀ → z̃(t) → [frozen Dec] → Û_rec(t)`
- SVD basis `V` is computed offline from the LLAE latent sequences before training
- Surrogate module: `LSLAESurrogateLightningModule` (`lslae_surrogate_module.py`)

## Surrogate Training (Phase 2)

The surrogate predicts Laplace-domain latents from `θ`, then uses the frozen decoder and inverse transform to recover the full transient field.

- Entry point: `scripts/train_surrogate.py`
- Dispatcher: selects `SLAESurrogateLightningModule`, `LLAESurrogateLightningModule`, or `LSLAESurrogateLightningModule` from `cfg.model.name`
- Reads `K` from the AE checkpoint via `ckpt_utils.peek_ae_hparams` before `dm.setup()`
- `SurrogateLightningModule` requires `dm.setup()` before construction (dataset stats needed at init)
- Core surrogate network: `FreqSurrogate` (`surrogate_base.py`) — trunk FFN shared across frequencies + per-frequency head
- Training signal: combined latent loss + physical-time reconstruction L2 loss

## Correction AE (Phase 3, optional)

A lightweight residual UNet that corrects Gibbs-like oscillations from spectral truncation.

- Class: `src/laplace_surrogate/models/corrector.py` → `CorrectionAE`
- Chained model: `CorrectedSLAEModel` (wraps `SLAEModel` + `CorrectionAE`)
- Input: reconstructed temporal field `Û_rec(t)`; output: residual correction `δ̂(t)`
- Lightning module: `CorrectorLightningModule`

## Main Workflow

```bash
# Phase 1 — Train AE  (model= slae | llae | lslae)
PYTHONPATH=src .venv/bin/python scripts/train_ae.py model=slae training=ae
PYTHONPATH=src .venv/bin/python scripts/train_ae.py model=llae training=ae
PYTHONPATH=src .venv/bin/python scripts/train_ae.py model=lslae training=ae

# Phase 2 — Train surrogate  (must match the AE used in Phase 1)
PYTHONPATH=src .venv/bin/python scripts/train_surrogate.py model=slae training=surrogate_slae training.ae_ckpt=<ckpt>
PYTHONPATH=src .venv/bin/python scripts/train_surrogate.py model=llae training=surrogate_llae training.ae_ckpt=<ckpt>
PYTHONPATH=src .venv/bin/python scripts/train_surrogate.py model=lslae training=surrogate_lslae training.ae_ckpt=<ckpt>

# Phase 3 — Optional corrector (SLAE pipeline only)
PYTHONPATH=src .venv/bin/python scripts/train_corrector.py training=corrector

# Evaluation
PYTHONPATH=src .venv/bin/python scripts/evaluate.py eval.ckpt_path=<ckpt>

# Streamlit demo
PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py
```

## Transforms

Module location: `src/laplace_surrogate/laplace_transform/`.

- `learnable.py` → `LearnableLaplace(K, dt, Nt)` with learnable poles `s_k`
- `forward.py` → `laplace_forward_tik(U, s_list, dt, rule)`
- `inverse.py` → `laplace_inverse_tik(U_hat, s_list, dt, Nt, alpha_t, lam, rule)`

## Inference

```python
from laplace_surrogate.inference.pipeline import InferencePipeline

pipe = InferencePipeline.from_checkpoint('checkpoints/SLAEModel__slae_ld64_K16_g0.0__t4h2.pt')
U_pred = pipe.predict([[k, A, C]])  # (B, Nt, N, N) float32
```

`InferencePipeline.from_checkpoint` reads `model_type` from the checkpoint and handles `θ` normalization automatically. Supported backends: `SLAEModel`, `LLAEModel`, `LSLAEModel`, `CorrectionAE`.

## Checkpoint Naming Convention

AE checkpoints are saved as `{ae_tag}.pt` where `ae_tag` encodes key hyperparameters:
- SLAE: `slae_ld{latent_dim}_K{K}_g{gamma_init}[_ll][_ol]`
- LLAE: `llae_ld{latent_dim}_K{K}_g{gamma_init}[_ll]`
- LSLAE: `lslae_ld{latent_dim}_K{K}_ksvd{k_svd}[_ll]`

Surrogate checkpoints: `{ModelClass}__{ae_stem}__t{n_trunk}h{n_head}.pt`

## Lightning / Hydra Notes

- `AELightningModule` handles all three AE types (SLAE, LLAE, LSLAE); instantiates the right model from `cfg.model.name`.
- `AELightningModule.on_train_epoch_start` calls `dm.train_dataset.reshuffle()` to reshuffle simulation/frequency pairs each epoch.
- `SurrogateLightningModule` requires `dm.setup()` before construction.
- `ckpt_utils.peek_ae_hparams` reads `K` from an AE checkpoint without loading model weights.

## Useful Commands

```bash
# Smoke tests
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q

# Ablation sweep on latent_dim
PYTHONPATH=src .venv/bin/python scripts/train_ae.py --multirun model=slae model.latent_dim=16,32,64,128

# Extract plots from a notebook into article/images/
python notebooks/extract_plots.py <notebook_stem>          # e.g. eval_checkpoints
python notebooks/extract_plots.py <notebook_stem> --prefix fig_ --out article/images
```