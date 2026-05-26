# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Environment

The project uses a local conda environment at `.conda/`. Always run scripts with:
```bash
.conda/bin/python <script>
```

All scripts add their parent directory to `sys.path`, so they must be run from the repo root.

## Repository Layout

The transient code has been fully migrated into `src/laplace_surrogate/`. The legacy `transient/`, `models/`, and `utils/` folders were removed.

```text
src/laplace_surrogate/
  data/         dataset.py, datamodule.py (LightningDataModule)
  models/       slae, llae, lslae, surrogate, corrector, laplace_svd,
                svd_surrogate, laplace_model, laplace_ffn, tucker_pod, base
  transform/    forward.py, inverse.py, learnable.py (LearnableLaplace)
  lightning/    ae_module, surrogate_module, corrector_module
  inference/    pipeline.py (InferencePipeline.from_checkpoint)
  utils/        visualization.py, rotate.py, make_split.py

configs/
  model/        slae.yaml, laplace_latent.yaml
  training/     autoencoder.yaml, surrogate.yaml, corrector.yaml
  data/         combustion.yaml
  eval/         default.yaml
  experiment/   ablation_{latent_dim,kmax,gamma}.yaml
  config.yaml   (Hydra root)

scripts/
  train_ae.py, train_surrogate.py, train_corrector.py, evaluate.py

app/
  streamlit_app.py, requirements-app.txt

tests/
  test_transform.py, test_models.py
```

## Dataset

The target is transient CH4 concentration fields `U(t)` as a function of the physical parameters `θ = (k, A, C)`. The main dataset is `dataset/ch4_rotated.npy` (8,100 simulations, 150 time steps, 200×200 grids).

`src/laplace_surrogate/data/dataset.py` exposes `TransientDataset(data_path, laplace, s_list, rule, dt, interp_size)`.

- `laplace=False`: `__getitem__` returns `(theta_norm, U)` with shape `(Nt, N, N)`.
- `laplace=True`: applies the Laplace transform and returns `(theta_norm, U_laplace_norm)` with shape `(K, 2, N, N)`.
- Call `dataset.fit(train_indices)` before training to compute normalization statistics.

## Pipelines From The Article

The article compares one linear baseline, three Laplace-domain surrogate families, and a final correction stage.

### Pipeline 1: Tucker POD / SVDSurrogate

This is the linear baseline. Time-series snapshots are compressed with a Tucker/POD basis, the surrogate predicts the reduced coefficients from `θ`, and the result is reconstructed in physical time.

- Main class: `src/laplace_surrogate/models/tucker_pod.py` -> `TuckerPODModel`
- Wrapper class: `src/laplace_surrogate/models/svd_surrogate.py` -> `SVDSurrogate`
- Limitation: linear basis truncation error, especially on sharp transients

### Pipeline 2: LaplaceSVDModel

This baseline keeps the Laplace-domain structure but uses a truncated SVD representation per frequency. It is still a linear compression strategy, but frequency-aware.

- Main class: `src/laplace_surrogate/models/laplace_svd.py` -> `LaplaceSVDModel`
- Role: Laplace-domain linear baseline
- Limitation: truncation error from the spatial basis, and no learned non-linear compression

### Pipeline 3: LaplaceFFNModel / LaplaceModel style direct surrogate

This family predicts Laplace-domain quantities directly with independent frequency heads. It is the simplest Laplace surrogate family and a useful comparison point against the AE-based pipelines.

- Main class: `src/laplace_surrogate/models/laplace_ffn.py` -> `LaplaceFFNModel`
- Legacy naming in the article/codebase: `LaplaceModel`
- Role: direct multi-head Laplace surrogate
- Limitation: weaker compression than AE-based models, higher burden on the surrogate heads

### Pipeline 4: SLAE

Spatial Laplace AE. The Laplace transform is applied in the physical domain first, then each complex frequency frame is encoded and decoded independently with a frequency-conditioned convolutional AE.

- Main class: `src/laplace_surrogate/models/slae.py` -> `SLAE`
- Conditioning: sinusoidal coordinate encoding + FiLM on the frequency ratio
- Shape intuition: `U(t) -> L -> U_hat(s_k) -> Encoder -> z(s_k) -> Decoder -> L^{-1} -> U_rec(t)`
- Strength: strong spatial compression with shared weights across frequencies
- Limitation: must process all frequency frames during AE training and inference

### Pipeline 5: LLAE

Latent Laplace AE. The time-domain frames are first encoded into a latent sequence, then the Laplace transform is applied in latent space.

- Main class: `src/laplace_surrogate/models/llae.py` -> `LLAE`
- Conditioning: time ratio `t/T`
- Shape intuition: `U(t) -> Encoder -> z(t) -> L -> z_hat(s_k) -> L^{-1} -> z_rec(t) -> Decoder -> U_rec(t)`
- Strength: compresses the dynamics before the transform
- Limitation: must pass all time frames through the AE, so it is heavier than SLAE

### Pipeline 6: LSLAE

Latent SVD Laplace AE. This is the LLAE family with an offline SVD projection on the latent sequence to reduce the surrogate dimension.

- Main class: `src/laplace_surrogate/models/lslae.py` -> `LSLAE`
- Legacy alias: `LSLAEModel`
- Shape intuition: `U(t) -> Encoder -> z(t) -> V -> G(t) -> L -> G_hat(s_k) -> L^{-1} -> G_rec(t) -> V^T -> z_rec(t) -> Decoder -> U_rec(t)`
- Strength: simplifies surrogate learning by shrinking the latent dimension
- Limitation: adds SVD truncation error and still processes all time frames

### Pipeline 7: Surrogate Training

The main end-to-end surrogate predicts the Laplace-domain latents from `θ`, then uses the frozen decoder and inverse transform to recover the full transient field.

- Main class: `src/laplace_surrogate/models/slae_surrogate.py` -> `SLAEModel`
- Related classes: `LLAEModel`, `LSLAEModel`
- Training entry point: `scripts/train_surrogate.py`
- Core idea: `θ -> latent(s_k) -> decoder -> inverse Laplace -> U_rec(t)`
- Training signal: combined latent loss and physical-time reconstruction loss

### Pipeline 8: Correction AE

The final optional stage is a lightweight residual UNet that corrects Gibbs-like oscillations introduced by spectral truncation.

- Main class: `src/laplace_surrogate/models/corrector.py` -> `CorrectionAE`
- Chained model: `CorrectedSLAEModel`
- Training entry point: `scripts/train_corrector.py training=corrector`
- Input: inverted temporal field `U_rec(t)`
- Output: residual correction `δ_pred(t)`

## Main Workflow

The production pipeline described in the article is SLAE + surrogate + optional corrector.

```bash
# Step 1 - Train the Laplace AE
PYTHONPATH=src .conda/bin/python scripts/train_ae.py

# Step 2 - Train the end-to-end surrogate θ→z→U(t)
PYTHONPATH=src .conda/bin/python scripts/train_surrogate.py

# Step 3 - Optional residual correction
PYTHONPATH=src .conda/bin/python scripts/train_corrector.py training=corrector

# Evaluation
PYTHONPATH=src .conda/bin/python scripts/evaluate.py eval.ckpt_path=checkpoints/LaplaceLatentModel_best.pt

# Streamlit app
PYTHONPATH=src .conda/bin/streamlit run app/streamlit_app.py
```

## Transforms

- `src/laplace_surrogate/laplace_transform/learnable.py` -> `LearnableLaplace(K, dt, Nt)` with learnable poles `s_k`
- `src/laplace_surrogate/laplace_transform/forward.py` -> `laplace_forward_tik(U, s_list, dt, rule)`
- `src/laplace_surrogate/laplace_transform/inverse.py` -> `laplace_inverse_tik(U_hat, s_list, dt, Nt, alpha_t, lam, rule)`
- `src/laplace_surrogate/transform/` remains as a compatibility shim for older imports

## Inference

```python
from laplace_surrogate.inference.pipeline import InferencePipeline

pipe = InferencePipeline.from_checkpoint('checkpoints/LaplaceLatentModel_best.pt')
U_pred = pipe.predict([[k, A, C]])  # (B, Nt, N, N) float32
```

`InferencePipeline` detects the `model_type` from the checkpoint and handles `θ` normalization automatically. It supports `SLAEModel`, `LLAEModel`, `LSLAEModel`, `LaplaceSVDModel`, `LaplaceFFNModel`, `TuckerPODModel`, and `CorrectionAE`.

## Lightning / Hydra

- `src/laplace_surrogate/lightning/` contains `AELightningModule`, `SurrogateLightningModule`, and `CorrectorLightningModule`.
- `configs/` contains `model/slae.yaml`, `training/{autoencoder,surrogate,corrector}.yaml`, `data/combustion.yaml`, and `eval/default.yaml`.
- `SurrogateLightningModule` requires `dm.setup()` before construction because the dataset statistics are needed at init time.
- `AELightningModule.on_train_epoch_start` calls `dm.train_dataset.reshuffle()` to reshuffle simulation/frequency pairs.

## Useful Commands

```bash
# Smoke tests
PYTHONPATH=src .conda/bin/python -m pytest tests/ -q

# Example ablation sweep
PYTHONPATH=src .conda/bin/python scripts/train_ae.py --multirun experiment=ablation_latent_dim model.latent_dim=16,32,64,128
```

