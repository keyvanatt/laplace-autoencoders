# diffusion_ae

Émulation de solveurs EDP (convection-diffusion stationnaire et transitoire CH4) par apprentissage automatique.

## Structure

```
src/laplace_surrogate/         — code transitoire (package principal)
  data/                        — dataset.py, datamodule.py
  models/                      — SLAE, LLAE, LaplaceLatentModel, CorrectionAE, …
  laplace_transform/           — LearnableLaplace, laplace_forward_tik/laplace_inverse_tik
  lightning/                   — AE/Surrogate/CorrectorLightningModule
  inference/                   — InferencePipeline.from_checkpoint
  utils/                       — visualization, rotate, make_split

configs/                       — Hydra (model, training, data, eval, experiment)
scripts/                       — train_ae.py, train_surrogate.py, train_corrector.py, evaluate.py
app/                           — streamlit_app.py
tests/                         — test_transform.py, test_models.py

dataset/                       — données brutes (.npy / .npz)
checkpoints/                   — modèles sauvegardés (.pt)
```

## Transitoire (SLAE surrogate)

Prédit les champs CH4 transitoires U(t) depuis θ = (k, A, C).
Dataset principal : `dataset/ch4_rotated.npy` (8 100 sims × 150 pas × 200×200).

### Entraînement

```bash
# Étape 1 — Autoencoder Laplace (SLAE)
PYTHONPATH=src .conda/bin/python scripts/train_ae.py

# Étape 2 — Surrogate end-to-end θ→z→U(t)
PYTHONPATH=src .conda/bin/python scripts/train_surrogate.py

# Étape 3 — CorrectionAE (post-processing, optionnel)
PYTHONPATH=src .conda/bin/python scripts/train_corrector.py training=corrector
```

### Évaluation

```bash
PYTHONPATH=src .conda/bin/python scripts/evaluate.py \
    eval.ckpt_path=checkpoints/LaplaceLatentModel_best.pt
```

Produit : L2rel (%) JSON + histogramme + GIFs comparaison (best/median/worst).

### App interactive

```bash
PYTHONPATH=src .conda/bin/streamlit run app/streamlit_app.py
```

### Inférence

```python
from laplace_surrogate.inference.pipeline import InferencePipeline

pipe   = InferencePipeline.from_checkpoint('checkpoints/LaplaceLatentModel_best.pt')
U_pred = pipe.predict([[k, A, C]])  # (B, Nt, N, N) float32
```

### Ablations

```bash
PYTHONPATH=src .conda/bin/python scripts/train_ae.py \
    --multirun experiment=ablation_latent_dim model.latent_dim=16,32,64,128
```

### Tests

```bash
PYTHONPATH=src .conda/bin/python -m pytest tests/ -q
```

## Dépendances

```bash
.conda/bin/pip install pytorch-lightning>=2.0 hydra-core>=1.3 omegaconf>=2.3
```

## Experiment tracking

Tous les scripts loggent sur [Weights & Biases](https://wandb.ai) (projet `convdiff`).
Checkpoints dans `checkpoints/<ModelName>_best.pt`.
