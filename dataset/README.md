# Dataset

The simulation data is not distributed with this repository (several tens of GB).
Contact the authors to obtain it.

Files expected in this directory (symbolic links to a scratch disk work fine):

| File | Tracked | Content |
|------|---------|---------|
| `doe.npy`         | yes | design of experiments: 225 parameter triplets θ = (k, A, C), structured array |
| `split.npz`       | yes | `train_idx` (6 480) / `test_idx` (1 620) indices into the rotated dataset |
| `CH4.npy`         | no  | raw CH₄ fields, `(225, 150, 200, 200)` |
| `ch4_rotated.npy` | no  | 36 rotations of each simulation, cropped to 128 × 128: `(8100, 150, 128, 128)` |
| `doe_rotated.npy` | no  | θ for every rotated simulation, `(8100,)` |

The rotated files are generated from `CH4.npy` and `doe.npy` with

```bash
PYTHONPATH=src python -m laplace_surrogate.utils.rotate --input_dir dataset --output_dir dataset
```
