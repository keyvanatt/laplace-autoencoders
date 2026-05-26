"""
For each entry in CH4 (shape: N_samples x T x H x W), generate N_ROTATIONS rotated copies
with uniformly spaced rotation angles. The rotation is performed at native resolution (200x200),
then center-cropped to OUTPUT_SIZE x OUTPUT_SIZE — no interpolation for downscaling.

Rotating before cropping avoids:
  - aliasing from rotating an already-coarse grid
  - border artifacts: the crop simply discards the region affected by rotation fill

Saves:
  - {output_dir}/ch4_rotated.npy  : shape (N_samples * N_ROTATIONS, T, output_size, output_size), float32, memmap
  - {output_dir}/doe_rotated.npy  : structured array (k, A, C, theta), shape (N_samples * N_ROTATIONS,)
"""

import os
import numpy as np
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm


def rotate_and_crop(frame: np.ndarray, M: np.ndarray, H: int, W: int, output_size: int) -> np.ndarray:
    rotated = cv2.warpAffine(
        frame, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    cy, cx = H // 2, W // 2
    half = output_size // 2
    return rotated[cy - half:cy + half, cx - half:cx + half]


def rotate_dataset(
    input_dir: str,
    output_dir: str,
    n_rotations: int = 36,
    n_jobs: int = 8,
    output_size: int = 128,
) -> None:
    """Rotate and crop a CH4 dataset, saving the result as memory-mapped numpy arrays.

    Args:
        input_dir:   Directory containing CH4.npy and doe.npy.
        output_dir:  Destination directory for ch4_rotated.npy and doe_rotated.npy.
        n_rotations: Number of uniformly-spaced rotation angles per sample.
        n_jobs:      Number of parallel jobs (joblib threads).
        output_size: Final spatial resolution after center-cropping.
    """
    ch4 = np.load(os.path.join(input_dir, "CH4.npy"))   # (N, T, H, W)
    doe = np.load(os.path.join(input_dir, "doe.npy"))    # structured: (N,) with fields k, A, C

    N, T, H, W = ch4.shape
    assert H == 200 and W == 200, f"Expected 200x200 input, got {H}x{W}"
    assert output_size < H, "output_size must be smaller than native resolution"
    print(f"CH4 shape: {ch4.shape}, dtype: {ch4.dtype}")
    print(f"doe shape: {doe.shape}, fields: {doe.dtype.names}")
    print(f"Pipeline: rotate at {H}x{W} → center-crop to {output_size}x{output_size}")

    angles = np.linspace(0, 360, n_rotations, endpoint=False)
    print(f"Rotation angles (degrees): {angles}")

    center = (W / 2.0, H / 2.0)
    rot_matrices = [cv2.getRotationMatrix2D(center, float(a), 1.0) for a in angles]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ch4_rotated.npy")
    out_shape = (N * n_rotations, T, output_size, output_size)
    fp = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=out_shape)
    print(f"Allocated memmap {out_path}  shape={out_shape}  dtype=float32")

    out_doe = np.empty(N * n_rotations, dtype=np.dtype([
        ('k', '<f8'), ('A', '<f8'), ('C', '<f8')
    ]))

    def process_sample(i):
        sample = ch4[i].astype(np.float32)  # (T, H, W)
        k_val = float(doe['k'][i])
        A_val = float(doe['A'][i])
        if A_val != 0.0:
            print(f"Warning: A={A_val} non nul pour i={i} — vérifie les données d'entrée.")
        C_val = float(doe['C'][i])

        for j, (angle, M) in enumerate(zip(angles, rot_matrices)):
            idx = i * n_rotations + j
            rotated = np.stack([
                rotate_and_crop(sample[t], M, H, W, output_size)
                for t in range(T)
            ])  # (T, output_size, output_size)
            fp[idx] = rotated
            out_doe[idx] = (k_val, angle, C_val)

        fp.flush()

    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_sample)(i) for i in tqdm(range(N), desc="Samples")
    )

    print(f"Saved {out_path}")

    doe_path = os.path.join(output_dir, "doe_rotated.npy")
    np.save(doe_path, out_doe)
    print(f"Saved {doe_path}")
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="/users/eleves-b/2023/keyvan.attarian/diffusion_ae/dataset")
    parser.add_argument("--output_dir", default="/Data/KAT")
    parser.add_argument("--n_rotations", type=int, default=36)
    parser.add_argument("--n_jobs",      type=int, default=8)
    parser.add_argument("--output_size", type=int, default=128)
    args = parser.parse_args()

    rotate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_rotations=args.n_rotations,
        n_jobs=args.n_jobs,
        output_size=args.output_size,
    )
