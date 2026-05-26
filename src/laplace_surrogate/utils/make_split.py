import os
import numpy as np


def make_train_test_split(
    doe_path: str,
    out_path: str,
    test_frac: float = 0.2,
    seed: int = 42,
) -> None:
    """Create a reproducible train/test split from a DOE file and save it as .npz.

    Args:
        doe_path:  Path to the .npy DOE array (used only to determine n_samples).
        out_path:  Destination .npz file with keys train_idx and test_idx.
        test_frac: Fraction of samples reserved for the test set.
        seed:      Random seed for reproducibility.
    """
    doe = np.load(doe_path)
    ns  = len(doe)

    rng       = np.random.default_rng(seed)
    idx       = rng.permutation(ns)
    n_test    = int(test_frac * ns)
    test_idx  = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, train_idx=train_idx, test_idx=test_idx)

    print(f"Split sauvegardé : {out_path}")
    print(f"  train : {len(train_idx)} samples  ({100*(1-test_frac):.0f}%)")
    print(f"  test  : {len(test_idx)} samples  ({100*test_frac:.0f}%)")
    print(f"  seed  : {seed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--doe_path",  default=os.path.join("dataset", "doe_rotated.npy"))
    parser.add_argument("--out_path",  default=os.path.join("dataset", "split.npz"))
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    make_train_test_split(
        doe_path=args.doe_path,
        out_path=args.out_path,
        test_frac=args.test_frac,
        seed=args.seed,
    )
