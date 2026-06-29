"""
tucker.py — Décomposition de Tucker (HOOI) sur deux modes d'un tenseur latent.

Utilisé par les surrogates LLAE-Tucker et SLAE-Tucker pour compresser conjointement
les modes fréquence (K → r_s) et latent (d_z → r_z) du tenseur des latents Laplace
Ẑ ∈ (N_train, K, d_z), sans toucher au mode échantillon (mode 0).

Convention (forme unitaire, valable réel ET complexe) :
  reconstruction : Ẑ_i ≈ U_s · G_i · U_zᴴ
  projection     : G_i = U_sᴴ · Ẑ_i · U_z
Les facteurs U_s [K, r_s] et U_z [d_z, r_z] ont des colonnes orthonormées.
Pour des tenseurs réels (SLAE) la conjugaison est sans effet et l'on retrouve
exactement la forme U_s G U_zᵀ de l'article.
"""
import torch


def _truncated_left(M: torch.Tensor, r: int) -> torch.Tensor:
    """r premiers vecteurs singuliers à gauche de M (réel ou complexe)."""
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    return U[:, :r].contiguous()


def project(Z: torch.Tensor, U_s: torch.Tensor, U_z: torch.Tensor) -> torch.Tensor:
    """(B, K, d_z) → core (B, r_s, r_z) :  G = U_sᴴ Z U_z."""
    return torch.einsum('kr,bkm,mz->brz', U_s.conj(), Z, U_z)


def reconstruct(G: torch.Tensor, U_s: torch.Tensor, U_z: torch.Tensor) -> torch.Tensor:
    """core (B, r_s, r_z) → (B, K, d_z) :  Z = U_s G U_zᴴ."""
    return torch.einsum('kr,brz,mz->bkm', U_s, G, U_z.conj())


@torch.no_grad()
def hooi(Z: torch.Tensor, r_s: int, r_z: int, n_iter: int = 5):
    """
    HOOI sur les modes 1 (taille K → r_s) et 2 (taille d_z → r_z) du tenseur
    Z (N, K, d_z). Le mode échantillon (0) n'est pas factorisé.

    Retourne (U_s [K, r_s], U_z [d_z, r_z]) à colonnes orthonormées, du même
    dtype que Z (réel ou complexe).
    """
    N, K, D = Z.shape
    r_s = min(r_s, K)
    r_z = min(r_z, D)

    # ── Initialisation HOSVD (SVD des dépliages de modes) ────────────────────
    # mode 1 : [K, N*D]
    U_s = _truncated_left(Z.permute(1, 0, 2).reshape(K, N * D), r_s)
    # mode 2 : [D, N*K]
    U_z = _truncated_left(Z.permute(2, 0, 1).reshape(D, N * K), r_z)

    # ── Itérations HOOI ──────────────────────────────────────────────────────
    for _ in range(n_iter):
        # mise à jour de U_s : projeter le mode latent, déplier le mode fréquence
        Y    = torch.einsum('bkm,mz->bkz', Z, U_z)            # (N, K, r_z)
        U_s  = _truncated_left(Y.permute(1, 0, 2).reshape(K, N * r_z), r_s)
        # mise à jour de U_z : projeter le mode fréquence, déplier le mode latent
        Y    = torch.einsum('kr,bkm->brm', U_s.conj(), Z)     # (N, r_s, d_z)
        U_z  = _truncated_left(Y.permute(2, 0, 1).reshape(D, N * r_s), r_z)

    return U_s.contiguous(), U_z.contiguous()
