import torch


def laplace_inverse_tik(U_hat, s_list, dt, Nt, alpha_t, lam, rule='trap'):
    """
    Regularised Laplace inversion at arbitrary s points.  Differentiable (PyTorch).

    Solves the normal equations with Tikhonov regularization:
        A v* = Re(F^H û)
        A   = Re(F^H F) + alpha_t * D_t^T D_t + lam * I

    Conjugate symmetry: frequencies with Im(s) > 0 are extended by their conjugates
    before forming the normal equations (doubles effective constraints cheaply).

    Works in float32 or float64 depending on s_list.dtype:
      - complex128 (default) → float64, CPU, best accuracy
      - complex64            → float32, stays on s_list.device, GPU-differentiable

    Parameters
    ----------
    U_hat  : (Nnodes, K) complex  – output of laplace_forward_tik
    s_list : (K,) complex Tensor
    dt     : float
    Nt     : int   – number of time steps to reconstruct
    alpha_t: float – temporal-smoothness weight
    lam    : float – ridge weight (use > 0 when K < Nt to avoid singularity)
    rule   : 'rect' | 'trap'  – must match forward

    Returns
    -------
    V_rec : (Nnodes, Nt) real Tensor  (same device as s_list)
    """
    if not isinstance(U_hat, torch.Tensor):
        U_hat = torch.tensor(U_hat, dtype=torch.complex128)
    if not isinstance(s_list, torch.Tensor):
        s_list = torch.tensor(s_list, dtype=torch.complex128)

    device = s_list.device
    cdtype = s_list.dtype                                    # complex64 or complex128
    rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64

    U_hat = U_hat.to(device=device, dtype=cdtype)
    t = torch.arange(Nt, dtype=rdtype, device=device) * dt
    w = torch.ones(Nt, dtype=rdtype, device=device)
    if rule == 'trap':
        w[0] = 0.5; w[-1] = 0.5

    # Conjugate extension: Im(s) > 0 → add conjugate pair
    c_mask     = s_list.imag > 0
    s_full     = torch.cat([s_list, torch.conj(s_list[c_mask])])
    U_hat_full = torch.cat([U_hat, torch.conj(U_hat[:, c_mask])], dim=1)  # (Nnodes, K_full)

    # F_full[k, t] = dt * w[t] * exp(-s_full[k] * t)   (K_full, Nt)
    F_full = dt * w[None, :] * torch.exp(-s_full[:, None] * t[None, :])
    FH     = torch.conj(F_full).T                           # (Nt, K_full)
    FtF    = torch.real(FH @ F_full)                        # (Nt, Nt) rdtype

    Dt    = (torch.diag(torch.ones(Nt - 1, dtype=rdtype, device=device), 1)
             - torch.eye(Nt, dtype=rdtype, device=device))[:Nt - 1, :]
    DtTDt = Dt.T @ Dt

    A   = FtF + alpha_t * DtTDt + lam * torch.eye(Nt, dtype=rdtype, device=device)
    RHS = torch.real(U_hat_full @ torch.conj(F_full))       # (Nnodes, Nt)

    return torch.linalg.solve(A, RHS.T).T                   # (Nnodes, Nt) rdtype
