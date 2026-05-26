from laplace_surrogate.laplace_transform.forward import laplace_forward_tik

__all__ = ["laplace_forward_tik"]


def laplace_forward_tik(C, s_list, dt, rule='trap'):
    """
    Forward Laplace at arbitrary s points.  Differentiable (PyTorch).

    Computes  U_hat[n, k] = sum_t C[n,t] * dt * w[t] * exp(-s_k * t)

    Parameters
    ----------
    C      : (Nnodes, Nt) real  – ndarray or Tensor
    s_list : (K,) complex       – ndarray or Tensor
    dt     : float
    rule   : 'rect' | 'trap'

    Returns
    -------
    U_hat : (Nnodes, K) complex Tensor
    """
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C, dtype=torch.float64)
    if not isinstance(s_list, torch.Tensor):
        s_list = torch.tensor(s_list, dtype=torch.complex128)

    device = s_list.device
    C = C.to(device=device)
    Nnodes, Nt = C.shape
    t = torch.arange(Nt, dtype=torch.float64, device=device) * dt

    w = torch.ones(Nt, dtype=torch.float64, device=device)
    if rule == 'trap':
        w[0] = 0.5
        w[-1] = 0.5

    # F[k, t] = dt * w[t] * exp(-s_k * t)   (K, Nt) complex
    F = dt * w[None, :] * torch.exp(-s_list[:, None] * t[None, :])
    return C.to(dtype=F.dtype) @ F.T   # (Nnodes, K)
