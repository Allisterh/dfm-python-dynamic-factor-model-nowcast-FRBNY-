"""Krylov sequences ``c' A^l b`` (the S3 "state space kernel"), from the Spacetime code."""

import torch
from einops import rearrange


def krylov_sequential(L, A, b, c=None):
    """Krylov sequence by ``L`` matrix-vector products.

    A : (..., N, N), b : (..., N), c : (..., N). Returns ``x[..., l] = c @ A^l @ b``,
    shape (..., L), or without ``c`` ``x[..., :, l] = A^l @ b``, shape (..., N, L).
    """
    # Iterate on the smaller of b and c to save memory.
    if c is not None and c.numel() < b.numel():
        return krylov_sequential(L, A.transpose(-1, -2), c, b)

    b_ = b
    x = []
    for _ in range(L):
        if c is not None:
            x_ = torch.sum(c*b_, dim=-1)
        else:
            x_ = b_
        x.append(x_)
        b_ = (A @ b_.unsqueeze(-1)).squeeze(-1)

    x = torch.stack(x, dim=-1)
    return x


def krylov(L, A, b, c=None, return_power=False):
    """Krylov matrix ``(b, A b, ..., A^{L-1} b)``, shape (..., N, L), by repeated squaring.

    With ``c`` returns ``c @ x``, shape (..., L); with ``return_power`` also
    ``A^{L-1}``. For ``L == 1`` the result is ``b[..., None]``, not broadcast
    against the batch dimensions of ``A``.
    """
    x = b.unsqueeze(-1) # (..., N, 1)
    A_ = A

    AL = None
    if return_power:
        AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        _L = L-1

    done = L == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1: AL = A_ @ AL
            _L //= 2

        # The last iteration only needs the first L - l new columns.
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L-l]
        else: _x = x

        _x = A_ @ _x
        x = torch.cat([x, _x], dim=-1)
        if not done: A_ = A_ @ A_

    assert x.shape[-1] == L, f"krylov: expected last dim {L}, got x.shape={tuple(x.shape)}"

    if c is not None:
        x = torch.einsum('...nl, ...n -> ...l', x, c)
    x = x.contiguous()
    if return_power:
        return x, AL
    else:
        return x

def power(L, A, v=None):
    """``A^L``, and with ``v`` (shape (..., N, L)) also the scan ``sum_i A^i v[..., i]``.

    A : (..., N, N).
    """

    I = torch.eye(A.shape[-1]).to(A)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # A reverse divide-and-conquer reduction: fewer padding steps for non-po2 L
    # and more contiguous arrays than the forward one.

    # Fold the columns past the largest power of 2 (a no-op when L is a power of 2).
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Reduction for a power of 2.
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)
