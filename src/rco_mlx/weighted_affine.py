"""Imatrix-weighted affine quantization with a least-squares refit.

This mirrors what oMLX's ``enhanced=True`` (``_weighted_affine_quantize``)
actually does -- note that oQe is NOT GPTQ: the oMLX package contains no
Hessian compensation at all (no ``H_inv``, no Cholesky, no actorder).  Its two
steps are:

1. search per-group ``(scale, bias)`` over a grid of factors applied to an
   edge-anchored min/max parameterization, keeping the imatrix-weighted error
   minimum.  Clipping is implicit: a smaller scale with the bias pinned at an
   edge maps the out-of-range values onto the end codes.
2. refit ``(scale, bias)`` by weighted least squares given the codes, and
   accept the refit only where it lowers the weighted error.

Output layout matches ``mlx.core.quantize(..., mode="affine")``: ``q`` packed
uint32, ``scales``/``biases`` ``[out, d/gs]`` and ``dequant = q*scale + bias``.
"""
from __future__ import annotations

import numpy as np

PACK_CACHE: dict[int, np.ndarray] = {}


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack ``[out, d]`` uint8 codes into MLX's uint32 layout.

    Each ROW is an independent contiguous bitstream: value ``j`` of a row
    occupies bits ``[j*bits, (j+1)*bits)`` of that row's packed words, crossing
    uint32 boundaries whenever ``bits`` does not divide 32.  A per-word layout
    (``32//bits`` values per word) is only equivalent for 2/4/8 bit; for 3 and
    6 bit it gives wrong shapes *and* wrong values.
    """
    out, d = codes.shape
    shifts = np.arange(bits, dtype=np.uint32)
    bit_arr = ((codes.astype(np.uint32)[:, :, None] >> shifts) & np.uint32(1))
    bit_arr = bit_arr.astype(np.uint8).reshape(out, d * bits)
    pad = (-d * bits) % 32
    if pad:
        bit_arr = np.pad(bit_arr, ((0, 0), (0, pad)))
    bytes_ = np.packbits(bit_arr, bitorder="little", axis=1).astype(np.uint32)
    bytes_ = np.pad(bytes_, ((0, 0), (0, (-bytes_.shape[1]) % 4)))
    packed = (
        bytes_[:, 0::4]
        | (bytes_[:, 1::4] << 8)
        | (bytes_[:, 2::4] << 16)
        | (bytes_[:, 3::4] << 24)
    )
    return packed.astype(np.uint32)


def unpack_codes(packed: np.ndarray, bits: int, d: int) -> np.ndarray:
    """Inverse of :func:`pack_codes` (used by the tests and the error DB)."""
    b = np.asarray(packed, dtype=np.uint32)
    out = b.shape[0]
    bytes_ = np.empty((out, b.shape[1] * 4), dtype=np.uint8)
    bytes_[:, 0::4] = (b & np.uint32(0xFF)).astype(np.uint8)
    bytes_[:, 1::4] = ((b >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
    bytes_[:, 2::4] = ((b >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
    bytes_[:, 3::4] = ((b >> np.uint32(24)) & np.uint32(0xFF)).astype(np.uint8)
    bit_arr = np.unpackbits(bytes_, axis=1, bitorder="little")[:, : d * bits]
    bit_arr = bit_arr.reshape(out, d, bits)
    # bit j of value i is at stream position i*bits + j -> weight 2**j
    weights = (np.uint32(1) << np.arange(bits, dtype=np.uint32)).astype(np.uint32)
    return (bit_arr.astype(np.uint32) * weights).sum(axis=2).astype(np.uint8)


def _weighted_lsq(
    w: np.ndarray, codes: np.ndarray, imp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form weighted fit of ``w ~ s*c + b`` per (row, group).

    ``w``/``codes`` are ``[out, g, gs]``; ``imp`` is ``[1, g, gs]``.
    """
    cw = imp * codes
    sum_w = cw.sum(axis=2, keepdims=True) + 1e-12
    mean_c = cw.sum(axis=2, keepdims=True) / sum_w
    mean_v = (cw * w).sum(axis=2, keepdims=True) / sum_w
    cov = (cw * (codes - mean_c) * (w - mean_v)).sum(axis=2, keepdims=True)
    var = (cw * (codes - mean_c) ** 2).sum(axis=2, keepdims=True) + 1e-12
    s = cov / var
    b = mean_v - s * mean_c
    return s, b


def weighted_affine_quantize(
    w: np.ndarray,
    col_w: np.ndarray,
    bits: int,
    gs: int,
    grid: tuple[float, ...] = (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25),
    refit_iters: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(packed_codes, scales, biases)`` in MLX affine layout."""
    w = np.asarray(w, dtype=np.float32)
    out, d = w.shape
    if d % gs:
        raise ValueError(f"last dim {d} not divisible by group size {gs}")
    g = d // gs
    wg = w.reshape(out, g, gs)
    imp = np.asarray(col_w, dtype=np.float32).reshape(1, g, gs)
    imp = np.maximum(imp, 1e-8)
    n_bins = float((1 << bits) - 1)

    w_max = wg.max(axis=2, keepdims=True)
    w_min = wg.min(axis=2, keepdims=True)
    best_err = np.full((out, g, 1), np.inf, dtype=np.float64)
    best_s = np.empty((out, g, 1), dtype=np.float32)
    best_b = np.empty((out, g, 1), dtype=np.float32)
    best_c = np.zeros((out, g, gs), dtype=np.uint8)

    for edge, opposite, sign in ((w_max, w_min, -1.0), (w_min, w_max, 1.0)):
        raw = np.maximum(np.abs(edge - opposite) / n_bins, 1e-7) * sign
        q0 = np.round(edge / raw)
        scale0 = np.where(q0 != 0, edge / np.where(q0 == 0, 1.0, q0), raw)
        bias0 = np.where(q0 == 0, 0.0, edge)
        for f in grid:
            s = scale0 * np.float32(f)
            b = bias0
            codes = np.clip(np.round((wg - b) / s), 0, n_bins)
            err = (imp * (wg - (codes * s + b)) ** 2).sum(axis=2, keepdims=True)
            take = err < best_err
            best_err = np.where(take, err, best_err)
            best_s = np.where(take, s, best_s)
            best_b = np.where(take, b, best_b)
            best_c = np.where(take, codes, best_c)

    # least-squares refit, accepted only where it lowers the weighted error
    s, b = best_s.copy(), best_b.copy()
    c = best_c.copy()
    tiny = np.float32(1e-7)
    for _ in range(refit_iters):
        s_new, b_new = _weighted_lsq(wg, c, imp)
        safe = np.where(np.abs(s_new) < tiny, tiny, s_new)
        c_new = np.clip(np.round((wg - b_new) / safe), 0, n_bins)
        err_new = (imp * (wg - (c_new * s_new + b_new)) ** 2).sum(axis=2, keepdims=True)
        err_old = (imp * (wg - (c * s + b)) ** 2).sum(axis=2, keepdims=True)
        take = (err_new < err_old) & np.isfinite(err_new) & (np.abs(s_new) >= tiny)
        s = np.where(take, s_new, s)
        b = np.where(take, b_new, b)
        c = np.where(take, c_new, c)

    scales = s.reshape(out, g).astype(np.float32)
    biases = b.reshape(out, g).astype(np.float32)
    return pack_codes(c.reshape(out, d).astype(np.uint8), bits), scales, biases


def weighted_error(w: np.ndarray, recon: np.ndarray, col_w: np.ndarray) -> float:
    imp = np.asarray(col_w, dtype=np.float64).reshape(1, -1)
    return float((imp * (np.asarray(w, np.float64) - np.asarray(recon, np.float64)) ** 2).sum())
