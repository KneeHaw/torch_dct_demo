import torch

def prod(x):
    assert len(x) > 0
    if len(x) == 1:
        return x[0]
    return x[0] * prod(x[1:])

totals = {"muls": 0, "cos": 0}

def dct_matrix(N, norm=None, inverse=False, dtype=torch.float32, device=None):
    """
    Forward: y = x @ C.T
    Inverse: x = y @ C   (same matrix, different row scaling when norm=None)
    """
    n = torch.arange(N, dtype=dtype, device=device)
    k = n.unsqueeze(1)

    if norm == "ortho":
        scale = torch.full((N,), (1 / (2 * N)) ** 0.5, dtype=dtype, device=device)
        scale[0] = (1 / (4 * N)) ** 0.5
    elif inverse:
        scale = torch.full((N,), 1 / (2 * N), dtype=dtype, device=device)
        scale[0] = 1 / (4 * N)
    else:
        scale = torch.ones(N, dtype=dtype, device=device)

    C = scale.unsqueeze(1) * (2 * torch.cos(torch.pi * k * (2 * n + 1) / (2 * N)))  # (k, n)
    totals["muls"] += 2 * N * N
    totals["cos"] += N * N
    return C

def dct_1d(x, norm, axis, inverse=False):
    C = dct_matrix(x.shape[axis], norm, inverse, x.dtype, x.device)
    M = C if inverse else C.T
    y = x.transpose(axis, -1) @ M
    totals["muls"] += prod(y.shape) * x.shape[axis]   # each output elem: N multiplies
    return y.transpose(axis, -1)

def _transform_Nd(x, norm, axis, inverse):
    assert norm in (None, "ortho")
    assert axis >= 0
    if axis >= x.ndim:                                # base case
        return x
    return _transform_Nd(dct_1d(x, norm, axis, inverse), norm, axis + 1, inverse)

def dct_Nd(x, norm=None, axis=0):
    return _transform_Nd(x, norm, axis, inverse=False)

def idct_Nd(x, norm=None, axis=0):
    return _transform_Nd(x, norm, axis, inverse=True)

def reset_totals():
    for key in totals:
        totals[key] = 0

if __name__ == "__main__":
    for shape in [[100], [480, 680], [3, 480, 480]]:
        print(f"Processing shape {shape}")
        x = torch.rand(shape, dtype=torch.float64)

        for norm in (None, "ortho"):
            reset_totals()
            coeffs = dct_Nd(x, norm)
            recon = idct_Nd(coeffs, norm)
            items = [f"{k}: {v:,}" for k,v in totals.items()]
            print(f"    norm={norm} | MSE = {torch.mean((recon - x) ** 2).item():.3e}"
                  f"| , {items[0]} | , {items[1]}")

            try:
                import scipy.fft
                ref = torch.from_numpy(scipy.fft.dctn(x.numpy(), type=2, norm=norm))
                ref_inv = torch.from_numpy(scipy.fft.idctn(ref.numpy(), type=2, norm=norm))
                print(f"    matches scipy fwd? {torch.allclose(coeffs, ref)}, "
                      f"inv? {torch.allclose(idct_Nd(ref, norm), ref_inv)}")
            except ImportError:
                pass