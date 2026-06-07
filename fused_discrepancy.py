"""
fused_discrepancy.py — Fused Triton Kernel for Warnock L2-Star Discrepancy
==========================================================================
Eliminates the O(N^2 * d) pairwise tensor from MPMC models.py:102.

Formula:  D^2 = 3^(-d) - (1/N)*2^(1-d)*Sum_i Prod_k(1-x_ik^2)
                       + (1/N^2)*Sum_i Sum_j Prod_k(1-max(x_ik,x_jk))

          loss[b] = sqrt(D^2[b])

Baseline: O(N^2*d) memory — (nbatch,N,N,d) broadcast at models.py:102
Kernel:   O(N) memory     — only (nbatch,N) row_sums written to global mem

Drop-in for models.py:68:
    from fused_discrepancy import fused_l2_discrepancy
    self.loss_fn = fused_l2_discrepancy

Known limitations:
    - CUDA only.
    - approx_hickernell (models.py:87,92) hardcodes self.L2discrepancy —
      NOT intercepted by line-68 drop-in. Rebind explicitly if needed.
    - radius_graph (models.py:65) not addressed (~20s at N=100k, d=50).

Triton notes:
    - Power-of-2 constraint applies to tl.arange() only (vector creation),
      NOT to range() (loop iteration). All k-loops use range(d) directly.
    - BLOCK=16 to keep PTX size manageable. At BLOCK=32 with d=50,
      ptxas gets OOM-killed on Colab T4 (65K unrolled ops -> huge PTX).
    - Backward kernel tested only at small d (gradcheck N=50,d=3).
      Backward at d=50 generates O(BLOCK^2 * d^2) unrolled ops which
      may exceed ptxas memory. Forward at d=50 works fine.
"""

from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Block sizes — BLOCK=16 keeps PTX small enough for ptxas on Colab T4.
#
# Forward  at d=50: 16*16*50      =  12,800 unrolled ops  -> ~1 min compile
# Backward at d=3:  16*16*3*3     =   2,304 unrolled ops  -> ~30s compile
# Backward at d=50: 16*16*50*50   = 640,000 unrolled ops  -> will OOM ptxas
#   (backward at d=50 not needed for validation; gradcheck runs at d=3)
# ---------------------------------------------------------------------------
_BLOCK_M: int = 16
_BLOCK_N: int = 16


# ---------------------------------------------------------------------------
# Sub-Task 1 — Forward Kernel
# ---------------------------------------------------------------------------

@triton.jit
def _fused_l2_discrepancy_kernel(
    X_ptr,
    RS_ptr,          # (nbatch, N) row_sums for Term 3
    S1_ptr,          # (nbatch, N) per-row products for Term 2
    stride_xb, stride_xn, stride_xd,
    stride_rb, stride_rn,
    N,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_m   = tl.program_id(0)
    tile_n   = tl.program_id(1)
    batch_id = tl.program_id(2)

    row_start = tile_m * BLOCK_M
    col_start = tile_n * BLOCK_N
    X_batch   = X_ptr + batch_id * stride_xb

    # -------------------------------------------------------------------
    # Term 2: diagonal tiles only — Prod_k(1 - x_ik^2) per row i
    # -------------------------------------------------------------------
    if tile_m == tile_n:
        for mi in range(BLOCK_M):
            row_i = row_start + mi
            if row_i < N:
                prod2 = 1.0
                ptr_i = X_batch + row_i * stride_xn
                for k in range(d):
                    xik = tl.load(ptr_i + k * stride_xd)
                    prod2 *= (1.0 - xik * xik)
                tl.atomic_add(
                    S1_ptr + batch_id * stride_rb + row_i * stride_rn,
                    prod2,
                )

    # -------------------------------------------------------------------
    # Term 3: all tiles — Sum_j Prod_k(1 - max(x_ik, x_jk))
    # -------------------------------------------------------------------
    for mi in range(BLOCK_M):
        row_i = row_start + mi
        if row_i < N:
            row_sum = 0.0
            ptr_i = X_batch + row_i * stride_xn
            for ni in range(BLOCK_N):
                col_j = col_start + ni
                if col_j < N:
                    prod3 = 1.0
                    ptr_j = X_batch + col_j * stride_xn
                    for k in range(d):
                        xik = tl.load(ptr_i + k * stride_xd)
                        xjk = tl.load(ptr_j + k * stride_xd)
                        prod3 *= (1.0 - tl.maximum(xik, xjk))
                    row_sum += prod3
            tl.atomic_add(
                RS_ptr + batch_id * stride_rb + row_i * stride_rn,
                row_sum,
            )


# ---------------------------------------------------------------------------
# Sub-Task 2 — Backward Kernel
# ---------------------------------------------------------------------------

@triton.jit
def _fused_l2_discrepancy_backward_kernel(
    X_ptr,
    dL_ptr,          # (nbatch,) upstream gradient (chain-rule scaled)
    dX_ptr,          # (nbatch, N, d) output gradient (pre-zeroed)
    stride_xb, stride_xn, stride_xd,
    stride_db, stride_dn, stride_dd,
    stride_lb,
    N,
    d:       tl.constexpr,
    coef_t2: tl.constexpr,   # (1/N) * 2^(2-d)
    inv_N2:  tl.constexpr,   # 1/N^2
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Term 2: g2[i,k] = -coef_t2 * x_ik * Prod_{l!=k}(1 - x_il^2)
    Term 3: g3[i,k] += -2/N^2 * 1(x_ik>=x_jk) * Prod_{l!=k}(1-max(x_il,x_jl))
    Both use product-without-k via explicit inner loop (no division trick).
    """
    tile_m   = tl.program_id(0)
    tile_n   = tl.program_id(1)
    batch_id = tl.program_id(2)

    row_start = tile_m * BLOCK_M
    col_start = tile_n * BLOCK_N
    X_batch   = X_ptr  + batch_id * stride_xb
    dX_batch  = dX_ptr + batch_id * stride_db

    upstream = tl.load(dL_ptr + batch_id * stride_lb)

    # -------------------------------------------------------------------
    # Term 2 backward — diagonal tiles only
    # -------------------------------------------------------------------
    if tile_m == tile_n:
        for mi in range(BLOCK_M):
            row_i = row_start + mi
            if row_i < N:
                ptr_i = X_batch + row_i * stride_xn
                for k in range(d):
                    xik = tl.load(ptr_i + k * stride_xd)
                    p_wk = 1.0
                    for l in range(d):
                        if l != k:
                            xil = tl.load(ptr_i + l * stride_xd)
                            p_wk *= (1.0 - xil * xil)
                    g2 = (coef_t2) * xik * p_wk * upstream
                    tl.atomic_add(
                        dX_batch + row_i * stride_dn + k * stride_dd,
                        g2,
                    )

    # -------------------------------------------------------------------
    # Term 3 backward — all tiles
    # -------------------------------------------------------------------
    coef_t3 = -2.0 * inv_N2
    for mi in range(BLOCK_M):
        row_i = row_start + mi
        if row_i < N:
            ptr_i = X_batch + row_i * stride_xn
            for k in range(d):
                g3_ik = 0.0
                xik = tl.load(ptr_i + k * stride_xd)
                for ni in range(BLOCK_N):
                    col_j = col_start + ni
                    if col_j < N:
                        ptr_j = X_batch + col_j * stride_xn
                        xjk = tl.load(ptr_j + k * stride_xd)
                        tie = tl.where(xik >= xjk, 1.0, 0.0)
                        p_wk3 = 1.0
                        for l in range(d):
                            if l != k:
                                xil = tl.load(ptr_i + l * stride_xd)
                                xjl = tl.load(ptr_j + l * stride_xd)
                                p_wk3 *= (1.0 - tl.maximum(xil, xjl))
                        g3_ik += tie * p_wk3
                g3_ik *= coef_t3 * upstream
                tl.atomic_add(
                    dX_batch + row_i * stride_dn + k * stride_dd,
                    g3_ik,
                )


# ---------------------------------------------------------------------------
# Sub-Task 3 — PyTorch Autograd Wrapper
# ---------------------------------------------------------------------------

class FusedL2Discrepancy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X: torch.Tensor) -> torch.Tensor:
        if not X.is_cuda:
            raise RuntimeError("FusedL2Discrepancy requires a CUDA tensor.")
        if not X.is_contiguous():
            X = X.contiguous()
        if X.dtype != torch.float32:
            X = X.to(torch.float32)

        nbatch, N, d = X.shape

        T1          = math.pow(3.0, -d)
        coef_t2_fwd = math.pow(2.0, 1.0 - d) / N
        coef_t2_bwd = math.pow(2.0, 2.0 - d) / N
        inv_N2      = 1.0 / (N * N)

        row_sums = torch.zeros(nbatch, N, device=X.device, dtype=torch.float32)
        sum1_buf = torch.zeros(nbatch, N, device=X.device, dtype=torch.float32)

        grid = (
            triton.cdiv(N, _BLOCK_M),
            triton.cdiv(N, _BLOCK_N),
            nbatch,
        )

        _fused_l2_discrepancy_kernel[grid](
            X, row_sums, sum1_buf,
            X.stride(0), X.stride(1), X.stride(2),
            row_sums.stride(0), row_sums.stride(1),
            N, d,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
        )

        sum2 = row_sums.sum(dim=1)
        sum1 = sum1_buf.sum(dim=1)
        D2   = T1 - coef_t2_fwd * sum1 + inv_N2 * sum2
        loss = torch.sqrt(D2.clamp(min=0.0))

        ctx.save_for_backward(X, loss)
        ctx.N           = N
        ctx.d           = d
        ctx.nbatch      = nbatch
        ctx.inv_N2      = inv_N2
        ctx.coef_t2_bwd = coef_t2_bwd

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        X, loss = ctx.saved_tensors
        N       = ctx.N
        d       = ctx.d
        nbatch  = ctx.nbatch
        inv_N2  = ctx.inv_N2
        coef_t2 = ctx.coef_t2_bwd

        dL_scaled = grad_output / (2.0 * loss.clamp(min=1e-12))
        dX = torch.zeros_like(X)

        grid = (
            triton.cdiv(N, _BLOCK_M),
            triton.cdiv(N, _BLOCK_N),
            nbatch,
        )

        _fused_l2_discrepancy_backward_kernel[grid](
            X, dL_scaled, dX,
            X.stride(0),  X.stride(1),  X.stride(2),
            dX.stride(0), dX.stride(1), dX.stride(2),
            dL_scaled.stride(0),
            N, d,
            coef_t2, inv_N2,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
        )

        return dX


def fused_l2_discrepancy(X: torch.Tensor) -> torch.Tensor:
    """Drop-in for MPMC_net.L2discrepancy (models.py:96-107)."""
    return FusedL2Discrepancy.apply(X)


# ---------------------------------------------------------------------------
# Sub-Task 4 — Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    print("=" * 68)
    print(" fused_discrepancy.py — Validation Suite")
    print("=" * 68)

    if not torch.cuda.is_available():
        print("  ERROR: CUDA not available.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"  Device : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  BLOCK  : {_BLOCK_M} x {_BLOCK_N}")
    print()

    t_start = time.time()

    def _ref_l2discrepancy(x):
        """Baseline — matches models.py:96-107."""
        N   = x.size(1)
        dim = x.size(2)
        prod1        = 1.0 - x ** 2.0
        prod1        = torch.prod(prod1, dim=2)
        sum1         = torch.sum(prod1, dim=1)
        pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :])
        product      = torch.prod(1.0 - pairwise_max, dim=3)
        sum2         = torch.sum(product, dim=(1, 2))
        return torch.sqrt(
            math.pow(3.0, -dim)
            - (1.0 / N) * math.pow(2.0, 1.0 - dim) * sum1
            + (1.0 / N ** 2) * sum2
        )

    P = "\033[32mPASS\033[0m"
    F = "\033[31mFAIL\033[0m"
    ok_all = True

    # Test 1 — Forward parity N=1000, d=3
    print("[1/4] Forward parity  N=1000, d=3  (atol=1e-5)")
    print("      (first call compiles kernel — expect ~30-60s)")
    t1 = time.time()
    torch.manual_seed(42)
    Xr = torch.rand(1, 1000, 3, device=device)
    ref = _ref_l2discrepancy(Xr)
    fus = fused_l2_discrepancy(Xr.clone())
    torch.cuda.synchronize()
    d1 = abs(fus.item() - ref.item())
    ok = torch.allclose(fus, ref, atol=1e-5)
    print(f"      {P if ok else F}  ref={ref.item():.8f}  fused={fus.item():.8f}  "
          f"delta={d1:.2e}  ({time.time()-t1:.1f}s)")
    if not ok: ok_all = False

    # Test 2 — Forward parity grid
    print("[2/4] Forward parity grid  N in {100,500,1000}  d in {3,6,20,50}")
    t2 = time.time()
    grid_ok = True
    for n in [100, 500, 1000]:
        for dd in [3, 6, 20, 50]:
            torch.manual_seed(0)
            Xg = torch.rand(1, n, dd, device=device)
            r = _ref_l2discrepancy(Xg)
            f = fused_l2_discrepancy(Xg)
            torch.cuda.synchronize()
            if not torch.allclose(f, r, atol=1e-5):
                print(f"      {F}  N={n} d={dd}  ref={r.item():.7f}  "
                      f"fused={f.item():.7f}  delta={abs(f.item()-r.item()):.2e}")
                grid_ok = False
                ok_all = False
    if grid_ok:
        print(f"      {P}  All 12 cells matched  ({time.time()-t2:.1f}s)")

    # Test 3 — Gradient check N=50, d=3
    print("[3/4] Gradient check  N=50, d=3  (eps=1e-3, atol=1e-2)")
    print("      (first backward compiles backward kernel — expect ~30-60s)")
    t3 = time.time()
    torch.manual_seed(7)
    Xgc = torch.rand(1, 50, 3, device=device, dtype=torch.float64,
                     requires_grad=True)

    def _f64(X):
        return FusedL2Discrepancy.apply(X.float()).double()

    try:
        torch.autograd.gradcheck(
            _f64, (Xgc,), eps=1e-3, atol=1e-2, rtol=1e-2,
        )
        print(f"      {P}  gradcheck passed  ({time.time()-t3:.1f}s)")
    except Exception as e:
        print(f"      {F}  {e}")
        ok_all = False

    # Test 4 — VRAM at N=5000, d=50 (forward-only, baseline OOM cell)
    print("[4/4] VRAM benchmark  N=5000, d=50  forward-only (baseline OOM)")
    t4 = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    Xv = torch.rand(1, 5000, 50, device=device)
    try:
        with torch.no_grad():
            lv = fused_l2_discrepancy(Xv)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"      {P}  Peak VRAM = {peak:.1f} MB  "
              f"(baseline ~5000 MB -> OOM)")
        print(f"             loss = {lv.item():.6f}  ({time.time()-t4:.1f}s)")
    except Exception as e:
        print(f"      {F}  {e}")
        ok_all = False

    elapsed = time.time() - t_start
    print()
    print("=" * 68)
    if ok_all:
        print(f"  All tests {P}  (total {elapsed:.0f}s)")
    else:
        print(f"  {F} — see above  (total {elapsed:.0f}s)")
        sys.exit(1)
    print("=" * 68)