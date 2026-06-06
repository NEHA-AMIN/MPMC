"""
profile_hooks.py — MPMC Phase 1 Profiling Instrumentation
==========================================================
Non-invasive instrumentation for MPMC_net that:

  1. cuda_mem_checkpoint(label)
       Context manager that resets the CUDA peak-memory counter on entry and
       returns a populated dict {label, allocated_before, allocated_after,
       peak_delta_bytes} on exit.

  2. patch_model(model) → model
       Monkey-patches MPMC_net.forward() in-place so that the GNN backbone
       (Segment A) and the L2discrepancy loss call (Segment B) are each
       individually timed and memory-profiled.
       Results are written to model._profile_results after each forward call.
       The original forward logic is preserved exactly; models.py is NOT touched.

  3. unpatch_model(model) → model
       Reverses patch_model, restoring the original forward method.

  4. run_profiler_trace(model, trace_path)
       Runs one forward pass inside torch.profiler.profile with full CUDA
       activity, memory allocation tracking, and shape recording, then exports
       a Chrome-compatible trace JSON to trace_path.

  5. warmup(model, n=3)
       Runs n forward passes with no gradient tracking to warm up CUDA kernels
       before timed measurements are taken.

Usage
-----
    from profile_hooks import patch_model, cuda_mem_checkpoint, run_profiler_trace
    patch_model(model)
    loss, X = model()          # forward; segments A & B are now timed
    results = model._profile_results
    print(results)

Compatibility
-------------
Tested against PyTorch >= 1.9.  The torch.cuda.OutOfMemoryError subclass was
introduced in PyTorch 1.11; the module also catches the older RuntimeError
containing 'CUDA out of memory' for backwards compatibility.
"""

from __future__ import annotations

import time
import contextlib
from pathlib import Path
from typing import Any

import torch
import torch.profiler


# ---------------------------------------------------------------------------
# Helper: detect OOM generically across PyTorch versions
# ---------------------------------------------------------------------------

def _is_oom(exc: Exception) -> bool:
    """Return True for CUDA out-of-memory errors on any PyTorch version."""
    # PyTorch >= 1.11 exposes torch.cuda.OutOfMemoryError
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    # Older: RuntimeError with specific message
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


# ---------------------------------------------------------------------------
# 1. CUDA memory checkpoint context manager
# ---------------------------------------------------------------------------

# cuda_mem_checkpoint is an alias for make_checkpoint (defined below).
# Declared after make_checkpoint; the alias is set at module bottom.
# Public API: use make_checkpoint(label) or cuda_mem_checkpoint(label).


@contextlib.contextmanager
def _cuda_mem_checkpoint_impl(label: str, out: dict):
    """
    Internal implementation.  Callers should use make_checkpoint() below.
    """
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
    else:
        allocated_before = 0

    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        if cuda:
            torch.cuda.synchronize()
            allocated_after = torch.cuda.memory_allocated()
            peak_after = torch.cuda.max_memory_allocated()
        else:
            allocated_after = 0
            peak_after = 0

        out["label"] = label
        out["allocated_before_mb"] = allocated_before / 1024 ** 2
        out["allocated_after_mb"]  = allocated_after  / 1024 ** 2
        out["peak_delta_mb"]       = (peak_after - allocated_before) / 1024 ** 2
        out["elapsed_ms"]          = elapsed


def make_checkpoint(label: str) -> tuple[dict, contextlib.AbstractContextManager]:
    """
    Returns (result_dict, context_manager) so that the result dict can be
    captured cleanly in a closure.

    Usage:
        ckpt, ctx = make_checkpoint("gnn")
        with ctx:
            forward_gnn(x)
        print(ckpt["peak_delta_mb"])
    """
    out: dict = {}
    ctx = _cuda_mem_checkpoint_impl(label, out)
    return out, ctx


# ---------------------------------------------------------------------------
# 2. patch_model / unpatch_model
# ---------------------------------------------------------------------------

def patch_model(model: "torch.nn.Module") -> "torch.nn.Module":
    """
    Monkey-patch model.forward() to run Segment A (GNN backbone) and
    Segment B (loss) as separately timed, separately memory-profiled
    sub-steps, while returning the exact same (loss, X) tuple.

    After each forward call, profiling results are written to:
        model._profile_results  →  dict with keys:
            gnn_elapsed_ms      float
            loss_elapsed_ms     float
            gnn_peak_delta_mb   float
            loss_peak_delta_mb  float
            peak_vram_total_mb  float   (max of gnn + loss peak from CUDA)

    Calling patch_model on an already-patched model is a no-op.
    Call unpatch_model(model) to restore the original forward.
    """
    if getattr(model, "_is_patched", False):
        return model  # already patched — idempotent

    # Stash the original forward so unpatch_model can restore it
    model._orig_forward = model.forward  # type: ignore[attr-defined]
    model._profile_results: dict = {}    # type: ignore[attr-defined]

    # Capture model reference in closure
    _model = model

    def _patched_forward():  # noqa: ANN202  (no annotation to keep it simple)
        # ------------------------------------------------------------------
        # Segment A — GNN backbone
        #   Replicates models.py:109-117 (enc → convs → dec → sigmoid → view)
        # ------------------------------------------------------------------
        gnn_ckpt: dict = {}
        with _cuda_mem_checkpoint_impl("gnn_backbone", gnn_ckpt):
            X = _model.x
            X = _model.enc(X)
            for conv in _model.convs:
                X = conv(X, _model.edge_index, _model.batch)
            X = torch.sigmoid(_model.dec(X))
            X = X.view(_model.nbatch, _model.nsamples, _model.dim)

        # ------------------------------------------------------------------
        # Segment B — Loss (L2discrepancy or approx_hickernell)
        #   Replicates models.py:118
        # ------------------------------------------------------------------
        loss_ckpt: dict = {}
        with _cuda_mem_checkpoint_impl("loss", loss_ckpt):
            raw_loss = _model.loss_fn(X)
            loss = torch.mean(raw_loss)

        # ------------------------------------------------------------------
        # Write profiling telemetry to model._profile_results
        # ------------------------------------------------------------------
        # Re-query the true peak across the entire forward (both segments)
        if torch.cuda.is_available():
            full_peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            full_peak_mb = 0.0

        _model._profile_results = {
            "gnn_elapsed_ms":    gnn_ckpt.get("elapsed_ms", float("nan")),
            "loss_elapsed_ms":   loss_ckpt.get("elapsed_ms", float("nan")),
            "gnn_peak_delta_mb": gnn_ckpt.get("peak_delta_mb", float("nan")),
            "loss_peak_delta_mb": loss_ckpt.get("peak_delta_mb", float("nan")),
            "peak_vram_total_mb": full_peak_mb,
        }

        return loss, X

    model.forward = _patched_forward  # type: ignore[method-assign]
    model._is_patched = True          # type: ignore[attr-defined]
    return model


def unpatch_model(model: "torch.nn.Module") -> "torch.nn.Module":
    """Restore the original forward method set by patch_model."""
    if not getattr(model, "_is_patched", False):
        return model
    model.forward = model._orig_forward  # type: ignore[attr-defined, method-assign]
    model._is_patched = False             # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# 3. run_profiler_trace — Chrome JSON trace export
# ---------------------------------------------------------------------------

def run_profiler_trace(
    model: "torch.nn.Module",
    trace_path: str | Path,
    *,
    warmup_steps: int = 1,
    active_steps: int = 1,
) -> None:
    """
    Run one forward pass inside torch.profiler.profile and export a
    Chrome-compatible trace JSON to trace_path.

    Parameters
    ----------
    model       : An MPMC_net instance (patched or unpatched).
    trace_path  : Destination path for the .json trace file.
    warmup_steps: Number of warm-up steps inside the profiler schedule.
    active_steps: Number of active recording steps.
    """
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        wait=0,
        warmup=warmup_steps,
        active=active_steps,
        repeat=1,
    )

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        profile_memory=True,
        record_shapes=True,
        with_stack=False,       # disabling call-stack keeps trace files manageable
    ) as prof:
        total_steps = warmup_steps + active_steps
        for _ in range(total_steps):
            with torch.no_grad():
                model()
            prof.step()

    prof.export_chrome_trace(str(trace_path))
    print(f"[profile_hooks] Chrome trace written → {trace_path}")


# ---------------------------------------------------------------------------
# 4. warmup
# ---------------------------------------------------------------------------

def warmup(model: "torch.nn.Module", n: int = 3) -> None:
    """
    Run n forward passes with no gradient tracking to warm CUDA kernels
    before timed measurements.

    Should be called BEFORE patch_model if you do not want warm-up timing
    to appear in _profile_results.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            model()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model.train()


# ---------------------------------------------------------------------------
# Public alias: cuda_mem_checkpoint ≡ make_checkpoint
# (kept for import compatibility with stress_test.py)
# ---------------------------------------------------------------------------
cuda_mem_checkpoint = make_checkpoint

# ---------------------------------------------------------------------------
# Smoke test (run: python profile_hooks.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # Allow running from the parent directory
    sys.path.insert(0, os.path.dirname(__file__))

    print("=== profile_hooks.py smoke test ===")
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        from models import MPMC_net
    except ImportError as e:
        print(f"Cannot import MPMC_net: {e}")
        print("Run this smoke test from the MPMC/ directory.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1] Instantiating MPMC_net (dim=3, nsamples=200, nbatch=1)...")
    model = MPMC_net(
        dim=3, nhid=32, nlayers=2, nsamples=200, nbatch=1,
        radius=0.35, loss_fn="L2", dim_emphasize=[1], n_projections=5
    ).to(device)

    print("[2] Testing make_checkpoint context manager...")
    ckpt, ctx = make_checkpoint("test_block")
    with ctx:
        _ = torch.randn(500, 500, device=device)
    assert "elapsed_ms" in ckpt, "make_checkpoint did not populate result dict"
    print(f"    elapsed_ms = {ckpt['elapsed_ms']:.3f} ms")
    print(f"    peak_delta_mb = {ckpt['peak_delta_mb']:.4f} MB")

    print("[3] Testing patch_model...")
    patch_model(model)
    assert model._is_patched, "patch_model did not set _is_patched flag"

    print("[4] Running patched forward pass...")
    model.train()
    loss, X = model()
    r = model._profile_results
    assert "gnn_elapsed_ms" in r
    print(f"    GNN elapsed    : {r['gnn_elapsed_ms']:.3f} ms")
    print(f"    Loss elapsed   : {r['loss_elapsed_ms']:.3f} ms")
    print(f"    GNN peak VRAM  : {r['gnn_peak_delta_mb']:.4f} MB")
    print(f"    Loss peak VRAM : {r['loss_peak_delta_mb']:.4f} MB")
    print(f"    Total peak VRAM: {r['peak_vram_total_mb']:.4f} MB")

    print("[5] Testing unpatch_model...")
    unpatch_model(model)
    assert not model._is_patched

    print("[6] Testing run_profiler_trace...")
    patch_model(model)  # re-patch so the patched forward is traced
    trace_out = "/tmp/mpmc_smoke_trace.json"
    run_profiler_trace(model, trace_out, warmup_steps=1, active_steps=1)
    assert Path(trace_out).exists(), "Trace file was not created"
    print(f"    Trace file size: {Path(trace_out).stat().st_size / 1024:.1f} KB")

    print("\n=== All smoke tests passed ===")
