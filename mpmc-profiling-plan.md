# MPMC Phase 1 GPU Stress-Test & Memory Profiling Plan

## Top-Level Overview

**Goal:** Produce two self-contained, copy-pasteable Python scripts that instrument the MPMC repository
to (1) isolate memory and time cost of the GNN message-passing step vs. the dense L2 discrepancy loss,
and (2) sweep a full stress-test grid across `nsamples × dim`, recording peak VRAM, timing, and
OOM events into a CSV and a dual-axis line chart.

**Scope:**
- Two new files: `MPMC/profile_hooks.py` (instrumentation) and `MPMC/stress_test.py` (sweep harness).
- No changes to `models.py`, `utils.py`, or `run_train.py`.
- `radius` fixed at `0.35` (repo default from `run_train.py:86`).
- `nbatch = 1` during profiling (removes batch-level noise; pure spatial scalability curve; `nbatch=16` training thresholds can be extrapolated by multiplying VRAM by 16×). **Confirmed.**
- `nhid = 128`, `nlayers = 3` (defaults from `run_train.py`).

**Non-goals:**
- Actual training, checkpointing, or result output to `outputs/` or `results/`.
- Modifying the `L2discrepancy` algorithm (no fused-kernel optimisation in this phase).
- Multi-GPU or distributed profiling.

---

## Architecture of What Is Being Measured

```
MPMC_net.__init__
  └─ [SEGMENT C] radius_graph(x, r=0.35)  ← TIMED: Graph_build_time_ms
       fixed-radius spatial query; O(N·k) edges at low N, potentially
       O(N²) at high N when all points fall within radius

MPMC_net.forward()
  ├─ [SEGMENT A] GNN backbone
  │    enc → MPNN_layer ×3 → dec → sigmoid → reshape
  └─ [SEGMENT B] L2discrepancy(X)
       sum1: O(N·d) element-wise
       pairwise_max: O(N²·d) broadcast     ← dominant VRAM allocation
       sum2: reduce over (N,N,d) tensor
```

Memory complexity of the pairwise broadcast:
`x[:, :, None, :]` and `x[:, None, :, :]` both expand to shape `(nbatch, N, N, d)`.
At `fp32`, one such tensor costs `nbatch × N² × d × 4` bytes.
At `N=100 000, d=50` this is ~`200 GB` — OOM is expected well before that frontier.

---

## Sub-Tasks

---

### Sub-Task 1 — Dependency Installation Script

**Intent:** Provide a shell-based install block that correctly handles the ABI-coupled build
requirements of `torch-cluster` and `torch-scatter` (C++ extensions that must match the exact
PyTorch version and CUDA version on the VM).

**Expected Outcomes:**
- A Markdown code block (inside `stress_test.py` header comment or a separate `install.sh`) that
  the user can run once on the VM before executing the profiling scripts.
- The block queries `torch.__version__` and `torch.version.cuda` at runtime to pin the wheel URLs.

**Todo List:**
1. Write an `install.sh` that detects the active Python, prints the detected `torch`/`CUDA` version
   pair, and installs `torch-cluster`, `torch-scatter`, `torch-sparse`, `torch-geometric` from the
   correct PyG wheel index (`https://data.pyg.org/whl/`).
2. Add a `# INSTALL` comment block at the top of `stress_test.py` that echoes the same commands so
   the script is self-documenting.

**Relevant Context:**
- README lists `torch==1.9.0`, `torch-cluster==1.5.9`, `torch-scatter==2.0.9`, but these are
  minimum versions; the VM may have a newer PyTorch, so the detection step is mandatory.
- `models.py:4` imports `from torch_cluster import radius_graph` — if this fails the whole script
  will crash before any measurement.

**Status:** [x] done

---

### Sub-Task 2 — Profiling Instrumentation Module (`profile_hooks.py`)

**Intent:** Build a standalone instrumentation module that wraps `MPMC_net.forward()` and the
`L2discrepancy` method with fine-grained CUDA memory and timing checkpoints, without modifying
`models.py`.

**Expected Outcomes:**
- File `MPMC/profile_hooks.py` containing:
  - A context manager `cuda_mem_checkpoint(label)` that records
    `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` before/after a block.
  - A function `patch_model(model)` that monkey-patches `model.forward` to call the GNN backbone
    and the loss separately, inserting `cuda_mem_checkpoint` around each segment.
  - A `run_profiler_trace(model, output_dir)` function that wraps a single forward pass in
    `torch.profiler.profile(activities=[CPU, CUDA], profile_memory=True, record_shapes=True)`
    and exports the Chrome trace JSON.

**Todo List:**
1. Implement `cuda_mem_checkpoint` as a `contextlib.contextmanager` that resets the peak counter
   with `torch.cuda.reset_peak_memory_stats()` on entry and returns a dict `{allocated, peak}` on
   exit.
2. Implement `patch_model(model)` — save `model.forward` as `model._orig_forward`; replace it
   with a new function that:
   a. Runs the GNN backbone (enc → convs → dec → sigmoid → reshape) inside segment A checkpoint.
   b. Calls `model.loss_fn(X)` inside segment B checkpoint.
   c. Returns the same `(loss, X)` tuple as the original.
3. Implement `run_profiler_trace` using `torch.profiler.profile` as a context manager around
   `model()`, saving the trace via `prof.export_chrome_trace(path)`.
4. Keep the module importable with a `if __name__ == '__main__'` smoke test.

**Relevant Context:**
- `MPMC_net.forward` source: `models.py:109-119`. The GNN body is lines 113-117; loss call is
  line 118.
- `MPMC_net.L2discrepancy` source: `models.py:96-107`. The allocation hotspot is line 102:
  `pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :])`.
- The model uses `self.x`, `self.edge_index`, `self.batch` as stored state — the patched forward
  must reference these via `self` / closure over the model instance.

**Status:** [x] done

---

### Sub-Task 3 — Stress-Test Sweep Script (`stress_test.py`)

**Intent:** Build the automated sweep harness that iterates over the full `nsamples × dim` grid,
instantiates a fresh `MPMC_net` per cell, runs forward + backward, catches OOM, and accumulates
results into a structured in-memory list.

**Expected Outcomes:**
- File `MPMC/stress_test.py` with a `run_stress_test()` function that:
  - Loops over the Cartesian product of `nsamples` and `dim` grids.
  - Per cell: instantiates `MPMC_net`, calls `patch_model` from `profile_hooks.py`, runs the
    patched forward, runs `loss.backward()`, and records the six target metrics.
  - Catches `torch.cuda.OutOfMemoryError` (and the older `RuntimeError` containing "CUDA out of
    memory") gracefully — sets `OOM_Triggered = True` and fills timing/memory fields with `NaN`.
  - Calls `torch.cuda.empty_cache()` and `gc.collect()` between cells.
  - Returns a `list[dict]` of result rows.

**Grid Definition (fixed):**
```python
NSAMPLES_GRID = [1000, 5000, 10000, 25000, 50000, 100000]
DIM_GRID      = [3, 6, 20, 50]
RADIUS        = 0.35
NBATCH        = 1
NHID          = 128
NLAYERS       = 3
```

**Metrics recorded per row (7 columns):**
| Field | Segment | Source |
|---|---|---|
| `nsamples` | — | loop variable |
| `dim` | — | loop variable |
| `Graph_build_time_ms` | C (`__init__`) | `time.perf_counter` wrapping `radius_graph(...)` call at `models.py:65` |
| `GNN_forward_time_ms` | A (`forward`) | `time.perf_counter` around backbone enc→convs→dec→sigmoid |
| `Loss_time_ms` | B (`forward`) | `time.perf_counter` around `model.loss_fn(X)` call |
| `Peak_VRAM_Allocated_MB` | A+B combined | `cuda_mem_checkpoint` peak ÷ 1024² |
| `OOM_Triggered` | C or A or B | `True` if any exception caught during init or forward |

**Todo List:**
1. Import `MPMC_net` from `models` (add `MPMC/` to `sys.path` if running from parent dir).
2. Import `patch_model`, `cuda_mem_checkpoint`, `timed_model_init` from `profile_hooks`.
3. Define the grid constants and the result accumulator list.
4. Write the nested loop with per-cell `try/except` blocks.
5. Inside the outer try block (wraps `__init__`):
   a. Start `t0 = time.perf_counter()`.
   b. Instantiate `MPMC_net` with the cell's `(dim, nhid, nlayers, nsamples, nbatch=1, radius,
      loss_fn='L2', dim_emphasize=[1], n_projections=15)` — `radius_graph` fires here.
   c. Record `Graph_build_time_ms = (time.perf_counter() - t0) * 1000`.
6. Inside the inner try block (wraps `forward`):
   a. Apply `patch_model(model)` to split segment A/B timing.
   b. Call `model.train(); optimizer.zero_grad(); loss, X = model()`.
   c. Call `loss.backward()`.
   d. Extract `GNN_forward_time_ms`, `Loss_time_ms`, `Peak_VRAM_Allocated_MB` from patch results.
7. Append the result dict; call cleanup (`del model; torch.cuda.empty_cache(); gc.collect()`).
8. Return the results list.

**Relevant Context:**
- `models.py:44-119` — `MPMC_net` constructor and forward.
- `models.py:65` — `radius_graph` called in `__init__`; at `N=100 000` the spatial query can
  itself become a bottleneck or trigger allocation failure before `forward()` is ever called.
  This is why `Graph_build_time_ms` is captured in a separate outer try block.
- `run_train.py:12-13` — canonical instantiation pattern to follow.
- `nbatch=1` keeps the pairwise tensor at `(1, N, N, d)` for clean per-cell measurement.

**Status:** [x] done

---

### Sub-Task 4 — Data Export and Visualisation (`stress_test.py`, continued)

**Intent:** After the sweep completes, write results to a timestamped CSV and generate a
publication-quality dual-axis matplotlib figure.

**Expected Outcomes:**
- A `save_results(rows, out_dir)` function that writes `stress_test_results_<timestamp>.csv` with
  all seven columns.
- A `plot_results(csv_path)` function that produces a single figure:
  - **Primary y-axis (left):** Peak VRAM Allocated (MB) vs. nsamples — one line per `dim`.
  - **Secondary y-axis (right):** Total forward time (GNN + Loss) in ms vs. nsamples — one
    line per `dim`, dashed.
  - OOM cells marked with a red `×` scatter overlay on the primary axis.
  - Legend, grid, log-scale x-axis, saved as `stress_test_results_<timestamp>.png`.

**Todo List:**
1. Implement `save_results` using `csv.DictWriter` (no pandas dependency).
2. Implement `plot_results`:
   a. Parse the CSV with the standard `csv` module or `pandas` if available.
   b. Group rows by `dim`.
   c. Plot VRAM lines on `ax1`; plot total time lines on `ax2 = ax1.twinx()`.
   d. Scatter red `×` at `(nsamples, peak_vram)` for all OOM rows.
   e. Set `ax1.set_xscale('log')`, axis labels, title, and legend.
   f. `plt.tight_layout(); plt.savefig(png_path, dpi=150)`.
3. Add a `if __name__ == '__main__'` block in `stress_test.py` that calls `run_stress_test()`,
   then `save_results()`, then `plot_results()`.

**Relevant Context:**
- OOM cells will dominate the high-`N`, high-`d` corner — the red marker overlay makes the
  memory cliff visually obvious.
- `pandas` is not guaranteed on the VM; use `csv` as the primary parser with an optional
  `pandas`-based fallback.

**Status:** [x] done

---

### Sub-Task 5 — Final Integration and Smoke Test

**Intent:** Verify the two new files are internally consistent, importable, and runnable with a
tiny synthetic grid before committing to a full GPU sweep.

**Expected Outcomes:**
- A `--smoke` CLI flag in `stress_test.py` that overrides the grid with
  `nsamples=[100, 200]`, `dim=[3, 6]` so the full pipeline can be validated in under 60 s on any
  GPU (or CPU fallback).
- Printed confirmation that CSV and PNG were written successfully.
- No import errors when `python stress_test.py --smoke` is run from `MPMC/`.

**Todo List:**
1. Add `argparse` to `stress_test.py` with a `--smoke` boolean flag.
2. Override grid constants when `--smoke` is True.
3. Add a `--outdir` argument (default `./profiling_outputs/`) that is created with `Path.mkdir`.
4. Print a summary table to stdout after the sweep (nsamples, dim, VRAM MB, OOM).
5. Confirm that `profile_hooks.py` can be imported cleanly (no side-effects at import time).

**Relevant Context:**
- `models.py:5` — `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` means
  the model will silently fall back to CPU if no GPU is present; the profiling script must warn
  the user if CUDA is not available.

**Status:** [x] done

---

## File Manifest After Implementation

```
MPMC/
├── models.py             (unchanged)
├── utils.py              (unchanged)
├── run_train.py          (unchanged)
├── profile_hooks.py      (NEW — instrumentation module)
├── stress_test.py        (NEW — sweep harness, export, visualisation)
├── install.sh            (NEW — dependency installer)
└── profiling_outputs/    (created at runtime)
    ├── stress_test_results_<ts>.csv
    ├── stress_test_results_<ts>.png
    └── trace_<nsamples>_<dim>.json   (Chrome profiler traces)
```

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| `nbatch=1` during profiling | Isolates pure O(N²·d) spatial scalability; `nbatch=16` training VRAM threshold extrapolated as 16× the measured value. **Confirmed.** |
| Monkey-patch instead of subclass | Non-invasive; `models.py` stays pristine; patch is reversible and portable across VM clusters. **Confirmed.** |
| `Graph_build_time_ms` as column 7 | `radius_graph` at N=10⁵ can stall or OOM before `forward()` ever runs; must be independently observable in a separate outer try block. **Confirmed.** |
| Two-level try/except per cell | Outer block catches `__init__`/graph-build failure; inner block catches forward/backward failure; both set `OOM_Triggered=True`. |
| `reset_peak_memory_stats()` per cell | Prevents peak from previous cell contaminating current measurement. |
| `gc.collect()` + `empty_cache()` between cells | Avoids fragmentation false positives on long sweeps. |
| No `pandas` hard dependency | VM may not have it; `csv` module is always available. |
| `torch.cuda.OutOfMemoryError` + `RuntimeError` catch | Older PyTorch versions raise `RuntimeError` not the dedicated subclass. |
