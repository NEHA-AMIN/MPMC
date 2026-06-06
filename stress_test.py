"""
stress_test.py — MPMC Phase 1 GPU Stress-Test & Memory Profiling
=================================================================
Automated sweep over a nsamples × dim grid to profile peak VRAM usage,
per-segment timing, and OOM boundaries in MPMC_net.

# ============================================================================
# INSTALL (run once before executing this script)
# ============================================================================
# Option A — use the bundled installer (recommended):
#   chmod +x install.sh && ./install.sh
#
# Option B — manual, for a known PyTorch + CUDA version (example: torch 2.1, cu121):
#   pip install torch-scatter torch-sparse torch-cluster \\
#       --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
#   pip install torch-geometric matplotlib numpy
#
# Option C — detect version dynamically in your shell:
#   TORCH=$(python -c "import torch,re; print(re.split(r'[+]',torch.__version__)[0])")
#   CUDA=$(python  -c "import torch; v=torch.version.cuda; print('cpu' if not v else 'cu'+v.replace('.','')[:3])")
#   pip install torch-scatter torch-sparse torch-cluster \\
#       --find-links "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
#   pip install torch-geometric matplotlib numpy
# ============================================================================

Usage
-----
    # Quick validation (tiny grid, ~60 s)
    python stress_test.py --smoke

    # Full GPU stress sweep
    python stress_test.py

    # Custom output directory
    python stress_test.py --outdir /data/profiling_runs/

    # Skip Chrome profiler traces (faster sweep)
    python stress_test.py --no-traces

Output
------
    profiling_outputs/
    ├── stress_test_results_<timestamp>.csv    7-column results table
    ├── stress_test_results_<timestamp>.png    dual-axis VRAM / time chart
    └── trace_N<nsamples>_D<dim>.json          Chrome trace per successful cell
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Path setup — allow running from parent or from MPMC/ directly
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Dependency check with helpful error message
# ---------------------------------------------------------------------------
def _check_imports() -> None:
    missing = []
    for pkg, import_name in [
        ("torch-cluster",  "torch_cluster"),
        ("torch-geometric", "torch_geometric"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n[ERROR] Missing dependencies:", ", ".join(missing))
        print("  Run:  chmod +x install.sh && ./install.sh")
        print("  or:   pip install torch-cluster torch-geometric "
              "--find-links https://data.pyg.org/whl/torch-<ver>+<cuda>.html\n")
        sys.exit(1)

_check_imports()

from models import MPMC_net                                    # noqa: E402
from profile_hooks import patch_model, make_checkpoint, run_profiler_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Sweep grid constants
# ---------------------------------------------------------------------------
NSAMPLES_GRID: list[int] = [1000, 5000, 10000, 25000, 50000, 100000]
DIM_GRID:      list[int] = [3, 6, 20, 50]
RADIUS:  float = 0.35
NBATCH:  int   = 1       # nbatch=1 isolates pure O(N²·d) spatial scalability
NHID:    int   = 128
NLAYERS: int   = 3

# Smoke-test overrides
SMOKE_NSAMPLES_GRID: list[int] = [100, 200]
SMOKE_DIM_GRID:      list[int] = [3, 6]

# Column names (order preserved in CSV)
CSV_FIELDNAMES = [
    "nsamples",
    "dim",
    "Graph_build_time_ms",
    "GNN_forward_time_ms",
    "Loss_time_ms",
    "Peak_VRAM_Allocated_MB",
    "OOM_Triggered",
]


# ---------------------------------------------------------------------------
# OOM detection helper
# ---------------------------------------------------------------------------
def _is_oom(exc: Exception) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ---------------------------------------------------------------------------
# Core sweep function
# ---------------------------------------------------------------------------
def run_stress_test(
    nsamples_grid: list[int],
    dim_grid: list[int],
    outdir: Path,
    *,
    emit_traces: bool = True,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """
    Iterate over the full nsamples × dim grid.  For each cell:

      Outer try (Segment C — __init__ / radius_graph):
        Instantiate MPMC_net and record Graph_build_time_ms.

      Inner try (Segments A+B — forward + backward):
        Run patched forward, collect per-segment timing and peak VRAM,
        then run loss.backward().

      Cleanup between cells: del model, empty_cache, gc.collect.

    Returns a list of result dicts with the seven CSV columns.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_cells = len(nsamples_grid) * len(dim_grid)
    rows: list[dict[str, Any]] = []

    print(f"\n{'='*66}")
    print(f"  MPMC Phase 1 Stress Test  |  device={device}")
    if smoke:
        print("  MODE: --smoke (small grid for pipeline validation)")
    print(f"  Grid: {len(nsamples_grid)} nsamples × {len(dim_grid)} dims = {total_cells} cells")
    print(f"  radius={RADIUS}  nbatch={NBATCH}  nhid={NHID}  nlayers={NLAYERS}")
    print(f"{'='*66}\n")

    cell_idx = 0
    for n in nsamples_grid:
        for d in dim_grid:
            cell_idx += 1
            row: dict[str, Any] = {
                "nsamples": n,
                "dim": d,
                "Graph_build_time_ms":   float("nan"),
                "GNN_forward_time_ms":   float("nan"),
                "Loss_time_ms":          float("nan"),
                "Peak_VRAM_Allocated_MB": float("nan"),
                "OOM_Triggered":         False,
            }

            prefix = f"[{cell_idx:>{len(str(total_cells))}}/{total_cells}]  N={n:<7} d={d:<3}"
            print(f"{prefix}  ", end="", flush=True)

            # ------------------------------------------------------------------
            # Reset CUDA peak stats for this cell
            # ------------------------------------------------------------------
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            gc.collect()

            model: MPMC_net | None = None

            # ==================================================================
            # Outer try — Segment C: MPMC_net.__init__ + radius_graph
            # ==================================================================
            try:
                t_init_start = time.perf_counter()
                model = MPMC_net(
                    dim=d,
                    nhid=NHID,
                    nlayers=NLAYERS,
                    nsamples=n,
                    nbatch=NBATCH,
                    radius=RADIUS,
                    loss_fn="L2",
                    dim_emphasize=[1],
                    n_projections=15,
                ).to(device)
                t_init_end = time.perf_counter()
                row["Graph_build_time_ms"] = (t_init_end - t_init_start) * 1000.0

            except Exception as exc:
                row["OOM_Triggered"] = True
                if _is_oom(exc):
                    print(f"OOM @ __init__ (radius_graph)")
                else:
                    print(f"ERROR @ __init__: {type(exc).__name__}: {exc}")
                rows.append(row)
                _cleanup(model)
                continue

            # ==================================================================
            # Inner try — Segments A+B: patched forward + backward
            # ==================================================================
            try:
                patch_model(model)
                model.train()

                # Warm-up: one pass without gradient tape to heat CUDA kernels
                with torch.no_grad():
                    model()

                # Reset peak stats again after warm-up
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                # ----- Timed forward pass (gradient tracking ON) ---------------
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                optimizer.zero_grad()

                loss, _X = model()          # patched forward writes _profile_results
                loss.backward()
                optimizer.step()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                pr = model._profile_results
                row["GNN_forward_time_ms"]    = pr["gnn_elapsed_ms"]
                row["Loss_time_ms"]           = pr["loss_elapsed_ms"]
                row["Peak_VRAM_Allocated_MB"] = pr["peak_vram_total_mb"]

                # ----- Optional Chrome trace (one per cell) ---------------------
                if emit_traces:
                    trace_path = outdir / f"trace_N{n}_D{d}.json"
                    try:
                        run_profiler_trace(model, trace_path, warmup_steps=1, active_steps=1)
                    except Exception as te:
                        warnings.warn(f"Trace export failed for N={n} d={d}: {te}")

                status = (
                    f"graph={row['Graph_build_time_ms']:.1f}ms  "
                    f"gnn={row['GNN_forward_time_ms']:.1f}ms  "
                    f"loss={row['Loss_time_ms']:.1f}ms  "
                    f"VRAM={row['Peak_VRAM_Allocated_MB']:.1f}MB"
                )
                print(status)

            except Exception as exc:
                row["OOM_Triggered"] = True
                if _is_oom(exc):
                    # Still record the graph-build time even when forward OOMs
                    print(f"OOM @ forward  (graph_build={row['Graph_build_time_ms']:.1f}ms)")
                else:
                    print(f"ERROR @ forward: {type(exc).__name__}: {exc}")

            finally:
                _cleanup(model)

            rows.append(row)

    print(f"\n{'='*66}")
    print(f"  Sweep complete — {len(rows)} cells recorded")
    oom_count = sum(1 for r in rows if r["OOM_Triggered"])
    print(f"  OOM cells: {oom_count} / {len(rows)}")
    print(f"{'='*66}\n")
    return rows


def _cleanup(model: Any) -> None:
    """Release model memory and flush CUDA allocator caches."""
    try:
        if model is not None:
            del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def save_results(rows: list[dict[str, Any]], outdir: Path) -> Path:
    """
    Write results to a timestamped CSV file in outdir.

    Uses csv.DictWriter — no pandas dependency.
    NaN values are written as the string 'NaN' for readability.
    Boolean OOM_Triggered is written as 'True' / 'False'.

    Returns the path of the created CSV file.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"stress_test_results_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for k, v in row.items():
                if isinstance(v, float) and math.isnan(v):
                    clean[k] = "NaN"
                elif isinstance(v, bool):
                    clean[k] = str(v)
                elif isinstance(v, float):
                    clean[k] = f"{v:.4f}"
                else:
                    clean[k] = v
            writer.writerow(clean)

    print(f"[save_results] CSV written → {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_results(csv_path: Path) -> Path:
    """
    Generate a dual-axis line chart from the results CSV.

    Primary y-axis  (left,  solid lines):    Peak VRAM Allocated (MB) vs nsamples
    Secondary y-axis (right, dashed lines):  Total forward time (ms) vs nsamples
    Both axes share a log-scale x-axis.
    OOM cells are overlaid as red × markers on the primary axis.

    Saves a PNG alongside the CSV and returns its path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend — safe on headless VMs
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[plot_results] matplotlib not available — skipping plot generation.")
        print("  Install with:  pip install matplotlib")
        return csv_path  # return csv path as fallback

    # -- Parse CSV (no pandas required) ----------------------------------------
    raw_rows: list[dict[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_rows.append(dict(r))

    def _float(s: str) -> float:
        try:
            return float(s)
        except (ValueError, TypeError):
            return float("nan")

    def _bool(s: str) -> bool:
        return s.strip().lower() == "true"

    parsed: list[dict[str, Any]] = [
        {
            "nsamples":              int(r["nsamples"]),
            "dim":                   int(r["dim"]),
            "Graph_build_time_ms":   _float(r["Graph_build_time_ms"]),
            "GNN_forward_time_ms":   _float(r["GNN_forward_time_ms"]),
            "Loss_time_ms":          _float(r["Loss_time_ms"]),
            "Peak_VRAM_Allocated_MB": _float(r["Peak_VRAM_Allocated_MB"]),
            "OOM_Triggered":         _bool(r["OOM_Triggered"]),
        }
        for r in raw_rows
    ]

    dims = sorted({r["dim"] for r in parsed})

    # -- Color palette ----------------------------------------------------------
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Pad in case dim_grid has more entries than default cycle
    while len(COLORS) < len(dims):
        COLORS += COLORS

    # -- Figure setup -----------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    ax1.set_xlabel("Number of Samples (N)", fontsize=13)
    ax1.set_ylabel("Peak VRAM Allocated (MB)", fontsize=13, color="tab:blue")
    ax2.set_ylabel("Total Forward Time (ms)\n[GNN + Loss]", fontsize=13, color="tab:red")
    ax1.set_xscale("log")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Force integer x-axis tick labels at the actual grid points
    all_nsamples = sorted({r["nsamples"] for r in parsed})
    ax1.set_xticks(all_nsamples)
    ax1.get_xaxis().set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    vram_lines = []
    time_lines = []
    oom_x, oom_y = [], []

    for idx, d in enumerate(dims):
        color = COLORS[idx]
        subset = sorted(
            [r for r in parsed if r["dim"] == d],
            key=lambda r: r["nsamples"],
        )

        xs  = [r["nsamples"]              for r in subset]
        vms = [r["Peak_VRAM_Allocated_MB"] for r in subset]
        tms = [
            (r["GNN_forward_time_ms"] or 0) + (r["Loss_time_ms"] or 0)
            for r in subset
        ]

        # Replace NaN with None so matplotlib skips OOM gaps cleanly
        vms_plot = [v if not math.isnan(v) else None for v in vms]
        tms_plot = [v if not math.isnan(v) else None for v in tms]

        # Primary: VRAM solid line
        ln1, = ax1.plot(
            xs, vms_plot,
            color=color, linewidth=2, marker="o", markersize=5,
            label=f"d={d} — VRAM",
        )
        vram_lines.append(ln1)

        # Secondary: time dashed line
        ln2, = ax2.plot(
            xs, tms_plot,
            color=color, linewidth=1.5, linestyle="--", marker="s", markersize=4,
            alpha=0.75,
            label=f"d={d} — Time",
        )
        time_lines.append(ln2)

        # Collect OOM positions (plot on primary axis at y=0 or at the last
        # known VRAM value to make them visible in context)
        for r in subset:
            if r["OOM_Triggered"]:
                # Use the last non-NaN VRAM value for this dim as the y position
                last_vram = next(
                    (v for v in reversed(vms) if not math.isnan(v)), 0
                )
                oom_x.append(r["nsamples"])
                oom_y.append(last_vram)

    # OOM markers — red × overlay on primary axis
    if oom_x:
        ax1.scatter(
            oom_x, oom_y,
            marker="x", color="red", s=120, linewidths=2.5, zorder=5,
            label="OOM (any segment)",
        )

    # -- Legend (combine both axes) --------------------------------------------
    all_lines = vram_lines + time_lines
    if oom_x:
        # Add a proxy artist for the OOM scatter
        import matplotlib.lines as mlines
        oom_proxy = mlines.Line2D(
            [], [], color="red", marker="x", linestyle="None",
            markersize=10, markeredgewidth=2.5, label="OOM (any segment)",
        )
        all_lines.append(oom_proxy)

    labels = [ln.get_label() for ln in all_lines]
    ax1.legend(all_lines, labels, loc="upper left", fontsize=8, framealpha=0.85)

    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    title = (
        "MPMC Phase 1 — Peak VRAM & Forward Time vs. N\n"
        f"radius={RADIUS}  nbatch={NBATCH}  nhid={NHID}  nlayers={NLAYERS}"
    )
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()

    png_path = csv_path.with_suffix(".png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] Chart written  → {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Stdout summary table
# ---------------------------------------------------------------------------
def print_summary_table(rows: list[dict[str, Any]]) -> None:
    """Print a formatted table of all results to stdout."""
    hdr = (
        f"{'N':>8}  {'d':>4}  "
        f"{'GraphBuild(ms)':>14}  {'GNN(ms)':>8}  {'Loss(ms)':>9}  "
        f"{'VRAM(MB)':>9}  {'OOM':>5}"
    )
    sep = "-" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in rows:
        def _fmt(v: Any, w: int = 8) -> str:
            if isinstance(v, float) and math.isnan(v):
                return "NaN".rjust(w)
            if isinstance(v, float):
                return f"{v:.1f}".rjust(w)
            return str(v).rjust(w)

        oom_str = "YES" if r["OOM_Triggered"] else "no"
        print(
            f"{_fmt(r['nsamples'], 8)}  "
            f"{_fmt(r['dim'], 4)}  "
            f"{_fmt(r['Graph_build_time_ms'], 14)}  "
            f"{_fmt(r['GNN_forward_time_ms'], 8)}  "
            f"{_fmt(r['Loss_time_ms'], 9)}  "
            f"{_fmt(r['Peak_VRAM_Allocated_MB'], 9)}  "
            f"{oom_str:>5}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MPMC Phase 1 GPU Stress Test & Memory Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help=(
            "Run a tiny validation grid [nsamples=[100,200], dim=[3,6]] "
            "to verify the pipeline before committing to the full sweep (~60 s)."
        ),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("profiling_outputs"),
        help="Directory for CSV, PNG, and trace files (default: ./profiling_outputs/).",
    )
    p.add_argument(
        "--no-traces",
        action="store_true",
        default=False,
        help="Disable per-cell Chrome trace export (faster sweep, less disk I/O).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # CUDA availability warning
    if not torch.cuda.is_available():
        print(
            "\n[WARNING] CUDA is not available on this system.\n"
            "  All profiling will run on CPU — VRAM columns will be 0.0 MB.\n"
            "  For meaningful GPU memory telemetry, run on a CUDA-capable machine.\n"
        )
    else:
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / 1024 ** 3
        print(
            f"\n[GPU] {props.name}  |  "
            f"VRAM: {total_vram_gb:.1f} GB  |  "
            f"CUDA {torch.version.cuda}"
        )

    # Choose grid
    if args.smoke:
        nsamples_grid = SMOKE_NSAMPLES_GRID
        dim_grid      = SMOKE_DIM_GRID
    else:
        nsamples_grid = NSAMPLES_GRID
        dim_grid      = DIM_GRID

    # Run sweep
    rows = run_stress_test(
        nsamples_grid=nsamples_grid,
        dim_grid=dim_grid,
        outdir=args.outdir,
        emit_traces=(not args.no_traces),
        smoke=args.smoke,
    )

    # Print summary table
    print_summary_table(rows)

    # Save CSV
    csv_path = save_results(rows, args.outdir)

    # Generate chart
    plot_results(csv_path)

    print(f"\nAll outputs written to:  {args.outdir.resolve()}\n")


if __name__ == "__main__":
    main()
