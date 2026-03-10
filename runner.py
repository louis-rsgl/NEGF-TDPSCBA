from __future__ import annotations

import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from backend.observables import current_alpha
from backend.system_classes import LeadParams, System
from backend.units import current_to_uA, time_to_ps


# =============================================================================
# Global configuration
# =============================================================================

GAMMA: float = 0.01  # eV
VERBOSE: bool = True
USE_FAKE_SOLVER: bool = False

ALPHA_DEFAULT: str = "L"
T_MAX: float = 2.0          # dimensionless, in units of ħ/Gamma
N_T: int = 20_000

W_GRID = np.array([1.0, 2.5, 5.0, 10.0, 20.0], dtype=float)
GQ_GRID = np.array([0.0, 0.02, 0.1, 0.5], dtype=float)

PARALLEL: bool = True
MAX_WORKERS: int = 20

LOG_ROOT = Path("")


# =============================================================================
# Small logging helpers
# =============================================================================

class TimestampedWriter:
    """
    Wrap a text stream and prepend a timestamp to each completed line.
    This keeps existing print() output but makes logs easier to inspect.
    """
    def __init__(self, stream):
        self.stream = stream
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.stream.write(f"{line}\n")

        self.stream.flush()
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self.stream.write(f"{self._buffer}\n")
            self._buffer = ""
        self.stream.flush()


def make_run_log_dir(root: Path = LOG_ROOT) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = root / f"run_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=False)
    return log_dir


def worker_log_path(log_dir: Path, i: int, j: int, W: float, g_q: float) -> Path:
    return log_dir / f"worker_i{i}_j{j}_W{W:.3f}_gq{g_q:.3f}.log"


# =============================================================================
# System construction
# =============================================================================

def make_sys(W: float, g_q: float) -> System:
    return System(
        ETA=1e-3,
        DELTA=5.0,
        leads={
            "L": LeadParams(
                Gamma0=0.5,
                Delta=10.0,
                beta=0.1,
                mu=0.0,
            ),
            "R": LeadParams(
                Gamma0=0.5,
                Delta=0.0,
                beta=0.1,
                mu=0.0,
            ),
        },
        W=W,
        g_q=g_q,
        w_q=0.2,
        e_0=0.0,
        beta_ph=20.0,
        mu_ph=0.0,
        beta_fd=0.1,
        mu_fd=0.0,
        e_min=-10.0,
        e_max=10.0,
        omega_min=-10.0,
        omega_max=10.0,
        scba_max_iter=2000,
        scba_tol_abs=1e-4,
        scba_tol_rel=1e-3,
        scba_mixing=0.01,
        scba_min_iter=10,
        verbose=VERBOSE,
    )


# =============================================================================
# Fake backend for quick UI testing
# =============================================================================

def fake_current_alpha(
    sys: System,
    alpha: str,
    t_max: float,
    n_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, t_max, n_t, dtype=float)

    W = sys.W
    gq = sys.g_q

    omega = 2.0 + 6.0 * gq
    decay = np.exp(-0.15 * W * t)

    current = decay * (
        np.sin(omega * t)
        + 0.35 * np.sin(2.0 * omega * t)
        + 1j * 0.7 * np.cos(omega * t)
    )

    if alpha == "R":
        current = -0.8 * current

    return t, current


# =============================================================================
# Current computation
# =============================================================================

def compute_current(
    W: float,
    g_q: float,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
) -> tuple[np.ndarray, np.ndarray]:
    sys = make_sys(W=W, g_q=g_q)

    if USE_FAKE_SOLVER:
        if VERBOSE:
            print()
            print("=" * 82)
            print(f"Fake current evaluation | alpha={alpha} | W={W:.3f} | g_q={g_q:.3f}")
            print("=" * 82)

        t_dimless, I_dimless = fake_current_alpha(sys=sys, alpha=alpha, t_max=t_max, n_t=n_t)
    else:
        sys.launch()
        if VERBOSE:
            sys.reporter().print_unit_system(GAMMA)

        t_dimless, I_dimless = current_alpha(
            sys=sys,
            alpha=alpha,
            t_max=t_max,
            n_t=n_t,
        )

    t_ps = time_to_ps(t_dimless, GAMMA)
    I_uA = current_to_uA(I_dimless, GAMMA)

    return t_ps, I_uA


# =============================================================================
# Parallel worker
# =============================================================================

def _compute_current_job(
    i: int,
    j: int,
    W: float,
    g_q: float,
    alpha: str,
    t_max: float,
    n_t: int,
    log_dir_str: str,
) -> tuple[int, int, np.ndarray, np.ndarray, str]:
    log_dir = Path(log_dir_str)
    log_path = worker_log_path(log_dir, i, j, W, g_q)

    with open(log_path, "w", encoding="utf-8", buffering=1) as raw_fh:
        writer = TimestampedWriter(raw_fh)

        with redirect_stdout(writer), redirect_stderr(writer):
            pid = os.getpid()
            print("#" * 82)
            print("Worker started")
            print(f"pid={pid}")
            print(f"job indices: i={i}, j={j}")
            print(f"W={W:.6f}")
            print(f"g_q={g_q:.6f}")
            print(f"alpha={alpha}")
            print(f"t_max={t_max}")
            print(f"n_t={n_t}")
            print("#" * 82)

            try:
                t, current = compute_current(
                    W=W,
                    g_q=g_q,
                    alpha=alpha,
                    t_max=t_max,
                    n_t=n_t,
                )

                print("Worker finished successfully")
                print(f"max|J| = {np.max(np.abs(current)):.6e} µA")
                print(f"log_path = {log_path}")

            except Exception:
                print("Worker failed with exception:")
                import traceback
                traceback.print_exc()
                raise

    return i, j, t, current, str(log_path)


# =============================================================================
# Precomputation over parameter grid
# =============================================================================

def precompute_currents_parallel(
    W_grid: np.ndarray,
    gq_grid: np.ndarray,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    t_ref: np.ndarray | None = None
    J_grid = np.empty((len(W_grid), len(gq_grid), n_t), dtype=np.complex128)

    log_dir = make_run_log_dir()

    jobs: list[tuple[int, int, float, float, str, float, int, str]] = [
        (i, j, float(W), float(gq), alpha, t_max, n_t, str(log_dir))
        for i, W in enumerate(W_grid)
        for j, gq in enumerate(gq_grid)
    ]

    total = len(jobs)
    count = 0

    print()
    print("#" * 82)
    print("Beginning PARALLEL current precomputation over parameter grid")
    print("#" * 82)
    print(f"Gamma = {GAMMA:.6e} eV")
    print(f"alpha = {alpha}")
    print(f"t_max (dimensionless) = {t_max}")
    print(f"n_t = {n_t}")
    print(f"number of W values = {len(W_grid)}")
    print(f"number of g_q values = {len(gq_grid)}")
    print(f"total jobs = {total}")
    print(f"max_workers = {max_workers}")
    print(f"log_dir = {log_dir}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_compute_current_job, *job) for job in jobs]

        for future in as_completed(futures):
            i, j, t, current, log_path = future.result()

            if t_ref is None:
                t_ref = t
            elif not np.allclose(t, t_ref):
                raise ValueError("Inconsistent time grids encountered during precomputation.")

            J_grid[i, j, :] = current
            count += 1

            print(
                f"Stored current {count}/{total} | "
                f"(i={i}, j={j}) | "
                f"max|J| = {np.max(np.abs(current)):.6e} µA | "
                f"log={log_path}",
                flush=True,
            )

    if t_ref is None:
        raise RuntimeError("No currents were computed.")

    print()
    print("#" * 82)
    print("Finished parallel current precomputation")
    print("#" * 82)

    return t_ref, J_grid


def precompute_currents(
    W_grid: np.ndarray,
    gq_grid: np.ndarray,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
    parallel: bool = PARALLEL,
    max_workers: int | None = MAX_WORKERS,
) -> tuple[np.ndarray, np.ndarray]:
    if parallel:
        return precompute_currents_parallel(
            W_grid=W_grid,
            gq_grid=gq_grid,
            alpha=alpha,
            t_max=t_max,
            n_t=n_t,
            max_workers=max_workers,
        )

    raise NotImplementedError("Serial path omitted here for brevity.")


# =============================================================================
# Plot helpers
# =============================================================================

def nearest_index(values: np.ndarray, x: float) -> int:
    return int(np.argmin(np.abs(values - x)))


def make_title(alpha: str, W: float, g_q: float) -> str:
    return rf"Transient current $J_{{{alpha}}}(t)$ | $W={W:.3f}\Gamma$, $g_q={g_q:.3f}\Gamma$"


def make_ylabel(alpha: str) -> str:
    return rf"$J_{{{alpha}}}(t)$ ($\mu$A)"


# =============================================================================
# Main program
# =============================================================================

def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.size"] = 18

    alpha0 = ALPHA_DEFAULT

    t_ps, J_grid_uA = precompute_currents(
        W_grid=W_GRID,
        gq_grid=GQ_GRID,
        alpha=alpha0,
        t_max=T_MAX,
        n_t=N_T,
        parallel=PARALLEL,
        max_workers=MAX_WORKERS,
    )

    W0 = 20.0
    gq0 = 0.1

    i0 = nearest_index(W_GRID, W0)
    j0 = nearest_index(GQ_GRID, gq0)
    I0 = J_grid_uA[i0, j0, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.14, bottom=0.28)

    (line_re,) = ax.plot(
        t_ps,
        np.real(I0),
        linewidth=2.0,
        label=rf"$\mathrm{{Re}}\,J_{{{alpha0}}}(t)$",
    )
    (line_im,) = ax.plot(
        t_ps,
        np.imag(I0),
        linewidth=2.0,
        linestyle="--",
        label=rf"$\mathrm{{Im}}\,J_{{{alpha0}}}(t)$",
    )

    ax.set_xlabel(r"$t$ (ps)")
    ax.set_ylabel(make_ylabel(alpha0))
    ax.set_title(make_title(alpha0, W_GRID[i0], GQ_GRID[j0]))
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax_W = plt.axes([0.14, 0.16, 0.70, 0.03])
    ax_gq = plt.axes([0.14, 0.09, 0.70, 0.03])

    s_W = Slider(
        ax=ax_W,
        label=r"$W/\Gamma$",
        valmin=float(W_GRID.min()),
        valmax=float(W_GRID.max()),
        valinit=float(W0),
    )

    s_gq = Slider(
        ax=ax_gq,
        label=r"$g_q/\Gamma$",
        valmin=float(GQ_GRID.min()),
        valmax=float(GQ_GRID.max()),
        valinit=float(gq0),
    )

    def update(_val=None) -> None:
        i = nearest_index(W_GRID, s_W.val)
        j = nearest_index(GQ_GRID, s_gq.val)

        current = J_grid_uA[i, j, :]

        line_re.set_ydata(np.real(current))
        line_im.set_ydata(np.imag(current))

        ax.relim()
        ax.autoscale_view()
        ax.set_title(make_title(alpha0, W_GRID[i], GQ_GRID[j]))
        fig.canvas.draw_idle()

    s_W.on_changed(update)
    s_gq.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()