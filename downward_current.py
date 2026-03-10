from __future__ import annotations

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
# Precomputation over parameter grid
# =============================================================================

def precompute_currents(
    W_grid: np.ndarray,
    gq_grid: np.ndarray,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
) -> tuple[np.ndarray, np.ndarray]:
    t_ref: np.ndarray | None = None
    J_grid = np.empty((len(W_grid), len(gq_grid), n_t), dtype=np.complex128)

    total = len(W_grid) * len(gq_grid)
    count = 0

    if VERBOSE:
        print()
        print("#" * 82)
        print("Beginning current precomputation over parameter grid")
        print("#" * 82)
        print(f"Gamma = {GAMMA:.6e} eV")
        print(f"alpha = {alpha}")
        print(f"t_max (dimensionless) = {t_max}")
        print(f"n_t = {n_t}")
        print(f"number of W values = {len(W_grid)}")
        print(f"number of g_q values = {len(gq_grid)}")
        print(f"total jobs = {total}")

    for i, W in enumerate(W_grid):
        for j, gq in enumerate(gq_grid):
            if VERBOSE:
                print()
                print("-" * 82)
                print(
                    f"Job {count + 1}/{total} | "
                    f"W = {W:.6f} Gamma | g_q = {gq:.6f} Gamma | alpha = {alpha}"
                )
                print("-" * 82)

            t, current = compute_current(
                W=W,
                g_q=gq,
                alpha=alpha,
                t_max=t_max,
                n_t=n_t,
            )

            if t_ref is None:
                t_ref = t
            elif not np.allclose(t, t_ref):
                raise ValueError("Inconsistent time grids encountered during precomputation.")

            J_grid[i, j, :] = current
            count += 1

            if VERBOSE:
                print(
                    f"Stored current {count}/{total} | "
                    f"max|J| = {np.max(np.abs(current)):.6e} µA",
                    flush=True,
                )

    if t_ref is None:
        raise RuntimeError("No currents were computed.")

    if VERBOSE:
        print()
        print("#" * 82)
        print("Finished current precomputation")
        print("#" * 82)

    return t_ref, J_grid


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