from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from backend.observables import current_alpha
from backend.system_classes import LeadParams, System


GAMMA: float = 1.0
USE_FAKE_SOLVER: bool = False

ALPHA_DEFAULT: str = "L"
T_MAX: float = 2.0
N_T: int = 2000

W_GRID = np.array([1.0, 2.5, 5.0, 10.0, 20.0], dtype=float) * GAMMA
GQ_GRID = np.linspace(0.0, 1.0, 41, dtype=float)


def make_sys(W: float, g_q: float) -> System:
    return System(
        ETA=1e-10,
        DELTA=5.0 * GAMMA,
        leads={
            "L": LeadParams(
                Gamma0=0.5 * GAMMA,
                Delta=10.0 * GAMMA,
                beta=0.1 * GAMMA,
                mu=0.0,
            ),
            "R": LeadParams(
                Gamma0=0.5 * GAMMA,
                Delta=0.0,
                beta=0.1 * GAMMA,
                mu=0.0,
            ),
        },
        W=W,
        g_q=g_q,
        w_q=0.2,
        e_0=0.0,
        beta_ph=20.0,
        mu_ph=0.0,
        beta_fd=0.1 * GAMMA,
        mu_fd=0.0,
        e_min=-10.0,
        e_max=10.0,
        omega_min=-10.0,
        omega_max=10.0,
        scba_max_iter=2000,
        scba_tol_abs=1e-8,
        scba_tol_rel=1e-6,
        scba_mixing=0.01,
        scba_min_iter=5,
        scba_verbose=True,
    )


def fake_current_alpha(sys: System, alpha: str, t_max: float, n_t: int) -> tuple[np.ndarray, np.ndarray]:
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


def compute_current(
    W: float,
    g_q: float,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
) -> tuple[np.ndarray, np.ndarray]:
    sys = make_sys(W=W, g_q=g_q)

    if USE_FAKE_SOLVER:
        return fake_current_alpha(sys=sys, alpha=alpha, t_max=t_max, n_t=n_t)

    return current_alpha(
        sys=sys,
        alpha=alpha,
        t_max=t_max,
        n_t=n_t,
    )


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

    for i, W in enumerate(W_grid):
        for j, gq in enumerate(gq_grid):
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
            print(f"Precomputed {count}/{total}: W={W:.3f}, g_q={gq:.3f}")

    if t_ref is None:
        raise RuntimeError("No currents were computed.")

    return t_ref, J_grid


def nearest_index(values: np.ndarray, x: float) -> int:
    return int(np.argmin(np.abs(values - x)))


def make_title(alpha: str, W: float, g_q: float) -> str:
    return rf"Transient current $J_{{{alpha}}}(t)$ | $W={W:.3f}$, $g_q={g_q:.3f}$"


def make_ylabel(alpha: str) -> str:
    return rf"$J_{{{alpha}}}(t)\;(e\Gamma)$"


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.size"] = 18
    # plt.rcParams["text.usetex"] = True

    alpha0 = ALPHA_DEFAULT

    t, J_grid = precompute_currents(
        W_grid=W_GRID,
        gq_grid=GQ_GRID,
        alpha=alpha0,
        t_max=T_MAX,
        n_t=N_T,
    )

    W0 = 20.0 * GAMMA
    gq0 = 0.1

    i0 = nearest_index(W_GRID, W0)
    j0 = nearest_index(GQ_GRID, gq0)
    I0 = J_grid[i0, j0, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.14, bottom=0.28)

    (line_re,) = ax.plot(
        t,
        np.real(I0),
        linewidth=2.0,
        label=rf"$\mathrm{{Re}}\,J_{{{alpha0}}}(t)$",
    )
    (line_im,) = ax.plot(
        t,
        np.imag(I0),
        linewidth=2.0,
        linestyle="--",
        label=rf"$\mathrm{{Im}}\,J_{{{alpha0}}}(t)$",
    )

    ax.set_xlabel(r"$t\;(\Gamma^{-1})$")
    ax.set_ylabel(make_ylabel(alpha0))
    ax.set_title(make_title(alpha0, W_GRID[i0], GQ_GRID[j0]))
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax_W = plt.axes([0.14, 0.16, 0.70, 0.03])
    ax_gq = plt.axes([0.14, 0.09, 0.70, 0.03])

    s_W = Slider(
        ax=ax_W,
        label=r"$W\;(\Gamma)$",
        valmin=float(W_GRID.min()),
        valmax=float(W_GRID.max()),
        valinit=float(W0),
    )

    s_gq = Slider(
        ax=ax_gq,
        label=r"$g_q$",
        valmin=float(GQ_GRID.min()),
        valmax=float(GQ_GRID.max()),
        valinit=float(gq0),
    )

    def update(_val=None) -> None:
        i = nearest_index(W_GRID, s_W.val)
        j = nearest_index(GQ_GRID, s_gq.val)

        current = J_grid[i, j, :]

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