from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy as sp

from backend.distribution import bose_einstein, fermi_dirac
from backend.green_function import GR_eq


@dataclass
class SolverResult:
    converged: bool
    n_iter: int
    err_GR: float
    err_Gless: float
    history_GR: list[float] = field(default_factory=list)
    history_Gless: list[float] = field(default_factory=list)


class Solver:
    """
    Self-consistent solver for the coupled nonequilibrium system

        G^R(w) = [w - e0 - DELTA - V^R(w)]^{-1}

        G^<(w) = G^R(w) Sigma^<(w) G^A(w)

    with

        Sigma^<(w)
          = i sum_beta f_beta(w) Gamma0_beta * W / (w^2 + W^2)
            + g_q^2 [ G^<(w-w_q) f_ph(w_q)
                    + G^<(w+w_q) (f_ph(w_q)+1) ]

    and

        V^R(w)
          = [ integral_term
              - 2 (f_ph(w_q)+1) G^R(w-w_q) g_q^2
              - 2 f_ph(w_q)     G^R(w+w_q) g_q^2
            ]
            * [ 1
                + DELTA G^R(w)
                + (1/2) sum_beta Gamma0_beta W Delta_beta
                    / ((w-Delta_beta+iW)(w+iW)) * G^R(w)
              ]
            + sum_beta (1/2) Gamma0_beta W / (w-Delta_beta+iW)

    where

        integral_term
          = ∫ dΩ/(iπ) g_q^2 G^<(w-Ω)
                [1/(Ω-w_q+iη) - 1/(Ω+w_q+iη)]
    """

    def __init__(
        self,
        sys,
        w_min: float,
        w_max: float,
        n_w: int,
    ):
        self.sys = sys
        self.w = np.linspace(w_min, w_max, n_w, dtype=float)
        self.dw = float(self.w[1] - self.w[0]) if n_w > 1 else 0.0

        # Solver controls are stored on the System object
        self.max_iter = sys.scba_max_iter
        self.tol_abs = sys.scba_tol_abs
        self.tol_rel = sys.scba_tol_rel
        self.mixing = sys.scba_mixing
        self.min_iter = sys.scba_min_iter
        self.verbose = sys.scba_verbose

        self.GR_values: Optional[np.ndarray] = None
        self.Gless_values: Optional[np.ndarray] = None

        self.history_GR: list[float] = []
        self.history_Gless: list[float] = []
        self.result: Optional[SolverResult] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initial guess:
          - G^R_0 from equilibrium retarded GF
          - G^<_0 from a simple Keldysh-like ansatz
        """
        GR0 = np.asarray(GR_eq(self.sys, self.w), dtype=np.complex128)

        f0 = np.asarray(
            fermi_dirac(self.w, self.sys.beta_fd, self.sys.mu_fd),
            dtype=np.complex128,
        )

        Gless0 = -2j * np.imag(GR0) * f0

        self.GR_values = GR0
        self.Gless_values = Gless0
        self.history_GR = []
        self.history_Gless = []
        self.result = None

    def _require_initialized(self) -> None:
        if self.GR_values is None or self.Gless_values is None:
            raise RuntimeError("Solver not initialized. Call initialize() or solve().")

    # ------------------------------------------------------------------
    # Public runtime evaluation
    # ------------------------------------------------------------------

    def GR(self, e):
        self._require_initialized()
        return self._interp_complex(self.GR_values, e)

    def Gless(self, e):
        self._require_initialized()
        return self._interp_complex(self.Gless_values, e)

    def GA(self, e):
        return np.conjugate(self.GR(e))

    # ------------------------------------------------------------------
    # Interpolation helper
    # ------------------------------------------------------------------

    def _interp_complex(self, y_grid: np.ndarray, x):
        """
        Interpolate in the real part of x only.
        Intended for shifted frequencies such as w ± w_q and w - Omega.
        """
        x_arr = np.asarray(x, dtype=np.complex128)
        xr = x_arr.real

        yr = np.interp(xr, self.w, np.real(y_grid))
        yi = np.interp(xr, self.w, np.imag(y_grid))
        return yr + 1j * yi

    # ------------------------------------------------------------------
    # Self-energies / kernels from current iterate
    # ------------------------------------------------------------------

    def compute_sigma_less_from_state(self) -> np.ndarray:
        """
        Sigma^<(w) =
            i sum_beta f_beta(w) Gamma0_beta * W / (w^2 + W^2)
            + g_q^2 [ G^<(w-w_q) f_ph
                    + G^<(w+w_q) (f_ph+1) ]
        """
        self._require_initialized()
        sys = self.sys

        omega = self.w
        fph = bose_einstein(sys.w_q, sys.beta_ph, sys.mu_ph)

        lead_term = np.zeros_like(omega, dtype=np.complex128)

        for beta in sys.lead_names:
            fbeta = np.asarray(
                fermi_dirac(omega, sys.beta_fc(beta), sys.mu_fc(beta)),
                dtype=np.complex128,
            )
            lead_term += (
                1j
                * fbeta
                * sys.Gamma0(beta)
                * sys.W
                / (omega**2 + sys.W**2)
            )

        phonon_term = sys.g_q**2 * (
            self.Gless(omega - sys.w_q) * fph
            + self.Gless(omega + sys.w_q) * (fph + 1.0)
        )

        return np.asarray(lead_term + phonon_term, dtype=np.complex128)

    def compute_VR_from_state(self) -> np.ndarray:
        """
        Compute V^R(w) on the full frequency grid.

        V^R(w) =
            [ integral_term(w)
            - 2 (f_ph+1) G^R(w-w_q) g_q^2
            - 2 f_ph     G^R(w+w_q) g_q^2
            ]
            * [ 1
                + DELTA G^R(w)
                + (1/2) sum_beta Gamma0_beta W Delta_beta
                    / ((w-Delta_beta+iW)(w+iW)) * G^R(w)
            ]
            + sum_beta (1/2) Gamma0_beta W / (w-Delta_beta+iW)
        """
        self._require_initialized()
        sys = self.sys

        omega = self.w
        fph = bose_einstein(sys.w_q, sys.beta_ph, sys.mu_ph)

        val, _ = sp.integrate.quad_vec(
            lambda Omega: (
                self.Gless(omega - Omega)
                * (
                    1.0 / (Omega - sys.w_q + 1j * sys.ETA)
                    - 1.0 / (Omega + sys.w_q + 1j * sys.ETA)
                )
            ),
            sys.omega_min,
            sys.omega_max,
        )
        term_int = sys.g_q**2 * val / (1j * np.pi)

        term_shift_minus = -2.0 * (fph + 1.0) * self.GR(omega - sys.w_q) * sys.g_q**2
        term_shift_plus = -2.0 * fph * self.GR(omega + sys.w_q) * sys.g_q**2

        bracket1 = term_int + term_shift_minus + term_shift_plus

        sum_factor = np.zeros_like(omega, dtype=np.complex128)
        for beta in sys.lead_names:
            gamma_beta = sys.Gamma0(beta)
            delta_beta = sys.Delta(beta)

            sum_factor += (
                0.5
                * gamma_beta
                * sys.W
                * delta_beta
                / ((omega - delta_beta + 1j * sys.W) * (omega + 1j * sys.W))
            )

        GRw = self.GR(omega)
        bracket2 = 1.0 + sys.DELTA * GRw + sum_factor * GRw

        add_term = np.zeros_like(omega, dtype=np.complex128)
        for beta in sys.lead_names:
            gamma_beta = sys.Gamma0(beta)
            delta_beta = sys.Delta(beta)

            add_term += 0.5 * gamma_beta * sys.W / (omega - delta_beta + 1j * sys.W)

        return np.asarray(bracket1 * bracket2 + add_term, dtype=np.complex128)

    # ------------------------------------------------------------------
    # Trial updates
    # ------------------------------------------------------------------

    def update_trials(self) -> tuple[np.ndarray, np.ndarray]:
        """
        One nonlinear map step:
            GR_trial    from current V^R[state]
            Gless_trial from current Sigma^<[state] and GR_trial
        """
        VR = self.compute_VR_from_state()

        GR_trial = 1.0 / (
            self.w - self.sys.e_0 - self.sys.DELTA - VR + 1j * self.sys.ETA
        )

        Sigma_less = self.compute_sigma_less_from_state()
        GA_trial = np.conjugate(GR_trial)
        Gless_trial = GR_trial * Sigma_less * GA_trial

        return (
            np.asarray(GR_trial, dtype=np.complex128),
            np.asarray(Gless_trial, dtype=np.complex128),
        )

    # ------------------------------------------------------------------
    # Convergence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def linear_mix(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * old + alpha * new

    @staticmethod
    def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    @staticmethod
    def rel_err(a: np.ndarray, b: np.ndarray, eps: float = 1e-14) -> float:
        denom = np.maximum(np.abs(b), eps)
        return float(np.max(np.abs(a - b) / denom))

    # ------------------------------------------------------------------
    # Solve loop
    # ------------------------------------------------------------------

    def solve(self) -> SolverResult:
        if self.GR_values is None or self.Gless_values is None:
            self.initialize()

        converged = False
        err_GR = np.inf
        err_Gless = np.inf

        for it in range(1, self.max_iter + 1):
            GR_trial, Gless_trial = self.update_trials()

            GR_next = self.linear_mix(self.GR_values, GR_trial, self.mixing)
            Gless_next = self.linear_mix(self.Gless_values, Gless_trial, self.mixing)

            err_GR_abs = self.max_abs_err(GR_next, self.GR_values)
            err_GR_rel = self.rel_err(GR_next, self.GR_values)

            err_Gless_abs = self.max_abs_err(Gless_next, self.Gless_values)
            err_Gless_rel = self.rel_err(Gless_next, self.Gless_values)

            err_GR = max(err_GR_abs, err_GR_rel)
            err_Gless = max(err_Gless_abs, err_Gless_rel)

            self.history_GR.append(err_GR)
            self.history_Gless.append(err_Gless)

            self.GR_values = GR_next
            self.Gless_values = Gless_next

            if self.verbose:
                print(
                    f"[noneq] iter={it:4d} "
                    f"err_GR={err_GR:.3e} "
                    f"err_Gless={err_Gless:.3e}"
                )

            gr_ok = (err_GR_abs < self.tol_abs) or (err_GR_rel < self.tol_rel)
            gl_ok = (err_Gless_abs < self.tol_abs) or (err_Gless_rel < self.tol_rel)

            if it >= self.min_iter and gr_ok and gl_ok:
                converged = True
                break

        self.result = SolverResult(
            converged=converged,
            n_iter=it,
            err_GR=err_GR,
            err_Gless=err_Gless,
            history_GR=self.history_GR.copy(),
            history_Gless=self.history_Gless.copy(),
        )
        return self.result