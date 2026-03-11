from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy as sp

from backend.distribution import bose_einstein, fermi_dirac
from backend.green_function import GR_eq

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


@dataclass
class SolverResult:
    converged: bool
    n_iter: int
    res_GR_abs: float
    res_GR_rel: float
    res_Gless_abs: float
    res_Gless_rel: float
    history_res_GR_abs: list[float] = field(default_factory=list)
    history_res_GR_rel: list[float] = field(default_factory=list)
    history_res_Gless_abs: list[float] = field(default_factory=list)
    history_res_Gless_rel: list[float] = field(default_factory=list)


class Solver:
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

        self.max_iter = sys.scba_max_iter
        self.tol_abs = sys.scba_tol_abs
        self.tol_rel = sys.scba_tol_rel
        self.mixing = sys.scba_mixing
        self.min_iter = sys.scba_min_iter
        self.verbose = sys.verbose

        self.GR_values: Optional[np.ndarray] = None
        self.Gless_values: Optional[np.ndarray] = None

        self.result: Optional[SolverResult] = None

        self.history_res_GR_abs: list[float] = []
        self.history_res_GR_rel: list[float] = []
        self.history_res_Gless_abs: list[float] = []
        self.history_res_Gless_rel: list[float] = []

    def initialize(self) -> None:
        GR0 = np.asarray(GR_eq(self.sys, self.w), dtype=np.complex128)

        f0 = np.asarray(
            fermi_dirac(self.w, self.sys.beta_fd, self.sys.mu_fd),
            dtype=np.complex128,
        )

        Gless0 = -2j * np.imag(GR0) * f0

        self.GR_values = GR0
        self.Gless_values = Gless0

        self.history_res_GR_abs = []
        self.history_res_GR_rel = []
        self.history_res_Gless_abs = []
        self.history_res_Gless_rel = []

        self.result = None

    def _require_initialized(self) -> None:
        if self.GR_values is None or self.Gless_values is None:
            raise RuntimeError("Solver not initialized. Call initialize() or solve().")

    def GR(self, e):
        self._require_initialized()
        return self._interp_complex(self.GR_values, e)

    def Gless(self, e):
        self._require_initialized()
        return self._interp_complex(self.Gless_values, e)

    def GA(self, e):
        return np.conjugate(self.GR(e))

    def _interp_complex(self, y_grid: np.ndarray, x):
        x_arr = np.asarray(x, dtype=np.complex128)
        xr = x_arr.real

        yr = np.interp(xr, self.w, np.real(y_grid))
        yi = np.interp(xr, self.w, np.imag(y_grid))
        return yr + 1j * yi

    def compute_sigma_less_from_state(self) -> np.ndarray:
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

    def update_trials(self) -> tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def linear_mix(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * old + alpha * new

    @staticmethod
    def integrated_l2_norm(x: np.ndarray, dw: float) -> float:
        if dw <= 0.0:
            return float(np.linalg.norm(x))
        return float(np.sqrt(np.sum(np.abs(x) ** 2) * dw))

    def absolute_residual(self, current: np.ndarray, trial: np.ndarray) -> float:
        return self.integrated_l2_norm(current - trial, self.dw)

    def relative_residual(
        self,
        current: np.ndarray,
        trial: np.ndarray,
        eps: Optional[float] = None,
    ) -> float:
        if eps is None:
            eps = float(self.sys.ETA)

        num = self.integrated_l2_norm(current - trial, self.dw)
        den = max(self.integrated_l2_norm(current, self.dw), float(eps))
        return float(num / den)

    def solve(self) -> SolverResult:
        if self.GR_values is None or self.Gless_values is None:
            self.initialize()

        rep = self.sys.reporter()

        if self.sys.verbose:
            rep.section("SCBA nonequilibrium solve")
            rep.info(
                f"grid points = {len(self.w)} | "
                f"window = [{self.w[0]}, {self.w[-1]}] | "
                f"mixing = {self.mixing} | "
                f"max_iter = {self.max_iter}"
            )

        converged = False

        res_GR_abs = np.inf
        res_GR_rel = np.inf
        res_Gless_abs = np.inf
        res_Gless_rel = np.inf

        iterator = range(1, self.max_iter + 1)
        pbar = None

        if self.sys.verbose and tqdm is not None:
            pbar = tqdm(
                iterator,
                desc="SCBA solve",
                total=self.max_iter,
                unit="iter",
                dynamic_ncols=True,
                leave=True,
            )
            iterator = pbar

        for it in iterator:
            GR_trial, Gless_trial = self.update_trials()

            # Fixed-point residuals: compare the current state to the unmixed trial map.
            res_GR_abs = self.absolute_residual(self.GR_values, GR_trial)
            res_GR_rel = self.relative_residual(
                self.GR_values,
                GR_trial,
                eps=self.sys.ETA,
            )

            res_Gless_abs = self.absolute_residual(self.Gless_values, Gless_trial)
            res_Gless_rel = self.relative_residual(
                self.Gless_values,
                Gless_trial,
                eps=self.sys.ETA,
            )

            self.history_res_GR_abs.append(res_GR_abs)
            self.history_res_GR_rel.append(res_GR_rel)
            self.history_res_Gless_abs.append(res_Gless_abs)
            self.history_res_Gless_rel.append(res_Gless_rel)

            GR_next = self.linear_mix(self.GR_values, GR_trial, self.mixing)
            Gless_next = self.linear_mix(self.Gless_values, Gless_trial, self.mixing)

            self.GR_values = GR_next
            self.Gless_values = Gless_next

            if pbar is not None:
                pbar.set_postfix(
                    GR_abs=f"{res_GR_abs:.5e}",
                    GR_rel=f"{res_GR_rel:.5e}",
                    Gl_abs=f"{res_Gless_abs:.5e}",
                    Gl_rel=f"{res_Gless_rel:.5e}",
                    mix=f"{self.mixing}",
                )
            elif self.sys.verbose:
                rep.info(
                    f"[SCBA] iter={it:7d} "
                    f"GR_abs={res_GR_abs:.5e} "
                    f"GR_rel={res_GR_rel:.5e} "
                    f"G<_abs={res_Gless_abs:.5e} "
                    f"G<_rel={res_Gless_rel:.5e}"
                )

            gr_ok = (res_GR_abs < self.tol_abs) and (res_GR_rel < self.tol_rel)
            gl_ok = (res_Gless_abs < self.tol_abs) and (res_Gless_rel < self.tol_rel)

            if it >= self.min_iter and gr_ok and gl_ok:
                converged = True
                break

        if pbar is not None:
            pbar.set_postfix(
                status="ok" if converged else "stop",
                GR_abs=f"{res_GR_abs:.5e}",
                GR_rel=f"{res_GR_rel:.5e}",
                Gl_abs=f"{res_Gless_abs:.5e}",
                Gl_rel=f"{res_Gless_rel:.5e}",
            )
            pbar.close()

        self.result = SolverResult(
            converged=converged,
            n_iter=it,
            res_GR_abs=res_GR_abs,
            res_GR_rel=res_GR_rel,
            res_Gless_abs=res_Gless_abs,
            res_Gless_rel=res_Gless_rel,
            history_res_GR_abs=self.history_res_GR_abs.copy(),
            history_res_GR_rel=self.history_res_GR_rel.copy(),
            history_res_Gless_abs=self.history_res_Gless_abs.copy(),
            history_res_Gless_rel=self.history_res_Gless_rel.copy(),
        )

        if self.sys.verbose:
            rep.info(
                f"SCBA finished | converged={self.result.converged} | "
                f"iterations={self.result.n_iter} | "
                f"GR_abs={self.result.res_GR_abs:.5e} | "
                f"GR_rel={self.result.res_GR_rel:.5e} | "
                f"G<_abs={self.result.res_Gless_abs:.5e} | "
                f"G<_rel={self.result.res_Gless_rel:.5e}"
            )

        return self.result