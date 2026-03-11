from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

Lead = str

if TYPE_CHECKING:
    from backend.SCBA import Solver, SolverResult
    from backend.reporting import Reporter


@dataclass
class LeadParams:
    Gamma0: float
    Delta: float
    beta: float
    mu: float


@dataclass
class System:
    ETA: float = 1e-5
    DELTA: float = 0.0

    leads: Dict[Lead, LeadParams] = field(default_factory=dict)

    W: float = 1.0
    g_q: float = 0.0
    w_q: float = 0.0
    e_0: float = 0.0

    beta_ph: float = 1.0
    mu_ph: float = 0.0

    beta_fd: float = 1.0
    mu_fd: float = 0.0

    e_min: float = -10.0
    e_max: float = 10.0
    omega_min: float = -10.0
    omega_max: float = 10.0

    scba_max_iter: int = 200
    scba_tol_abs: float = 1e-8
    scba_tol_rel: float = 1e-6
    scba_mixing: float = 0.05
    scba_min_iter: int = 5
    n_w_scba: int = 2001

    verbose: bool = True

    _cached_eq_poles: list | None = None
    _noneq_result: Optional["SolverResult"] = None
    _noneq_solver: Optional["Solver"] = None
    _reporter: Optional["Reporter"] = field(default=None, init=False, repr=False)

    _omega_int_xgrid: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _omega_int_values: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _omega_int_omega_grid: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @property
    def lead_names(self) -> List[Lead]:
        return list(self.leads.keys())

    def Gamma0(self, lead: Lead) -> float:
        return self.leads[lead].Gamma0

    def Delta(self, lead: Lead) -> float:
        return self.leads[lead].Delta

    def beta_fc(self, lead: Lead) -> float:
        return self.leads[lead].beta

    def mu_fc(self, lead: Lead) -> float:
        return self.leads[lead].mu

    def reporter(self) -> "Reporter":
        if self._reporter is None:
            from backend.reporting import Reporter
            self._reporter = Reporter(verbose=self.verbose)
        return self._reporter

    def launch(self) -> None:
        rep = self.reporter()

        rep.banner()
        rep.section("System configuration")

        rep.info(f"ETA = {self.ETA:.6e}")
        rep.info(f"DELTA = {self.DELTA:.6e}")
        rep.info(f"W = {self.W:.6e}")
        rep.info(f"g_q = {self.g_q:.6e}")
        rep.info(f"w_q = {self.w_q:.6e}")
        rep.info(f"e_0 = {self.e_0:.6e}")
        rep.info(f"beta_ph = {self.beta_ph:.6e}")
        rep.info(f"mu_ph = {self.mu_ph:.6e}")
        rep.info(f"beta_fd = {self.beta_fd:.6e}")
        rep.info(f"mu_fd = {self.mu_fd:.6e}")
        rep.info(f"e window = [{self.e_min:.6e}, {self.e_max:.6e}]")
        rep.info(f"omega window = [{self.omega_min:.6e}, {self.omega_max:.6e}]")

        rep.info("SCBA solver parameters:")
        rep.info(f"  max_iter = {self.scba_max_iter}")
        rep.info(f"  tol_abs  = {self.scba_tol_abs:.3e}")
        rep.info(f"  tol_rel  = {self.scba_tol_rel:.3e}")
        rep.info(f"  mixing   = {self.scba_mixing:.3e}")
        rep.info(f"  min_iter = {self.scba_min_iter}")
        rep.info(f"  n_w      = {self.n_w_scba}")

        rep.info("Leads:")
        for lead in self.lead_names:
            rep.info(
                f"  {lead}: "
                f"Gamma0={self.Gamma0(lead):.6e}, "
                f"Delta={self.Delta(lead):.6e}, "
                f"beta={self.beta_fc(lead):.6e}, "
                f"mu={self.mu_fc(lead):.6e}"
            )

    def invalidate_noneq_cache(self) -> None:
        self._noneq_result = None
        self._noneq_solver = None
        self.invalidate_omega_int_cache()

    def invalidate_omega_int_cache(self) -> None:
        self._omega_int_xgrid = None
        self._omega_int_values = None
        self._omega_int_omega_grid = None

    def solve_noneq(
        self,
        w_min: Optional[float] = None,
        w_max: Optional[float] = None,
        n_w: Optional[int] = None,
        force: bool = False,
    ):
        if self._noneq_solver is not None and self._noneq_result is not None and not force:
            return self._noneq_result

        from backend.SCBA import Solver

        solver = Solver(
            self,
            w_min=self.omega_min if w_min is None else w_min,
            w_max=self.omega_max if w_max is None else w_max,
            n_w=self.n_w_scba if n_w is None else n_w,
        )

        result = solver.solve()
        self._noneq_solver = solver
        self._noneq_result = result

        self.invalidate_omega_int_cache()

        return result

    def require_noneq(self) -> None:
        if self._noneq_solver is None or self._noneq_result is None:
            raise RuntimeError("Nonequilibrium solution not available. Call solve_noneq() first.")

    def GR_noneq(self, e):
        self.require_noneq()
        return self._noneq_solver.GR(e)

    def Gless_noneq(self, e):
        self.require_noneq()
        return self._noneq_solver.Gless(e)

    def GA_noneq(self, e):
        self.require_noneq()
        return self._noneq_solver.GA(e)

    def build_omega_int_table(
        self,
        x_min: float,
        x_max: float,
        n_x: Optional[int] = None,
        n_omega: Optional[int] = None,
        force: bool = False,
    ) -> None:
        self.require_noneq()

        if (
            self._omega_int_xgrid is not None
            and self._omega_int_values is not None
            and not force
        ):
            return

        if self._noneq_solver is not None:
            default_n = len(self._noneq_solver.w)
        else:
            default_n = self.n_w_scba

        if n_x is None:
            n_x = default_n
        if n_omega is None:
            n_omega = default_n

        rep = self.reporter()

        x_grid = np.linspace(x_min, x_max, n_x, dtype=float)
        omega_grid = np.linspace(self.omega_min, self.omega_max, n_omega, dtype=float)

        kernel = 1.0 / (
            (omega_grid - self.w_q + 1j * self.ETA)
            * (omega_grid + self.w_q + 1j * self.ETA)
        )

        values = np.empty_like(x_grid, dtype=np.complex128)

        if self.verbose:
            rep.info(
                "Building omega-integral table | "
                f"x-range=[{x_grid[0]:.6e}, {x_grid[-1]:.6e}] | "
                f"n_x={n_x} | n_omega={n_omega}"
            )

        for i, x in enumerate(x_grid):
            integrand = self.Gless_noneq(x - omega_grid) * kernel
            values[i] = np.trapezoid(integrand, omega_grid)

        self._omega_int_xgrid = x_grid
        self._omega_int_values = values
        self._omega_int_omega_grid = omega_grid

    def omega_int(self, x):
        if self._omega_int_xgrid is None or self._omega_int_values is None:
            raise RuntimeError(
                "Omega-integral table not available. Call build_omega_int_table() first."
            )

        x_arr = np.asarray(x, dtype=np.complex128)
        xr = x_arr.real

        x_min = float(self._omega_int_xgrid[0])
        x_max = float(self._omega_int_xgrid[-1])

        if np.any(xr < x_min) or np.any(xr > x_max):
            raise ValueError(
                "omega_int queried outside precomputed table range: "
                f"valid range is [{x_min}, {x_max}]"
            )

        yr = np.interp(xr, self._omega_int_xgrid, np.real(self._omega_int_values))
        yi = np.interp(xr, self._omega_int_xgrid, np.imag(self._omega_int_values))
        return yr + 1j * yi