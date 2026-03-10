from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

Lead = str

if TYPE_CHECKING:
    from backend.SCBA import Solver, SolverResult


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

    # phonon bath
    beta_ph: float = 1.0
    mu_ph: float = 0.0

    # central region
    beta_fd: float = 1.0
    mu_fd: float = 0.0

    # integration / frequency windows
    e_min: float = -10.0
    e_max: float = 10.0
    omega_min: float = -10.0
    omega_max: float = 10.0

    # solver configuration
    scba_max_iter: int = 200
    scba_tol_abs: float = 1e-8
    scba_tol_rel: float = 1e-6
    scba_mixing: float = 0.05
    scba_min_iter: int = 5
    scba_verbose: bool = True

    # cached nonequilibrium solver state
    _noneq_result: Optional["SolverResult"] = None
    _noneq_solver: Optional["Solver"] = None

    @property
    def lead_names(self) -> List[Lead]:
        return list(self.leads.keys())

    # -----------------------------
    # lead-dependent accessors
    # -----------------------------
    def Gamma0(self, lead: Lead) -> float:
        return self.leads[lead].Gamma0

    def Delta(self, lead: Lead) -> float:
        return self.leads[lead].Delta

    def beta_fc(self, lead: Lead) -> float:
        return self.leads[lead].beta

    def mu_fc(self, lead: Lead) -> float:
        return self.leads[lead].mu

    # -----------------------------
    # nonequilibrium solver interface
    # -----------------------------
    def solve_noneq(
        self,
        w_min: Optional[float] = None,
        w_max: Optional[float] = None,
        n_w: int = 2001,
        force: bool = False,
    ):
        """
        Solve the nonequilibrium SCBA problem and cache the solver.

        Parameters
        ----------
        w_min, w_max
            Frequency window for the solver grid. If omitted, use
            self.omega_min and self.omega_max.
        n_w
            Number of frequency points in the solver grid.
        force
            If True, discard the cached solution and recompute.
        """
        if self._noneq_solver is not None and self._noneq_result is not None and not force:
            return self._noneq_result

        from backend.SCBA import Solver

        solver = Solver(
            self,
            w_min=self.omega_min if w_min is None else w_min,
            w_max=self.omega_max if w_max is None else w_max,
            n_w=n_w,
        )

        result = solver.solve()
        self._noneq_solver = solver
        self._noneq_result = result
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

    @property
    def noneq_result(self):
        self.require_noneq()
        return self._noneq_result

    @property
    def noneq_grid(self):
        self.require_noneq()
        return self._noneq_solver.w

    @property
    def GR_noneq_values(self):
        self.require_noneq()
        return self._noneq_solver.GR_values

    @property
    def Gless_noneq_values(self):
        self.require_noneq()
        return self._noneq_solver.Gless_values

    def clear_noneq(self) -> None:
        self._noneq_solver = None
        self._noneq_result = None