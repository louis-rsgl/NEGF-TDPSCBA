from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

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

    verbose: bool = True

    _cached_eq_poles: list | None = None
    _noneq_result: Optional["SolverResult"] = None
    _noneq_solver: Optional["Solver"] = None
    _reporter: Optional["Reporter"] = field(default=None, init=False, repr=False)

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

        rep.info("Leads:")
        for lead in self.lead_names:
            rep.info(
                f"  {lead}: "
                f"Gamma0={self.Gamma0(lead):.6e}, "
                f"Delta={self.Delta(lead):.6e}, "
                f"beta={self.beta_fc(lead):.6e}, "
                f"mu={self.mu_fc(lead):.6e}"
            )

    def solve_noneq(
        self,
        w_min: Optional[float] = None,
        w_max: Optional[float] = None,
        n_w: int = 2001,
        force: bool = False,
    ):
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