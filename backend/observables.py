from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.integrate import quad_vec

from backend.distribution import fermi_dirac, bose_einstein, expc
from backend.green_function import GR_eq, get_eq_poles_residues, R_gamma
from backend.system_classes import System, Lead


def _omega_int_x_bounds(sys: System, margin: float = 2.0) -> tuple[float, float]:
    equi_poles = get_eq_poles_residues(sys)

    delta_diffs = [
        sys.Delta(alpha) - sys.Delta(mu)
        for alpha in sys.lead_names
        for mu in sys.lead_names
    ]
    delta_min = min(delta_diffs) if delta_diffs else 0.0
    delta_max = max(delta_diffs) if delta_diffs else 0.0

    pole_reals = [complex(w_r).real for _, w_r in equi_poles]
    pole_min = min(pole_reals) if pole_reals else sys.e_min
    pole_max = max(pole_reals) if pole_reals else sys.e_max

    x_min = min(sys.e_min + delta_min, pole_min) - margin
    x_max = max(sys.e_max + delta_max, pole_max) + margin

    return float(x_min), float(x_max)


def A(sys: System, e, t, alpha: Lead):
    Delta_alpha = sys.Delta(alpha)
    nph = bose_einstein(sys.w_q, sys.beta_ph, sys.mu_ph)

    term1 = GR_eq(sys, e)
    equi_poles = get_eq_poles_residues(sys)

    t = np.asarray(t, dtype=float)
    term2 = np.zeros_like(t, dtype=np.complex128)

    GR_noneq_e_alpha = sys.GR_noneq(e + Delta_alpha)

    for R_r, w_r in equi_poles:
        omega_int_wr = sys.omega_int(w_r)

        GR_noneq_w_r = sys.GR_noneq(w_r)
        GR_noneq_w_rmw_q = sys.GR_noneq(w_r - sys.w_q)
        GR_noneq_w_rpw_q = sys.GR_noneq(w_r + sys.w_q)

        term2_1 = Delta_alpha / (w_r - e)
        term2_2 = sys.DELTA * GR_noneq_e_alpha

        term2_3 = (
            (2j * sys.w_q * sys.g_q**2 / np.pi)
            * GR_noneq_e_alpha
            * omega_int_wr
        )

        term2_4 = (
            (2j * sys.w_q * sys.g_q**2 * sys.DELTA / np.pi)
            * GR_noneq_e_alpha
            * GR_noneq_w_r
            * omega_int_wr
        )

        term2_5 = (
            2.0 * (nph + 1.0)
            * GR_noneq_w_rmw_q
            * sys.g_q**2
            * GR_noneq_e_alpha
        )

        term2_6 = (
            2.0 * (nph + 1.0)
            * GR_noneq_w_rmw_q
            * sys.g_q**2
            * GR_noneq_e_alpha
            * sys.DELTA
            * GR_noneq_w_r
        )

        term2_7 = (
            2.0 * nph
            * GR_noneq_w_rpw_q
            * sys.g_q**2
            * GR_noneq_e_alpha
        )

        term2_8 = (
            2.0 * nph
            * GR_noneq_w_rpw_q
            * sys.g_q**2
            * GR_noneq_e_alpha
            * sys.DELTA
            * GR_noneq_w_r
        )

        term2_9 = 0.0 + 0.0j
        for mu in sys.lead_names:
            Delta_mu = sys.Delta(mu)
            Gamma_mu = sys.Gamma0(mu)

            x_em = e + Delta_alpha - Delta_mu
            omega_int_em = sys.omega_int(x_em)

            GR_noneq_em = sys.GR_noneq(x_em)
            GR_noneq_em_mwq = sys.GR_noneq(x_em - sys.w_q)
            GR_noneq_em_pwq = sys.GR_noneq(x_em + sys.w_q)

            term2_9_1 = 0.5 * (
                (Gamma_mu * sys.W) / (w_r + 1j * sys.W)
                - (Gamma_mu * sys.W) / (x_em + 1j * sys.W)
            ) / (w_r - e + Delta_mu - Delta_alpha + 1j * sys.ETA)

            term2_9_2_1 = 2j * sys.w_q * omega_int_wr
            term2_9_2_2 = -2.0 * (nph + 1.0) * GR_noneq_w_rmw_q
            term2_9_2_3 = -2.0 * nph * GR_noneq_w_rpw_q

            term2_9_2 = 0.5 * (
                (Gamma_mu * sys.W * GR_noneq_w_r * sys.g_q**2)
                / ((w_r + 1j * sys.W) * (w_r - e + Delta_mu - Delta_alpha + 1j * sys.ETA))
            ) * (term2_9_2_1 + term2_9_2_2 + term2_9_2_3)

            term2_9_3_1 = -2j * sys.w_q * omega_int_em
            term2_9_3_2 = -2.0 * (nph + 1.0) * GR_noneq_em_mwq
            term2_9_3_3 = -2.0 * nph * GR_noneq_em_pwq

            term2_9_3 = 0.5 * (
                (Gamma_mu * sys.W * GR_noneq_em * sys.g_q**2)
                / ((x_em + 1j * sys.W) * (w_r - e + Delta_mu - Delta_alpha + 1j * sys.ETA))
            ) * (term2_9_3_1 + term2_9_3_2 + term2_9_3_3)

            term2_9 -= Delta_mu * (term2_9_1 + term2_9_2 + term2_9_3) * GR_noneq_e_alpha

        pref = -np.exp(-1j * (w_r - e) * t) * R_r / (w_r - e - Delta_alpha)

        term2 += pref * (
            term2_1
            + term2_2
            + term2_3
            + term2_4
            + term2_5
            + term2_6
            + term2_7
            + term2_8
            + term2_9
        )

    return term1 + term2


def B(sys: System, e, e_prime, t, alpha: Lead):
    Delta_alpha = sys.Delta(alpha)
    nph = bose_einstein(sys.w_q, sys.beta_ph, sys.mu_ph)

    t = np.asarray(t, dtype=float)
    term1 = expc(e - e_prime, t) * GR_eq(sys, e_prime)

    equi_poles = get_eq_poles_residues(sys)
    term2 = np.zeros_like(t, dtype=np.complex128)

    GR_noneq_e_prime_alpha = sys.GR_noneq(e_prime + Delta_alpha)

    for R_r, w_r in equi_poles:
        omega_int_wr = sys.omega_int(w_r)

        GR_noneq_w_r = sys.GR_noneq(w_r)
        GR_noneq_w_rmw_q = sys.GR_noneq(w_r - sys.w_q)
        GR_noneq_w_rpw_q = sys.GR_noneq(w_r + sys.w_q)

        term2_1 = Delta_alpha / (w_r - e_prime)
        term2_2 = sys.DELTA * GR_noneq_e_prime_alpha

        term2_3 = (
            (2j * sys.w_q * sys.g_q**2 / np.pi)
            * GR_noneq_e_prime_alpha
            * omega_int_wr
        )

        term2_4 = (
            (2j * sys.w_q * sys.g_q**2 * sys.DELTA / np.pi)
            * GR_noneq_e_prime_alpha
            * GR_noneq_w_r
            * omega_int_wr
        )

        term2_5 = (
            2.0 * (nph + 1.0)
            * GR_noneq_w_rmw_q
            * sys.g_q**2
            * GR_noneq_e_prime_alpha
        )

        term2_6 = (
            2.0 * (nph + 1.0)
            * GR_noneq_w_rmw_q
            * sys.g_q**2
            * GR_noneq_e_prime_alpha
            * sys.DELTA
            * GR_noneq_w_r
        )

        term2_7 = (
            2.0 * nph
            * GR_noneq_w_rpw_q
            * sys.g_q**2
            * GR_noneq_e_prime_alpha
        )

        term2_8 = (
            2.0 * nph
            * GR_noneq_w_rpw_q
            * sys.g_q**2
            * GR_noneq_e_prime_alpha
            * sys.DELTA
            * GR_noneq_w_r
        )

        term2_9 = 0.0 + 0.0j
        for mu in sys.lead_names:
            Delta_mu = sys.Delta(mu)
            Gamma_mu = sys.Gamma0(mu)

            x_em = e_prime + Delta_alpha - Delta_mu
            omega_int_em = sys.omega_int(x_em)

            GR_noneq_em = sys.GR_noneq(x_em)
            GR_noneq_em_mwq = sys.GR_noneq(x_em - sys.w_q)
            GR_noneq_em_pwq = sys.GR_noneq(x_em + sys.w_q)

            term2_9_1 = 0.5 * (
                (Gamma_mu * sys.W) / (w_r + 1j * sys.W)
                - (Gamma_mu * sys.W) / (x_em + 1j * sys.W)
            ) / (w_r - e_prime + Delta_mu - Delta_alpha + 1j * sys.ETA)

            term2_9_2_1 = 2j * sys.w_q * omega_int_wr
            term2_9_2_2 = -2.0 * (nph + 1.0) * GR_noneq_w_rmw_q
            term2_9_2_3 = -2.0 * nph * GR_noneq_w_rpw_q

            term2_9_2 = 0.5 * (
                (Gamma_mu * sys.W * GR_noneq_w_r * sys.g_q**2)
                / ((w_r + 1j * sys.W) * (w_r - e_prime + Delta_mu - Delta_alpha + 1j * sys.ETA))
            ) * (term2_9_2_1 + term2_9_2_2 + term2_9_2_3)

            term2_9_3_1 = -2j * sys.w_q * omega_int_em
            term2_9_3_2 = -2.0 * (nph + 1.0) * GR_noneq_em_mwq
            term2_9_3_3 = -2.0 * nph * GR_noneq_em_pwq

            term2_9_3 = 0.5 * (
                (Gamma_mu * sys.W * GR_noneq_em * sys.g_q**2)
                / ((x_em + 1j * sys.W) * (w_r - e_prime + Delta_mu - Delta_alpha + 1j * sys.ETA))
            ) * (term2_9_3_1 + term2_9_3_2 + term2_9_3_3)

            term2_9 -= Delta_mu * (term2_9_1 + term2_9_2 + term2_9_3) * GR_noneq_e_prime_alpha

        pref = -expc(e - w_r, t) * R_r / (w_r - e_prime - Delta_alpha)

        term2 += pref * (
            term2_1
            + term2_2
            + term2_3
            + term2_4
            + term2_5
            + term2_6
            + term2_7
            + term2_8
            + term2_9
        )

    return term1 + term2


def current_alpha(
    sys: System,
    alpha: Lead,
    t_max: float,
    n_t: int,
    omega_int_n_x: int | None = None,
    omega_int_n_omega: int | None = None,
):
    rep = sys.reporter()

    if sys.verbose:
        rep.section(f"Current evaluation for lead {alpha}")
        rep.info(f"t_max = {t_max:.6f} | n_t = {n_t}")

    t = np.linspace(0.0, t_max, n_t)
    Delta_alpha = sys.Delta(alpha)
    Gamma_alpha = sys.Gamma0(alpha)
    R_alpha = R_gamma(sys, alpha, "+")

    result = sys.solve_noneq()

    if sys.verbose:
        rep.info(
            "nonequilibrium solution ready | "
            f"converged={result.converged} | "
            f"iterations={result.n_iter} | "
            f"GR_res_abs={result.res_GR_abs:.3e} | "
            f"GR_res_rel={result.res_GR_rel:.3e} | "
            f"G<_res_abs={result.res_Gless_abs:.3e} | "
            f"G<_res_rel={result.res_Gless_rel:.3e}"
        )

    x_min, x_max = _omega_int_x_bounds(sys)

    sys.build_omega_int_table(
        x_min=x_min,
        x_max=x_max,
        n_x=omega_int_n_x,
        n_omega=omega_int_n_omega,
    )

    nph = bose_einstein(sys.w_q, sys.beta_ph, sys.mu_ph)

    def integrand1(e):
        return (
            fermi_dirac(e, sys.beta_fc(alpha), sys.mu_fc(alpha))
            * A(sys, e, t, alpha)
            * (e**2 + sys.W**2) ** (-1)
        )

    if sys.verbose:
        with rep.timed("Evaluating term1"):
            val1, _ = quad_vec(integrand1, sys.e_min, sys.e_max)
    else:
        val1, _ = quad_vec(integrand1, sys.e_min, sys.e_max)

    term1 = -2 * sp.constants.e * (Gamma_alpha * sys.W**2 / (2 * np.pi)) * np.imag(val1)

    term2 = np.zeros_like(t, dtype=np.float64)
    term3 = np.zeros_like(t, dtype=np.float64)

    if sys.verbose:
        rep.info("Summing over lead index beta for term2 and term3")

    for beta in sys.lead_names:
        Gamma_beta = sys.Gamma0(beta)
        Delta_beta = sys.Delta(beta)
        beta_beta = sys.beta_fc(beta)
        mu_beta = sys.mu_fc(beta)

        if sys.verbose:
            rep.info(
                f"  beta = {beta} | "
                f"Gamma_beta = {Gamma_beta:.6f} | "
                f"Delta_beta = {Delta_beta:.6f}"
            )

        def integrand2(e):
            return (
                np.exp(-1j * e * t)
                * fermi_dirac(e, beta_beta, mu_beta)
                * A(sys, e, t, beta)
                * Gamma_beta
                * (e**2 + sys.W**2) ** (-1)
                * np.conjugate(sys.GR_noneq(e + Delta_beta))
                * 1j
                * R_alpha
                * np.exp(-sys.W * t)
                * (1j * sys.W - e + Delta_alpha - Delta_beta) ** (-1)
            )

        if sys.verbose:
            with rep.timed(f"Evaluating term2 contribution for beta={beta}"):
                val2, _ = quad_vec(integrand2, sys.e_min, sys.e_max)
        else:
            val2, _ = quad_vec(integrand2, sys.e_min, sys.e_max)

        term2 += 2 * sp.constants.e * (sys.W**2 / (2 * np.pi)) * np.imag(val2)

        def integrand3(e):
            return (
                np.exp(-1j * e * t)
                * fermi_dirac(e, beta_beta, mu_beta)
                * A(sys, e, t, beta)
                * Gamma_beta
                * (e**2 + sys.W**2) ** (-1)
                * R_alpha
                * np.exp(-sys.W * t)
                * np.conjugate(B(sys, 1j * sys.W, e, t, beta))
            )

        if sys.verbose:
            with rep.timed(f"Evaluating term3 contribution for beta={beta}"):
                val3, _ = quad_vec(integrand3, sys.e_min, sys.e_max)
        else:
            val3, _ = quad_vec(integrand3, sys.e_min, sys.e_max)

        term3 += 2 * sp.constants.e * (sys.W**2 / (2 * np.pi)) * np.imag(val3)

    def integrand4(e):
        return (
            sys.GR_noneq(e)
            * np.conjugate(sys.GR_noneq(e))
            * (
                sys.Gless_noneq(e - sys.w_q) * nph
                + sys.Gless_noneq(e + sys.w_q) * (nph + 1.0)
            )
            * (
                (1.0 - np.exp(-(sys.W + 1j * e) * t)) / (sys.W + 1j * e)
                - (1j * (e - Delta_alpha)) / ((e - Delta_alpha) ** 2 + sys.W**2)
            )
        )

    if sys.verbose:
        with rep.timed("Evaluating term4"):
            val4, _ = quad_vec(integrand4, sys.e_min, sys.e_max)
    else:
        val4, _ = quad_vec(integrand4, sys.e_min, sys.e_max)

    term4 = -2 * sp.constants.e * sys.g_q**2 * (sys.W * Gamma_alpha / (2 * np.pi)) * np.imag(val4)

    def integrand5(e):
        return (
            sys.GR_noneq(e + Delta_alpha)
            * np.conjugate(sys.GR_noneq(e + Delta_alpha))
            * sys.Gless_noneq(e + Delta_alpha - sys.w_q)
            * nph
        )

    if sys.verbose:
        with rep.timed("Evaluating term5"):
            val5, _ = quad_vec(integrand5, sys.e_min, sys.e_max)
    else:
        val5, _ = quad_vec(integrand5, sys.e_min, sys.e_max)

    term5 = -2 * sp.constants.e * sys.g_q**2 * (sys.W**2 * Gamma_alpha / (2 * np.pi)) * np.imag(val5)

    def integrand6(e):
        return (
            sys.GR_noneq(e + Delta_alpha)
            * np.conjugate(sys.GR_noneq(e + Delta_alpha))
            * sys.Gless_noneq(e + Delta_alpha + sys.w_q)
            * (nph + 1.0)
        )

    if sys.verbose:
        with rep.timed("Evaluating term6"):
            val6, _ = quad_vec(integrand6, sys.e_min, sys.e_max)
    else:
        val6, _ = quad_vec(integrand6, sys.e_min, sys.e_max)

    term6 = -2 * sp.constants.e * sys.g_q**2 * (sys.W**2 * Gamma_alpha / (2 * np.pi)) * np.imag(val6)

    I_alpha = term1 + term2 + term3 + term4 + term5 + term6

    if sys.verbose:
        rep.info(
            f"Current evaluation finished for lead {alpha} | "
            f"max|I| = {np.max(np.abs(I_alpha)):.6e}"
        )

    return t, I_alpha