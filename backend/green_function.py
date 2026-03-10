from backend.distribution import bose_einstein, fermi_dirac
from backend.system_classes import System, Lead
import sympy as sy

def GR_eq(sys: System, w):
    """
    Numerical equilibrium retarded Green function G^R(w).

    Parameters
    ----------
    sys : System
        System object containing model and distribution parameters.
    w : complex or np.ndarray
        Frequency value(s).

    Returns
    -------
    complex or np.ndarray
        Numerical value of G^R(w).
    """
    e0 = sys.e_0
    W = sys.W
    gq = sys.g_q
    wq = sys.w_q
    eta = sys.ETA

    # Phonon occupation
    n_b = bose_einstein(wq, sys.beta_ph, sys.mu_ph)

    # Central-region electronic occupation
    f_e0 = fermi_dirac(e0, sys.beta_fd, sys.mu_fd)

    # Sum over lead broadenings
    gamma_sum = sum(sys.Gamma0(mu) for mu in sys.lead_names)

    Sigma_expr = (
        0.5 * gamma_sum * W / (w + 1j * W)
        + gq**2 * (n_b + f_e0) / (w + wq - e0 + 1j * eta)
        + gq**2 * (1 + n_b - f_e0) / (w - wq - e0 + 1j * eta)
    )

    G_expr = 1.0 / (w - e0 - Sigma_expr)
    return G_expr


def build_GR_eq_symbolic(sys: System):
    """
    Build the symbolic equilibrium retarded Green function G^R(w).

    Parameters
    ----------
    sys : System
        System object containing model and distribution parameters.

    Returns
    -------
    w : sympy.Symbol
        Complex frequency variable.
    G_expr : sympy.Expr
        Symbolic expression for G^R(w).
    """
    w = sy.symbols("w")
    I = sy.I

    # Model parameters as SymPy constants
    e0 = sy.nsimplify(sys.e_0)
    W = sy.nsimplify(sys.W)
    gq = sy.nsimplify(sys.g_q)
    wq = sy.nsimplify(sys.w_q)
    eta = sy.nsimplify(sys.ETA)

    # Constants with respect to w
    n_b = sy.nsimplify(complex(bose_einstein(sys.w_q, sys.beta_ph, sys.mu_ph)))
    f_e0 = sy.nsimplify(complex(fermi_dirac(sys.e_0, sys.beta_fd, sys.mu_fd)))
    gamma_sum = sy.nsimplify(sum(sys.Gamma0(mu) for mu in sys.lead_names))

    Sigma_expr = (
        sy.Rational(1, 2) * gamma_sum * W / (w + I * W)
        + gq**2 * (n_b + f_e0) / (w + wq - e0 + I * eta)
        + gq**2 * (1 + n_b - f_e0) / (w - wq - e0 + I * eta)
    )

    G_expr = 1 / (w - e0 - Sigma_expr)
    return w, G_expr


def get_eq_poles_residues(sys: System):
    if sys._cached_eq_poles is None:
        sys._cached_eq_poles = eq_poles_residues(sys)
    return sys._cached_eq_poles


def eq_poles_residues(sys: System):
    """
    Compute poles and residues of the equilibrium retarded Green function.

    Returns
    -------
    list[tuple[complex, complex]]
        List of (residue, pole) pairs.
    """
    w, G_expr = build_GR_eq_symbolic(sys)

    num, den = sy.fraction(sy.together(G_expr))
    num = sy.expand(num)
    den = sy.expand(den)

    poly = sy.Poly(den, w)
    deg = poly.degree()
    if deg != 4:
        raise RuntimeError(
            "Expected quartic denominator, got degree "
            f"{deg}\n\n"
            f"G_expr:\n{sy.pretty(G_expr)}"
        )

    roots = poly.nroots()
    dden = sy.diff(den, w)

    out = []
    for r in roots:
        r_num = complex(sy.N(r, 30))
        num_at_r = complex(sy.N(num.subs(w, r), 30))
        dden_at_r = complex(sy.N(dden.subs(w, r), 30))

        if abs(dden_at_r) < 1e-14:
            raise RuntimeError(
                f"Residue evaluation unstable: D'(w) too small at root {r_num}"
            )

        residue = num_at_r / dden_at_r
        out.append((residue, r_num))

    return out


def R_gamma(sys: System, alpha: Lead, sign: str) -> complex:
    """
    Residue of Gamma_alpha(w) at w = ± iW.

    R_gamma(+) = - i Gamma0_alpha W / 2
    R_gamma(-) = + i Gamma0_alpha W / 2
    """
    if sign == "+":
        return -0.5j * sys.Gamma0(alpha) * sys.W
    if sign == "-":
        return 0.5j * sys.Gamma0(alpha) * sys.W
    raise ValueError("sign must be '+' or '-'")