from __future__ import annotations

import numpy as np
import scipy.constants as const


HBAR_EV_S = const.hbar / const.e  # [eV*s]


def gamma_to_time_unit_s(gamma_eV: float) -> float:
    """
    Natural time unit corresponding to dimensionless t=1:
        t_phys = t * (ħ / Γ)
    with Γ given in eV.
    """
    return HBAR_EV_S / gamma_eV


def gamma_to_current_unit_A(gamma_eV: float) -> float:
    """
    Natural current unit corresponding to dimensionless I=1:
        I_phys = I * (e Γ / ħ)
    with Γ given in eV.
    """
    return const.e * gamma_eV / HBAR_EV_S


def time_to_si_seconds(t_dimless, gamma_eV: float):
    return np.asarray(t_dimless) * gamma_to_time_unit_s(gamma_eV)


def time_to_ps(t_dimless, gamma_eV: float):
    return 1e12 * time_to_si_seconds(t_dimless, gamma_eV)


def current_to_si_ampere(I_dimless, gamma_eV: float):
    return np.asarray(I_dimless) * gamma_to_current_unit_A(gamma_eV)


def current_to_uA(I_dimless, gamma_eV: float):
    return 1e6 * current_to_si_ampere(I_dimless, gamma_eV)