# NEGF-TDPSCBA

[![DOI](https://img.shields.io/badge/DOI-10.1103/PhysRevB.74.085324-blue)](https://doi.org/10.1103/PhysRevB.74.085324)

**NEGF-TDPSCBA** is a numerical implementation of **time-dependent nonequilibrium Green's function (NEGF) transport** including **electron-phonon scattering** within the **self-consistent Born approximation (SCBA)**.

The code computes **transient currents** in nanoscale systems with **finite lead bandwidth**, extending the exact time-dependent transport theory of

**Maciejko, Wang, and Guo**
*Time-dependent quantum transport far from equilibrium: An exact nonlinear response theory*
Phys. Rev. B **74**, 085324 (2006).

Unlike most transient transport approaches, this framework **does not rely on the wide-band limit**, allowing the study of **finite-bandwidth effects in time-dependent transport**.

---

# Physical Model

The system is a **lead–device–lead (LDL)** configuration:

```
Lead L  ──  Device  ──  Lead R
```

The scattering region contains

* a **single electronic level**
* a **single phonon mode**
* coupling to two leads with **finite bandwidth**

Transport is treated using the **nonequilibrium Green function formalism**.

---

# Hamiltonian

The system Hamiltonian is

$$H = H_{\text{leads}} + H_{\text{device}} + H_{\text{coupling}} + H_{\text{e-ph}}$$

---

## Leads

$$H_{\text{leads}} = \sum_{k\alpha} \epsilon_{k\alpha} c^\dagger_{k\alpha} c_{k\alpha}$$

where $\alpha = L, R$.

---

## Device (single electronic level)

$$H_{\text{device}} = \epsilon_0, d^\dagger d$$

---

## Lead–Device Coupling

$$H_{\text{coupling}} = \sum_{k\alpha} \left( t_{k\alpha} c^\dagger_{k\alpha} d + t_{k\alpha}^* d^\dagger c_{k\alpha} \right)$$

The coupling produces an **energy-dependent linewidth**

$$\Gamma_\alpha(\omega) = \Gamma_\alpha^0 \frac{W}{\omega^2 + W^2}$$

where

* $W$ is the **lead bandwidth**
* finite bandwidth effects appear explicitly in the transient dynamics.

---

## Phonon Mode

$$H_{\text{ph}} = \omega_q b^\dagger b$$

---

## Electron–Phonon Interaction

$$H_{\text{e-ph}} = g_q, d^\dagger d (b + b^\dagger)$$

where

* $g_q$ is the electron-phonon coupling
* $\omega_q$ is the phonon frequency.

Electron–phonon scattering is evaluated within **SCBA**.

---

# Nonequilibrium Green Functions

The transport problem is formulated in terms of

* retarded Green function

$$G^R(\omega)$$

* lesser Green function

$$G^<(\omega)$$

---

## Retarded Green Function

$$G^R(\omega) = \left[\omega - \epsilon_0 - \Sigma^R_{\text{leads}}(\omega) - \Sigma^R_{\text{ph}}(\omega)\right]^{-1}$$

---

## Lesser Green Function

$$G^<(\omega) = G^R(\omega)\Sigma^<(\omega)G^A(\omega)$$

---

## Lead Self-Energy

$$\Sigma^<*{\text{leads}}(\omega) = i\sum*\alpha f_\alpha(\omega)\Gamma_\alpha(\omega)$$

where $f_\alpha(\omega)$ are the Fermi distributions of the leads.

---

# Numerical Method

The nonequilibrium Green functions are obtained through a **self-consistent SCBA iteration**.

Initial guess

$$G^R_0(\omega) = G^R_{\text{eq}}(\omega)$$

$$G^<_0(\omega) = -2i,\text{Im}[G^R_0(\omega)]f(\omega)$$

---

## Iteration

At iteration $n$

$$G^R_{n+1}(\omega) = [\omega - \epsilon_0 - \Sigma^R(G_n)]^{-1}$$

$$G^<*{n+1}(\omega) = G^R*{n+1}\Sigma^<(G_n)G^A_{n+1}$$

---

## Mixing

To stabilize convergence the solver uses **linear mixing**

$$X_{n+1} = (1-\alpha)X_n + \alpha X_{\text{trial}}$$

with

$$X = (G^R, G^<)$$

---

# Transient Current

After convergence the **time-dependent current**

$$J_\alpha(t)$$

is evaluated using the transient transport formalism derived in the original theory.

The current depends on

$$A_\alpha(\omega,t), \qquad \Phi_\alpha(\omega,t)$$

which are obtained from the converged Green functions.

---

# Project Structure

```
backend/

system_classes.py
    System parameters and configuration

SCBA.py
    Self-consistent NEGF solver

green_function.py
    Equilibrium Green functions and pole expansion

distribution.py
    Fermi and Bose distributions

observables.py
    Current evaluation and kernels

runner.py
    Interactive simulation runner
```

---

# Example System Definition

```python
sys = System(
    W=5.0,
    g_q=0.2,
    w_q=0.2,
    e_0=0.0
)
```

Lead parameters

```python
LeadParams(
    Gamma0,
    Delta,
    beta,
    mu
)
```

---

# Running the Simulation

Run

```
python runner.py
```

The runner

* solves the nonequilibrium SCBA problem
* computes transient currents
* allows interactive parameter exploration.

---

# Dependencies

Python ≥ 3.10

Required packages

```
numpy
scipy
matplotlib
tqdm
```

Install with

```
pip install numpy scipy matplotlib tqdm
```

---

# Performance Notes

The runtime scales roughly as

$$O(N_{\text{iter}} \times N_\omega \times N_{\text{quad}})$$

where

* $N_\omega$ is the frequency grid
* $N_{\text{iter}}$ the SCBA iterations
* $N_{\text{quad}}$ quadrature operations.

---

# Possible Improvements

Potential future developments

* Pulay / Anderson mixing
* Broyden quasi-Newton updates
* FFT-based convolution acceleration
* pole-expansion techniques
* parallel parameter sweeps

---

# Author

Louis Rossignol and Hong Guo
McGill University

Research code for studying

**time-dependent quantum transport with electron–phonon scattering using NEGF-SCBA beyond the wide-band limit.**

---

# Citation

If you use this code in research, please cite

Maciejko, J., Wang, J., & Guo, H.
*Time-dependent quantum transport far from equilibrium: An exact nonlinear response theory*
Physical Review B **74**, 085324 (2006).

DOI:

[https://doi.org/10.1103/PhysRevB.74.085324](https://doi.org/10.1103/PhysRevB.74.085324)

BibTeX:

```bibtex
@article{Maciejko2006,
  author = {Maciejko, Joseph and Wang, Jian and Guo, Hong},
  title = {Time-dependent quantum transport far from equilibrium: An exact nonlinear response theory},
  journal = {Physical Review B},
  volume = {74},
  pages = {085324},
  year = {2006},
  doi = {10.1103/PhysRevB.74.085324}
}
```