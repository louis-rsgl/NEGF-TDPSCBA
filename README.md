# NEGF-TDPSCBA

**NEGF-TDPSCBA** implements the **time-dependent nonequilibrium Green's function (NEGF) transport** in nanoscale systems with **finite lead bandwidth** including **electron-phonon scattering** within the **self-consistent Born approximation (SCBA)**, extending the exact time-dependent transport theory of

**Maciejko, Wang, and Guo**
*Time-dependent quantum transport far from equilibrium: An exact nonlinear response theory*
Phys. Rev. B **74**, 085324 (2006) [![DOI](https://img.shields.io/badge/DOI-10.1103/PhysRevB.74.085324-blue)](https://doi.org/10.1103/PhysRevB.74.085324).

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

## Hamiltonian

The system Hamiltonian is

$$H = H_{\text{leads}} + H_{\text{device}} + H_{\text{coupling}} + H_{\text{e-ph}}$$

### Leads

$$H_{\text{leads}} = \sum_{k\alpha} \epsilon_{k\alpha} c^\dagger_{k\alpha} c_{k\alpha}$$

where $\alpha = L, R$.

### Device (single electronic level)

$$H_{\text{device}} = \epsilon_0, d^\dagger d$$


### Lead–Device Coupling

$$H_{\text{coupling}} = \sum_{k\alpha} \left( t_{k\alpha} c^\dagger_{k\alpha} d + t_{k\alpha}^* d^\dagger c_{k\alpha} \right)$$

The coupling produces an **energy-dependent linewidth**

$$\Gamma_\alpha(\omega) = \Gamma_\alpha^0 \frac{W}{\omega^2 + W^2}$$

where

* $W$ is the **lead bandwidth**
* finite bandwidth effects appear explicitly in the transient dynamics.

### Phonon Mode

$$H_{\text{ph}} = \omega_q b^\dagger b$$


### Electron–Phonon Interaction

$$H_{\text{e-ph}} = g_q, d^\dagger d (b + b^\dagger)$$

where

* $g_q$ is the electron-phonon coupling
* $\omega_q$ is the phonon frequency.

Electron–phonon scattering is evaluated within **SCBA**.

# Nonequilibrium Green Functions

The transport problem is formulated in terms of

* retarded Green function

$$G^R(\omega)$$

* lesser Green function

$$G^<(\omega)$$

## Retarded Green Function

$$G^R(\omega) = \left[\omega - \epsilon_0 - \Sigma^R_{\text{leads}}(\omega) - \Sigma^R_{\text{ph}}(\omega)\right]^{-1}$$

## Lesser Green Function

$$G^<(\omega) = G^R(\omega)\Sigma^<(\omega)G^A(\omega)$$

## Lead Self-Energy

$$\Sigma^< {\text{leads}}(\omega) = i\sum_\alpha f_\alpha(\omega)\Gamma_\alpha(\omega)$$

where $f_\alpha(\omega)$ are the Fermi distributions of the leads.

# Numerical Method

The nonequilibrium Green functions are obtained through a **self-consistent SCBA iteration**.

Initial guess

$$G^R_0(\omega) = G^R_{\text{eq}}(\omega)$$

$$G^<_0(\omega) = -2i,\text{Im}[G^R_0(\omega)]f(\omega)$$

## Iteration

At iteration $n$

$$G^R_{n+1}(\omega) = [\omega - \epsilon_0 - \Sigma^R(G_n)]^{-1}$$

$$G^<*{n+1}(\omega) = G^R*{n+1}\Sigma^<(G_n)G^A_{n+1}$$

## Mixing

To stabilize convergence the solver uses **linear mixing**

$$X_{n+1} = (1-\alpha)X_n + \alpha X_{\text{trial}}$$

with

$$X = (G^R, G^<)$$

# Transient Current

After convergence the **time-dependent current**

$$J_\alpha(t)$$

is evaluated using the transient transport formalism derived from the original theory but with the phonons.

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

Python 3.13

Required packages

```
numpy
scipy
simpy
matplotlib
tqdm
```

Install with

```
pip install numpy scipy matplotlib tqdm simpy
```

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

If you use **NEGF-TDPSCBA** in your research, please cite the repository:

```
Rossignol, Louis and Guo, Hong.
NEGF-TDPSCBA: Time-Dependent Quantum Transport far from equilibrium with Electron-Phonon SCBA in Finite Band Limit.
GitHub repository.
```

Repository:

```
https://github.com/louis-rsgl/NEGF-TDPSCBA
```

---

## BibTeX

```bibtex
@software{rossignol_negf_tdpscba,
  author = {Rossignol, Louis and Guo, Hong},
  title = {NEGF-TDPSCBA: Time-Dependent far from equilibrium Quantum Transport beyond wide-band limit with Electron-Phonon SCBA},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/louis-rsgl/NEGF-TDPSCBA}
}
```