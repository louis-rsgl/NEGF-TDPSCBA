# NEGF-TDPSCBA

Numerical implementation of a **nonequilibrium Green’s function (NEGF)** solver for transient transport in an interacting quantum system with **electron–phonon coupling**, solved within the **self-consistent Born approximation (SCBA)**.

The code computes **time-dependent currents** in a system coupled to multiple leads and a phonon bath.

The nonequilibrium Green functions are obtained by solving a **self-consistent Dyson–Keldysh system** and are then used to evaluate observables such as transient currents.

---

# Physical Model

The system consists of

* a **single electronic level**
* **two leads** (L/R)
* a **phonon mode**
* electron–phonon coupling

Transport is described using the **nonequilibrium Green function formalism**.

The central quantities are the nonequilibrium Green functions

$$
\bar G^R(\omega), \qquad \bar G^<(\omega)
$$

satisfying the coupled equations

### Retarded Green function

$$
\bar G^R(\omega) =
\left[
\omega-\varepsilon_0-\Delta-\bar V^R(\omega)
\right]^{-1}
$$

---

### Lesser Green function

$$
\bar G^<(\omega)
=
\bar G^R(\omega),\Sigma^<(\omega),\bar G^A(\omega)
$$

$$
\bar G^A = (\bar G^R)^*
$$

---

### Lesser self-energy

$$
\Sigma^<(\omega)
=
i\sum_\beta
f_\beta(\omega),
\Gamma_\beta^0
\frac{W}{\omega^2+W^2}
+
g_q^2
\left[
\bar G^<(\omega-\omega_q),f_{\rm ph}(\omega_q)
+
\bar G^<(\omega+\omega_q),(f_{\rm ph}(\omega_q)+1)
\right]
$$

---

### Retarded interaction kernel

$$
\bar V^R(\omega)
=

\Bigg(
\int \frac{d\Omega}{i\pi}
g_q^2,\bar G^<(\omega-\Omega)
\left(
\frac{1}{\Omega-\omega_q+i\eta}
-

\frac{1}{\Omega+\omega_q+i\eta}
\right)
+\dots
\Bigg)
$$

The equations are solved self-consistently on a frequency grid.

---

# Numerical Approach

The SCBA solver performs a **fixed-point iteration**:

1. Start with equilibrium (G^R_0)
2. Construct initial (G^<_0)
3. Iterate

$$
G^R_{n+1} = [\omega - \varepsilon_0 - \Delta - V^R(G_n)]^{-1}
$$

$$
G^<*{n+1} = G^R*{n+1} \Sigma^<(G_n) G^A_{n+1}
$$

A **linear mixing scheme** is used for stability

$$
X_{n+1} = (1-\alpha) X_n + \alpha X_{\text{trial}}
$$

with $X = (G^R, G^<)$.

The converged Green functions are cached inside the `System` object and can be reused during observable calculations.

---

# Project Structure

```
backend/
│
├── system_classes.py
│       System definition and physical parameters
│
├── SCBA.py
│       Self-consistent nonequilibrium solver
│
├── green_function.py
│       Equilibrium Green functions and pole expansion
│
├── distribution.py
│       Fermi and Bose distribution functions
│
├── observables.py
│       Physical observables (currents, kernels, etc.)
│
runner.py
    Interactive runner and parameter scan
```

---

# Key Components

## System

Defines the physical system and solver configuration.

Example:

```python
sys = System(
    W=5.0,
    g_q=0.2,
    w_q=0.2,
    e_0=0.0,
)
```

Lead parameters:

```
LeadParams(
    Gamma0
    Delta
    beta
    mu
)
```

---

## Nonequilibrium Solver

The SCBA solver computes

```
GR_noneq(ω)
Gless_noneq(ω)
```

Usage:

```python
sys.solve_noneq()

GR = sys.GR_noneq(0.5)
Gless = sys.Gless_noneq(-0.3)
```

The solver result is cached to avoid recomputation.

---

## Observables

Physical observables are computed using the nonequilibrium Green functions.

Example: transient current

```python
t, I = current_alpha(
    sys,
    alpha="L",
    t_max=2.0,
    n_t=2000
)
```

---

# Running the Code

The project includes an interactive runner that explores parameter space.

Run:

```
python runner.py
```

The runner

* precomputes currents over a grid of parameters
* provides interactive sliders for

  * phonon bandwidth (W)
  * coupling (g_q)

---

# Example Output

The runner produces plots of

$$
J_\alpha(t)
$$

showing the real and imaginary parts of the transient current.

---

# Dependencies

Python ≥ 3.10 recommended.

Required libraries

```
numpy
scipy
matplotlib
```

Install via

```
pip install numpy scipy matplotlib
```

---

# Performance Notes

The solver involves

* nested frequency integrals
* self-consistent iterations
* time-dependent integrals

Typical runtime scales with

```
O(N_iter × N_ω × N_quad)
```

For exploratory work a **fake solver** can be enabled in `runner.py`:

```
USE_FAKE_SOLVER = True
```

This generates synthetic currents for testing visualization.

---

# Future Improvements

Possible numerical improvements include

* Anderson / Pulay mixing
* Broyden quasi-Newton updates
* caching repeated integrals
* FFT-based convolution for phonon kernels
* parallel parameter sweeps

---

# Author

Research code developed for studies of

**time-dependent quantum transport with electron–phonon coupling** using NEGF and SCBA.
