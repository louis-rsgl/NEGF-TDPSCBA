# NEGF-TDDPSCBA

Numerical implementation of a **nonequilibrium Green’s function (NEGF)** solver for transient quantum transport with **electron–phonon interactions**, computed within the **self-consistent Born approximation (SCBA)**.

The code computes **time-dependent currents** in a quantum system coupled to multiple leads and a phonon bath.

Nonequilibrium Green functions are obtained by solving a **self-consistent Dyson–Keldysh system**, and observables such as transient currents are evaluated from the converged solution.

---

# Physical Model

The system consists of

* a **single electronic level**
* **two leads** (L/R)
* a **phonon mode**
* **electron–phonon coupling**

Transport is described using the **nonequilibrium Green function formalism**.

The key quantities are the nonequilibrium Green functions

```
G^R(ω),  G^<(ω)
```

---

## Retarded Green Function

```
G^R(ω) = [ ω − ε₀ − Δ − V^R(ω) ]⁻¹
```

---

## Lesser Green Function

```
G^<(ω) = G^R(ω) Σ^<(ω) G^A(ω)

G^A(ω) = (G^R(ω))*
```

---

## Lesser Self-Energy

```
Σ^<(ω) =
i ∑_β f_β(ω) Γ_β^0 W / (ω² + W²)
+
g_q² [
    G^<(ω − ω_q) f_ph(ω_q)
    +
    G^<(ω + ω_q) (f_ph(ω_q) + 1)
]
```

---

## Retarded Kernel

```
V^R(ω) =
(
∫ dΩ/(iπ) g_q² G^<(ω − Ω)
[
1/(Ω − ω_q + iη)
−
1/(Ω + ω_q + iη)
]
+ ...
)
```

These equations are solved self-consistently on a frequency grid.

---

# Numerical Method

The SCBA solver performs a **fixed-point iteration**.

Initial guess:

```
G^R₀(ω) = equilibrium Green function
G^<_0(ω) = −2i Im[G^R₀(ω)] f(ω)
```

Iteration:

```
G^R_{n+1}(ω) =
[ ω − ε₀ − Δ − V^R(G_n) ]⁻¹

G^<_{n+1}(ω) =
G^R_{n+1} Σ^<(G_n) G^A_{n+1}
```

To stabilize convergence, **linear mixing** is used:

```
X_{n+1} = (1 − α) X_n + α X_trial
```

where

```
X = (G^R, G^<)
```

The converged Green functions are cached inside the `System` object and reused during observable calculations.

---

# Project Structure

```
backend/
│
├── system_classes.py
│       System definition and parameters
│
├── SCBA.py
│       Self-consistent nonequilibrium solver
│
├── green_function.py
│       Equilibrium Green functions and pole expansion
│
├── distribution.py
│       Fermi and Bose distributions
│
├── observables.py
│       Physical observables (currents, kernels)
│
runner.py
    Interactive parameter exploration
```

---

# System Definition

A system is defined through the `System` class.

Example:

```python
sys = System(
    W=5.0,
    g_q=0.2,
    w_q=0.2,
    e_0=0.0
)
```

Lead parameters:

```python
LeadParams(
    Gamma0,
    Delta,
    beta,
    mu
)
```

---

# Solving the Nonequilibrium Problem

Compute the nonequilibrium Green functions:

```python
sys.solve_noneq()

GR = sys.GR_noneq(0.5)
Gless = sys.Gless_noneq(-0.3)
```

The solver result is cached automatically.

---

# Computing Observables

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

# Interactive Runner

The project includes a runner for exploring parameter space.

Run

```
python runner.py
```

The runner

* precomputes currents over a grid
* provides interactive sliders for

  * phonon bandwidth `W`
  * coupling strength `g_q`

---

# Example Plot

The runner displays the transient current

```
J_α(t)
```

showing

* real part
* imaginary part

as a function of time.

---

# Dependencies

Python ≥ 3.10 recommended.

Required libraries

```
numpy
scipy
matplotlib
```

Install with

```
pip install numpy scipy matplotlib
```

---

# Performance Notes

The solver involves

* nested frequency integrals
* self-consistent iterations
* time-dependent quadratures

Runtime roughly scales as

```
O(N_iter × N_ω × N_quad)
```

For testing and visualization, a **fake solver** can be enabled:

```python
USE_FAKE_SOLVER = True
```

in `runner.py`.

---

# Possible Improvements

Potential future improvements include

* Anderson / Pulay mixing
* Broyden quasi-Newton updates
* caching repeated integrals
* FFT-based convolution kernels
* parallel parameter scans

---

# Author

Research code for studying

**time-dependent quantum transport with electron–phonon coupling** using NEGF and SCBA.