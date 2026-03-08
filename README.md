# Computer-Modelling-Pantheon

Computer modelling of Pantheon supernova data using Markov Chain Monte Carlo
simulations (Metropolis-Hastings algorithm).  Fits cosmological parameters
(Hubble constant *H₀* and matter density *Ωₘ*) from Type Ia supernova distance
moduli to model the expansion of the universe.

## Overview

The code implements a flat ΛCDM (Lambda Cold Dark Matter) cosmological model
and uses the Metropolis-Hastings MCMC algorithm to sample the posterior
distribution of:

| Parameter | Symbol | Typical value |
|-----------|--------|---------------|
| Hubble constant | *H₀* | ~70 km/s/Mpc |
| Matter density | *Ωₘ* | ~0.3 |

## Requirements

```
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`.

## Usage

### With synthetic data (no data file needed)

```bash
python pantheon_mcmc.py
```

### With your own data file

The data file must be whitespace-delimited with three columns:
`z   mu   sigma_mu`

```bash
python pantheon_mcmc.py --data pantheon_data.txt
```

### Options

```
--data FILE    Path to SNe Ia data file (default: synthetic data)
--steps N      Total MCMC steps (default: 50000)
--burnin N     Burn-in steps to discard (default: 10000)
```

## Output

The script produces three plots:

* **hubble_diagram.png** – observed distance moduli with the best-fit model curve
* **mcmc_chains.png** – chain traces for each parameter
* **corner_plot.png** – marginalised 1-D and 2-D posterior distributions

And prints posterior estimates, e.g.:

```
=== Posterior estimates ===
  H0       = 70.02 ± 0.84 km/s/Mpc
  Omega_m  = 0.2998 ± 0.0121
```

## Method

1. **Cosmological model** – luminosity distances are computed by numerically
   integrating the Friedmann equation for a flat ΛCDM universe.
2. **Likelihood** – Gaussian likelihood on distance moduli residuals.
3. **Prior** – uniform prior: 50 < *H₀* < 100 km/s/Mpc, 0 < *Ωₘ* < 1.
4. **Sampler** – Metropolis-Hastings with Gaussian proposal distribution.
   Aim for an acceptance rate of 20–50 % by tuning `proposal_widths`.
