"""
Pantheon Supernova MCMC Cosmological Parameter Estimation
==========================================================
Uses the Metropolis-Hastings algorithm to fit cosmological parameters
(Hubble constant H0, matter density Omega_m) to Pantheon supernova data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Cosmological model: flat ΛCDM
# ---------------------------------------------------------------------------

def luminosity_distance(z, H0, Omega_m):
    """
    Compute the luminosity distance d_L (in Mpc) for a flat ΛCDM universe.

    Parameters
    ----------
    z : float
        Redshift.
    H0 : float
        Hubble constant in km/s/Mpc.
    Omega_m : float
        Matter density parameter (dark energy = 1 - Omega_m for flat universe).

    Returns
    -------
    float
        Luminosity distance in Mpc.
    """
    c = 299792.458  # speed of light in km/s
    Omega_lambda = 1.0 - Omega_m

    def integrand(z_prime):
        return 1.0 / np.sqrt(Omega_m * (1.0 + z_prime) ** 3 + Omega_lambda)

    comoving_distance, _ = quad(integrand, 0, z)
    comoving_distance *= c / H0

    return (1.0 + z) * comoving_distance


def distance_modulus(z, H0, Omega_m):
    """
    Compute the theoretical distance modulus mu = 5 log10(d_L / 10 pc).

    Parameters
    ----------
    z : float or array-like
        Redshift(s).
    H0 : float
        Hubble constant in km/s/Mpc.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float or np.ndarray
        Distance modulus (magnitudes).
    """
    if np.isscalar(z):
        dL = luminosity_distance(z, H0, Omega_m)
        return 5.0 * np.log10(dL * 1e6 / 10.0)  # Mpc → pc → modulus

    return np.array([
        5.0 * np.log10(luminosity_distance(zi, H0, Omega_m) * 1e6 / 10.0)
        for zi in z
    ])


# ---------------------------------------------------------------------------
# Likelihood / posterior
# ---------------------------------------------------------------------------

def log_likelihood(params, z_data, mu_data, sigma_data):
    """
    Gaussian log-likelihood for distance modulus data.

    Parameters
    ----------
    params : array-like [H0, Omega_m]
        Current parameter values.
    z_data : np.ndarray
        Observed redshifts.
    mu_data : np.ndarray
        Observed distance moduli.
    sigma_data : np.ndarray
        Uncertainties on distance moduli.

    Returns
    -------
    float
        Log-likelihood value.
    """
    H0, Omega_m = params
    mu_theory = distance_modulus(z_data, H0, Omega_m)
    residuals = mu_data - mu_theory
    return -0.5 * np.sum((residuals / sigma_data) ** 2)


def log_prior(params):
    """
    Uniform log-prior over physically reasonable parameter ranges.

    Parameters
    ----------
    params : array-like [H0, Omega_m]

    Returns
    -------
    float
        0 if within prior range, -inf otherwise.
    """
    H0, Omega_m = params
    if 50.0 < H0 < 100.0 and 0.0 < Omega_m < 1.0:
        return 0.0
    return -np.inf


def log_posterior(params, z_data, mu_data, sigma_data):
    """
    Log-posterior = log-prior + log-likelihood.
    """
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, z_data, mu_data, sigma_data)


# ---------------------------------------------------------------------------
# Metropolis-Hastings MCMC sampler
# ---------------------------------------------------------------------------

def metropolis_hastings(log_posterior_fn, initial_params, proposal_widths,
                        n_steps, data, random_seed=42):
    """
    Metropolis-Hastings MCMC sampler.

    Parameters
    ----------
    log_posterior_fn : callable
        Function returning log-posterior given (params, *data).
    initial_params : array-like
        Starting point in parameter space.
    proposal_widths : array-like
        Standard deviations of the Gaussian proposal distribution for each
        parameter.
    n_steps : int
        Number of MCMC steps.
    data : tuple
        Additional arguments forwarded to log_posterior_fn after params.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    chain : np.ndarray, shape (n_steps, n_params)
        Full MCMC chain (including burn-in).
    acceptance_rate : float
        Fraction of proposed steps that were accepted.
    """
    rng = np.random.default_rng(random_seed)
    params = np.array(initial_params, dtype=float)
    n_params = len(params)
    chain = np.empty((n_steps, n_params))
    n_accepted = 0

    current_log_post = log_posterior_fn(params, *data)

    for i in range(n_steps):
        # Gaussian proposal
        proposed = params + rng.normal(0.0, proposal_widths, size=n_params)
        proposed_log_post = log_posterior_fn(proposed, *data)

        # Metropolis acceptance criterion
        log_alpha = proposed_log_post - current_log_post
        if np.log(rng.uniform()) < log_alpha:
            params = proposed
            current_log_post = proposed_log_post
            n_accepted += 1

        chain[i] = params

    acceptance_rate = n_accepted / n_steps
    return chain, acceptance_rate


# ---------------------------------------------------------------------------
# Synthetic Pantheon-like data (used when no data file is provided)
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_sne=100, H0_true=70.0, Omega_m_true=0.3,
                             sigma=0.15, random_seed=0):
    """
    Generate a synthetic Pantheon-like supernova dataset.

    Parameters
    ----------
    n_sne : int
        Number of supernovae.
    H0_true : float
        True Hubble constant used to generate the data.
    Omega_m_true : float
        True matter density used to generate the data.
    sigma : float
        Intrinsic + measurement scatter in magnitudes.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    z : np.ndarray
        Redshifts.
    mu : np.ndarray
        Noisy distance moduli.
    sigma_arr : np.ndarray
        Per-SN uncertainties (all equal to sigma).
    """
    rng = np.random.default_rng(random_seed)
    z = np.sort(rng.uniform(0.01, 1.5, size=n_sne))
    mu_true = distance_modulus(z, H0_true, Omega_m_true)
    mu_obs = mu_true + rng.normal(0.0, sigma, size=n_sne)
    sigma_arr = np.full(n_sne, sigma)
    return z, mu_obs, sigma_arr


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_hubble_diagram(z_data, mu_data, sigma_data,
                        H0_fit, Omega_m_fit, filename="hubble_diagram.png"):
    """
    Plot the Hubble diagram (distance modulus vs redshift).
    """
    z_theory = np.linspace(z_data.min(), z_data.max(), 300)
    mu_theory = distance_modulus(z_theory, H0_fit, Omega_m_fit)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(z_data, mu_data, yerr=sigma_data, fmt="o", color="steelblue",
                alpha=0.6, markersize=4, label="Pantheon SNe Ia")
    ax.plot(z_theory, mu_theory, "r-", linewidth=2,
            label=f"Best fit: $H_0={H0_fit:.1f}$, $\\Omega_m={Omega_m_fit:.3f}$")
    ax.set_xlabel("Redshift $z$", fontsize=13)
    ax.set_ylabel("Distance modulus $\\mu$", fontsize=13)
    ax.set_title("Hubble Diagram – Pantheon SNe Ia", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved Hubble diagram → {filename}")


def plot_mcmc_chains(chain, burn_in, param_names, filename="mcmc_chains.png"):
    """
    Plot MCMC chain traces.
    """
    n_params = chain.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(chain[:, i], color="steelblue", alpha=0.7, linewidth=0.5)
        ax.axvline(burn_in, color="red", linestyle="--", label="End of burn-in")
        ax.set_ylabel(name, fontsize=12)
        if i == 0:
            ax.legend(fontsize=10)

    axes[-1].set_xlabel("Step", fontsize=12)
    fig.suptitle("MCMC Chain Traces", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved chain traces → {filename}")


def plot_corner(chain_post_burnin, param_names, filename="corner_plot.png"):
    """
    Simple corner plot (marginalised 1-D and 2-D posteriors).
    """
    n_params = chain_post_burnin.shape[1]
    fig, axes = plt.subplots(n_params, n_params,
                             figsize=(4 * n_params, 4 * n_params))

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                ax.hist(chain_post_burnin[:, i], bins=40,
                        color="steelblue", density=True)
                ax.set_ylabel("P", fontsize=10)
            elif i > j:
                ax.scatter(chain_post_burnin[:, j], chain_post_burnin[:, i],
                           s=0.5, alpha=0.2, color="steelblue")
                ax.set_ylabel(param_names[i], fontsize=10)
                ax.set_xlabel(param_names[j], fontsize=10)
            else:
                ax.axis("off")

            if j == 0 and i != j:
                ax.set_ylabel(param_names[i], fontsize=10)
            if i == n_params - 1 and i != j:
                ax.set_xlabel(param_names[j], fontsize=10)

    fig.suptitle("Posterior Corner Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved corner plot → {filename}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_mcmc(data_file=None, n_steps=50000, burn_in=10000):
    """
    Run the full MCMC pipeline.

    Parameters
    ----------
    data_file : str or None
        Path to a whitespace-delimited data file with columns
        ``z  mu  sigma_mu``.  If None, synthetic data are used.
    n_steps : int
        Total number of MCMC steps (including burn-in).
    burn_in : int
        Number of initial steps discarded as burn-in.

    Returns
    -------
    dict
        Dictionary with keys 'H0', 'Omega_m', 'chain', 'acceptance_rate'.
    """
    # ------------------------------------------------------------------
    # Load or generate data
    # ------------------------------------------------------------------
    if data_file is not None:
        print(f"Loading data from {data_file} …")
        data_arr = np.loadtxt(data_file)
        z_data, mu_data, sigma_data = data_arr[:, 0], data_arr[:, 1], data_arr[:, 2]
    else:
        print("No data file supplied – using synthetic Pantheon-like data.")
        z_data, mu_data, sigma_data = generate_synthetic_data()

    print(f"Dataset: {len(z_data)} supernovae, "
          f"z ∈ [{z_data.min():.3f}, {z_data.max():.3f}]")

    # ------------------------------------------------------------------
    # MCMC setup
    # ------------------------------------------------------------------
    initial_params = [70.0, 0.3]          # [H0, Omega_m]
    proposal_widths = [0.5, 0.01]         # Gaussian step sizes
    param_names = ["$H_0$ [km/s/Mpc]", "$\\Omega_m$"]

    print(f"\nRunning Metropolis-Hastings MCMC: {n_steps} steps …")
    chain, acceptance_rate = metropolis_hastings(
        log_posterior,
        initial_params,
        proposal_widths,
        n_steps,
        data=(z_data, mu_data, sigma_data),
    )
    print(f"Acceptance rate: {acceptance_rate * 100:.1f}%")

    # ------------------------------------------------------------------
    # Posterior statistics
    # ------------------------------------------------------------------
    chain_post_burnin = chain[burn_in:]
    H0_median = np.median(chain_post_burnin[:, 0])
    H0_std = np.std(chain_post_burnin[:, 0])
    Om_median = np.median(chain_post_burnin[:, 1])
    Om_std = np.std(chain_post_burnin[:, 1])

    print("\n=== Posterior estimates ===")
    print(f"  H0       = {H0_median:.2f} ± {H0_std:.2f} km/s/Mpc")
    print(f"  Omega_m  = {Om_median:.4f} ± {Om_std:.4f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_hubble_diagram(z_data, mu_data, sigma_data, H0_median, Om_median)
    plot_mcmc_chains(chain, burn_in, param_names)
    plot_corner(chain_post_burnin, param_names)

    return {
        "H0": H0_median,
        "H0_std": H0_std,
        "Omega_m": Om_median,
        "Omega_m_std": Om_std,
        "chain": chain,
        "acceptance_rate": acceptance_rate,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MCMC cosmological parameter estimation using Pantheon SNe Ia data"
    )
    parser.add_argument("--data", metavar="FILE", default=None,
                        help="Path to data file (z  mu  sigma_mu per row). "
                             "Defaults to synthetic data.")
    parser.add_argument("--steps", type=int, default=50000,
                        help="Number of MCMC steps (default: 50000).")
    parser.add_argument("--burnin", type=int, default=10000,
                        help="Burn-in steps to discard (default: 10000).")
    args = parser.parse_args()

    run_mcmc(data_file=args.data, n_steps=args.steps, burn_in=args.burnin)
