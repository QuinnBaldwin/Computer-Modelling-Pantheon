import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Cosmology:
    def __init__(self, H0, Omega_m, Omega_lambda, Omega_k):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_k = Omega_k
        self.c = 299792.458  # Speed of light in km/s

    def integrand(self, z_prime):
        return 1.0 / ((self.Omega_m * (1 + z_prime)**3 + self.Omega_k * (1 + z_prime)**2 + self.Omega_lambda) ** 0.5)
    
    def flat_universe(self):
        return abs(self.Omega_k) < 1e-5
    
    def setomega_m(self, new_Omega_m):
        self.Omega_m = new_Omega_m
        self.Omega_lambda = 1.0 - self.Omega_m - self.Omega_k
        return self.Omega_lambda
    
    def setomega_lambda(self, new_Omega_lambda):
        self.Omega_lambda = new_Omega_lambda
        self.Omega_m = 1.0 - self.Omega_lambda - self.Omega_k
        return self.Omega_m
   
    def omega_m_h2(self):
        h = self.H0 / 100.0
        return self.Omega_m * h**2
    
    def __str__(self):
        Omega_k = 1.0 - self.Omega_m - self.Omega_lambda
        return f"<Cosmology with H0={self.H0}, Omega_m={self.Omega_m}, Omega_lambda={self.Omega_lambda}, Omega_k={Omega_k}>"
        ## The __str__ method returns a string representation of the Cosmology object, showing its current parameters. This makes the code easier to read and makes debugging easier.
    
    def rectangle_rule(self, z_max, num_intervals):
        dz = z_max / num_intervals
        integral = 0.0
        for i in range(num_intervals):
            z_i = i * dz
            integral += self.integrand(z_i) 
        integral *= dz
        d_z_rectangle = integral * (self.c / self.H0)  # Convert to Mpc (c is in km/s and H0 in km/s/Mpc)
        return d_z_rectangle
   
    def trapezoidal_rule(self, z_max, num_intervals):
        dz = z_max / num_intervals
        integral = 0.5 * (self.integrand(0) + self.integrand(z_max))
        for i in range(1, num_intervals): # we have stared from 1 as the first and last terms are already included
            z_i = i * dz
            integral += self.integrand(z_i) 
        integral *= dz
        d_z_trapezoidal = integral * (self.c / self.H0)  # Convert to Mpc (c is in km/s and H0 in km/s/Mpc)
        return d_z_trapezoidal
   
    def simpsons_rule(self, z_max, num_intervals):
        if num_intervals % 2 == 1:
            raise ValueError("Number of intervals must be even for Simpson's rule.")
        dz = z_max / num_intervals
        integral = self.integrand(0) + self.integrand(z_max)
        for i in range(1, num_intervals, 2):  # odd indices
            z_i = i * dz
            integral += 4 * self.integrand(z_i)
        for i in range(2, num_intervals-1, 2):  # even indices
            z_i = i * dz
            integral += 2 * self.integrand(z_i)
        integral *= dz / 3
        d_z_simpsons = integral * (self.c / self.H0)  # Convert to Mpc (c is in km/s and H0 in km/s/Mpc)
        return d_z_simpsons
   
    def scipy_integral(self, z_max):
        from scipy import integrate  # use scipy to find the accurate value of the integral for comparison
        def integrand_scipy(z):
            return 1.0 / ((self.Omega_m * (1 + z)**3 +
                            self.Omega_k * (1 + z)**2 +
                            self.Omega_lambda) ** 0.5)
        integral_scipy, _ = integrate.quad(integrand_scipy, 0, z_max)
        d_z_scipy = integral_scipy * (self.c / self.H0)  # Convert to Mpc (c is in km/s and H0 in km/s/Mpc)
        return d_z_scipy
   
    def cumulative_trapezoidal(self, z_max, num_intervals):
        dz = z_max / num_intervals
        # cumulative integral values, start at 0 for z=0
        integral = [0.0] 
        z_values = [0.0]
        for i in range(1, num_intervals + 1):
            z0 = (i - 1) * dz
            z1 = i * dz
            z_values.append(z1)
            trapezoid = 0.5 * (self.integrand(z0) + self.integrand(z1)) * dz
            integral.append(integral[i - 1] + trapezoid)
        d_z_cumulative = [val * (3e5 / self.H0) for val in integral]  # Convert to Mpc
        return z_values, d_z_cumulative
   
    def interpolated_distance(self, z_array, num_intervals=1000):
        z_array = np.array(z_array)
        z_max = np.max(z_array)

        z_grid = np.linspace(0, z_max, num_intervals)
        D_grid = np.array([quad(self.integrand, 0, z)[0] * (self.c / self.H0) for z in z_grid])

        interpolator = interp1d(
            z_grid,
            D_grid,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate")
        
        return interpolator(z_array)
   
    def fractional_error_vs_evaluations(self, z_max, reference=None, n_start = 100, n_end = 1000, step = 10):
        "Compute fractional errors of rectangle, trapezoidal, and Simpson's rules"
        # Compute high-accuracy reference using SciPy
        reference = self.scipy_integral(z_max)
        num_intervals_list = list(range(n_start, n_end, step))

        frac_rect, frac_trap, frac_simp = [], [], []
        evals_rect, evals_trap, evals_simp = [], [], []

        for n in num_intervals_list:
            # rectangle and trapezoid can use n directly
            d_rect = self.rectangle_rule(z_max, n)
            d_trap = self.trapezoidal_rule(z_max, n)
            # ensure Simpson's rule has an even number of intervals
            n_simp = n if n % 2 == 0 else n + 1
            d_simp = self.simpsons_rule(z_max, n_simp)
            frac_rect.append(abs(d_rect - reference) / reference)
            frac_trap.append(abs(d_trap - reference) / reference)
            frac_simp.append(abs(d_simp - reference) / reference)
            # count actual integrand evaluations for each method
            evals_rect.append(n)            # rectangle: integrand evaluated n times
            evals_trap.append(n + 1)        # trapezoid: endpoints + (n-1) interior = n+1
            evals_simp.append(n_simp/2 + 1)   # Simpson: endpoints + n_simp interior = n_simp+1

        return evals_rect, frac_rect, evals_trap, frac_trap, evals_simp, frac_simp
   
    def D_L_array(self, z_array):
        D_L_list = []
        for z in z_array:
            # Comoving distance from SciPy
            D = self.scipy_integral(z)

            # Flat universe
            if abs(self.Omega_k) < 1e-5:
                D_L = (1 + z) * D
            else:
                sqrt_omega_k = np.sqrt(abs(self.Omega_k))
                x = sqrt_omega_k * self.H0 * D / self.c
                if self.Omega_k >= 1e-5:
                    S = np.sinh(x)
                elif self.Omega_k <= -1e-5:
                    S = np.sin(x)

                D_L = (1 + z) * (self.c / self.H0) * (S / sqrt_omega_k)
            D_L_list.append(D_L)
        return D_L_list
    
    def distance_modulus(self, z_array):
        D_L_array = self.D_L_array(z_array)
        mu_array = []
        for D_L in D_L_array:
            mu = 5 * np.log10(D_L) + 25
            mu_array.append(mu)
        return mu_array
    
    def interpolated_distance_modulus(self, z_array, N):
        # Interpolated comoving distances
        D_array = self.interpolated_distance(z_array, N)

        # Compute luminosity distance for each D
        mu_array = []
        for z, D in zip(z_array, D_array):
            # Flat universe
            if abs(self.Omega_k) < 1e-5:
                D_L = (1 + z) * D
            else:
                sqrt_ok = np.sqrt(abs(self.Omega_k))
                x = sqrt_ok * self.H0 * D / self.c
                if self.Omega_k > 0:
                    S = np.sinh(x)
                else:
                    S = np.sin(x)
                D_L = (1 + z) * (self.c / self.H0) * (S / sqrt_ok)
            # Calculate mu
            mu = 5 * np.log10(D_L) + 25
            mu_array.append(mu)
        return np.array(mu_array)


class Likelihood:
    def __init__(self, pantheon_data, M=-19.3):
        self.pantheon_data = pantheon_data
        self.M = M
        data = np.loadtxt(self.pantheon_data, skiprows=1)  # The first row is a header
        self.z = data[:, 0]
        self.mu_obs = data[:, 1]
        self.sigma = data[:, 2]
        self.points = len(self.z)

    def __call__(self, theta=None, N=10000, model="standard"):
        """__call__ methods allows the Likelihood object to be called as a function, which is convenient for optimization. 
        This method computes the log-likelihood for given cosmological parameters (theta) and interpolation points (N). 
        The model argument allows for different parameterizations (e.g., with or without Lambda)."""
        if model == "standard":
            H0, Omega_m, Omega_lambda = theta
        elif model == "no_lambda":
            H0, Omega_m = theta
            Omega_lambda = 0.0
        Omega_k = 1.0 - Omega_m - Omega_lambda
        cosmo = Cosmology(H0=H0, Omega_m=Omega_m, Omega_lambda=Omega_lambda, Omega_k=Omega_k)
        # compute model distance moduli
        mu_model = cosmo.interpolated_distance_modulus(self.z, N=N)
        mu_model += self.M
        logL = -0.5 * np.sum(((self.mu_obs - mu_model) / self.sigma) ** 2)
        return logL
    
    def convergence_plot(self, N_values):
        # reference likelihood (very large N)
        logL_ref = self(N=1000000) 
        logL_vals = []
        delta_logL = []

        for N in N_values:
            logL = self(N=N)
            logL_vals.append(logL)
            delta_logL.append(abs(logL - logL_ref))
        plt.figure()
        plt.loglog(N_values, delta_logL)
        plt.axhline(1.0, linestyle="--")
        plt.xlabel("Number of integration/interpolation points N")
        plt.ylabel(r"$|\log \mathcal{L}_N - \log \mathcal{L}_{\rm ref}|$")
        plt.title("Convergence of log-likelihood")
        plt.show()

    def fit(self, theta0, bounds, N=10000, model="standard"):
        """Fit the cosmological model to the data using maximum likelihood estimation."""
        # function to minimize
        def neg_logL(theta):
            return -self(theta, N=N, model=model)

        # run optimizer
        result = minimize(
            neg_logL,
            theta0,
            method="L-BFGS-B",
            bounds=bounds
        )

        # extract best-fit parameters
        if model == "standard":
            Omega_m, Omega_lambda, H0 = result.x
        elif model == "no_lambda":
            Omega_m, H0 = result.x
            Omega_lambda = 0.0
        Omega_k = 1.0 - Omega_m - Omega_lambda

        print("Model:", model)
        print("Best-fit parameters:" )
        print(f"  H0 = {H0:.2f} km/s/Mpc")
        print(f"  Omega_m = {Omega_m:.4f}")
        print(f"  Omega_lambda = {Omega_lambda:.4f}")
        print(f"  Omega_k = {Omega_k:.4f}")
        print(f"Log-likelihood = {-result.fun:.2f}")
        best_cosmo = Cosmology(H0=H0, Omega_m=Omega_m, Omega_lambda=Omega_lambda, Omega_k=Omega_k)
        return best_cosmo
    
    def plot_best_fit_model(self, best_cosmo, M=0.0, N=10000, key = "standard"):
        # Plot convergence of log-likelihood with increasing N
        plt.figure(figsize=(8,6))
        plt.errorbar(self.z, self.mu_obs, yerr=self.sigma, fmt='o', label='Pantheon data', markersize=4, capsize=3)

        # smooth model prediction at data redshifts
        mu_smooth = best_cosmo.interpolated_distance_modulus(self.z, N=N)
        mu_smooth += M  # add absolute magnitude offset

        plt.plot(self.z, mu_smooth, 'r-', linewidth=2, label='Best-fit model')
        plt.xlabel('Redshift z')
        plt.ylabel('Distance modulus μ(z) [mag]')
        plt.title('Pantheon Data with Best-Fit Model (' + key + ')')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_residuals(self, best_cosmo, M, N=10000, key = "standard"):
    
        # Model prediction at the data redshifts
        mu_smooth = best_cosmo.interpolated_distance_modulus(self.z, N=N)
        mu_smooth += M

        # Normalized residuals
        residuals = (self.mu_obs - mu_smooth) / self.sigma
        # Plot
        plt.figure(figsize=(8, 5))
        plt.errorbar(
            self.z,
            residuals,
            yerr=np.ones_like(residuals),  # unit variance after normalization
            fmt='o',
            markersize=5,
            capsize=3,
            label='Residuals'
        )

        plt.axhline(0, color='red', linestyle='--', label='Perfect fit')

        plt.xlabel('Redshift z')
        plt.ylabel('Normalized residuals $r_i$')
        plt.title('Normalized Residuals of Pantheon Distance Moduli (' + key + ')')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Summary statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals, ddof=1)
        print(f"Mean of residuals: {mean_residual:.6f}")
        print(f"Standard deviation of residuals: {std_residual:.6f}")

    def likelihood_grid(self, H0_vals, Omega_m_vals, Omega_lambda_vals, N=10000): #Compute a 3D grid of log-likelihood values:
        """ This method computes the log-likelihood on a 3D grid of cosmological parameters (H0, Omega_m, Omega_lambda)."""
        H0_vals = np.array(H0_vals)
        Omega_m_vals = np.array(Omega_m_vals)
        Omega_lambda_vals = np.array(Omega_lambda_vals)

        G = np.zeros((len(H0_vals), len(Omega_m_vals), len(Omega_lambda_vals)), dtype=float)

        # Loop over parameter grid
        for i, H0 in enumerate(H0_vals):
            for j, Omega_m in enumerate(Omega_m_vals):
                for k, Omega_lambda in enumerate(Omega_lambda_vals):

                    # Flatness condition
                    Omega_k = 1.0 - Omega_m - Omega_lambda
                    if Omega_k < -0.5:  # optional physical cut
                        G[i, j, k] = -np.inf
                        continue

                    theta = [Omega_m, Omega_lambda, H0]
                    G[i, j, k] = self(theta, N=N, model="standard")

        # convert log-likelihood to likelihood and normalize
        L = np.exp(G - np.max(G)) 
        L /= np.sum(L)
        return L
    
    def plot_2D_likelihood(self, L2D, x_vals, y_vals, xlabel, ylabel, title):
        """
        Plot a 2D marginalized likelihood using imshow.
        """
        plt.figure(figsize=(6,5))
        plt.imshow(
            L2D.T,                    # transpose so axes match intuitive x/y
            origin='lower',           # smallest y at bottom
            aspect='auto',
            extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
            cmap='viridis'
        )
        plt.colorbar(label='Likelihood')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_1D_likelihood(self, L1D, param_vals, xlabel, title):
        """Plot a 1D marginalized likelihood as a curve."""
        plt.figure(figsize=(6,4))
        plt.plot(param_vals, L1D, 'b-', lw=2)
        plt.xlabel(xlabel)
        plt.ylabel('Likelihood')
        plt.title(title)
        plt.grid(True)
        plt.show()


class Metropolis:
    def __init__(self, likelihood_function, theta0, bounds, sigmas, param_names, n_steps=50000):
        self.likelihood_function = likelihood_function
        self.theta0 = np.array(theta0, dtype=float)
        self.bounds = bounds
        self.sigmas = np.array(sigmas, dtype=float)
        self.n_steps = n_steps
        self.chain = np.zeros((n_steps, len(theta0)))
        self.log_likelihood_chain = np.zeros(n_steps)
        self.accepted = 0
        self.param_names = param_names

    def propose(self, theta):
        #Propose a new point using Gaussian random step.
        proposal = theta + self.sigmas * np.random.randn(len(theta))
        if self.bounds is not None:
            for i, (lower, upper) in enumerate(self.bounds):
                proposal[i] = np.clip(proposal[i], lower, upper)
        return proposal
    
    def run(self):
        """Run the Metropolis algorithm. 
        Initialize the chain and log-likelihood, then iterate proposing new points and accepting/rejecting based on likelihood ratio."""
        theta_current = self.theta0
        logL_current = self.likelihood_function(theta_current)
        
        self.chain[0] = theta_current
        self.log_likelihood_chain[0] = logL_current
        
        for i in range(1, self.n_steps):
            theta_proposed = self.propose(theta_current)
            logL_proposed = self.likelihood_function(theta_proposed)
            
            delta = logL_proposed - logL_current
            
            if delta > 0 or np.log(np.random.rand()) < delta:
                # Accept move
                theta_current = theta_proposed
                logL_current = logL_proposed
                self.accepted += 1
            
            self.chain[i] = theta_current
            self.log_likelihood_chain[i] = logL_current

    def likelihood_1D(self, bins=50, burn=0):
        chain = self.chain[burn:]
        # Plot the 1D marginalized likelihood for each parameter using a histogram.
        for i in range(chain.shape[1]):
            plt.figure(figsize=(6,4))
            plt.hist(chain[:, i], bins=bins, density=True)
            plt.xlabel(self.param_names[i])
            plt.ylabel("Posterior probability density")
            plt.title(f"Posterior distribution of {self.param_names[i]}")
            plt.show()

    def likelihood_2D(self, bins=50, burn=0):
        chain = self.chain[burn:]
        # Plot the 2D marginalized likelihood for two parameters using a 2D histogram.
        for i in range(chain.shape[1]):
            for j in range(i+1, chain.shape[1]):
                plt.figure(figsize=(6,5))
                plt.hist2d(chain[:, i], chain[:, j], bins=bins, density=True, cmap='viridis')
                plt.xlabel(self.param_names[i])
                plt.ylabel(self.param_names[j])
                plt.colorbar(label="Posterior probability density")
                plt.title(f"Joint probability density of {self.param_names[i]} and {self.param_names[j]}")
                plt.show()
                
    def likelihood_3D(self, burn=0):
        chain = self.chain[burn:]
        # Plot the 3D marginalized likelihood for all three parameters using a scatter plot.
        plt.scatter(chain[:, 1], chain[:, 2], c=chain[:, 0], s=5)
        plt.xlabel("Omega_m")
        plt.ylabel("Omega_lambda")
        plt.colorbar(label="H0")
        plt.title("Joint plot of Omega_m and Omega_lambda colored by H0")
        plt.show()
    def statistics(self, burn_in=300):
        # Returns summary statistics of the chain after burn-in.
        samples = self.chain[burn_in:]
        stats = {}

        for i, name in enumerate(self.param_names):
            param_samples = samples[:, i]
            stats[name] = {
                'mean': np.mean(param_samples),
                'std': np.std(param_samples, ddof=1),
                'median': np.median(param_samples),
                '16th percentile': np.percentile(param_samples, 16),
                '84th percentile': np.percentile(param_samples, 84)
            }
            print(f"Summary statistics for {name} after burn-in:")
            print("Mean = {:.4f}, Std = {:.4f}, Median = {:.4f}, 16th pct = {:.4f}, 84th pct = {:.4f}".format(
                stats[name]['mean'], stats[name]['std'], stats[name]['median'], stats[name]['16th percentile'], stats[name]['84th percentile']
            ))
        
        if self.accepted is not None:
            acc_rate = self.accepted / len(self.chain)
            stats['acceptance_rate'] = acc_rate
        print("Acceptance rate: {:.2f}%".format(stats['acceptance_rate'] * 100))
    
    def print_chain(self, burn=0):
        # Print the MCMC chain values and log-likelihoods after burn-in.
        print("Iteration |   H0   |   Omega_m   |   Omega_lambda   |  logL")
        print("-" * 60)
        for i, (theta, logL) in enumerate(zip(self.chain[burn:], self.log_likelihood_chain[burn:])):
            H0, Omega_m, Omega_lambda = theta
            print(f"{i:5d}   | {H0:10.4f} | {Omega_m:12.4f} | {Omega_lambda:14.4f} | {logL:10.4f}")


def task_41():    
    H0 = [71.2, 71.35, 71.5, 71.65, 71.8, 71.95, 72.1, 72.25, 72.4, 72.55, 72.7, 72.85, 73.0]         # Hubble constant in km/s/Mpc
    Omega_m = [0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46]                               # Matter density parameter
    Omega_lambda = [0.62, 0.65, 0.68, 0.71, 0.74, 0.77, 0.80, 0.83, 0.86, 0.89, 0.90, 0.93, 0.96, 0.99]                         # Dark energy density parameter
    M = -19.3                                   # Absolute magnitude of supernovae

    likelihood = Likelihood(pantheon_data="/Users/quinnbaldwin/Unit 1 computer modelling/Unit 4/pantheon_data.txt", M=M)
    Grid = likelihood.likelihood_grid(H0, Omega_m, Omega_lambda, N=10000)
    
    # 2D marginalized likelihoods
    L2D_H0_Omega_m = np.sum(Grid, axis=2)  # marginalize over Omega_lambda
    likelihood.plot_2D_likelihood(L2D_H0_Omega_m, H0, Omega_m, 'H0', 'Omega_m', 'marginalized over Omega_lambda')
    L2D_Omega_m_Omega_lambda = np.sum(Grid, axis=0)  # marginalize over H0
    likelihood.plot_2D_likelihood(L2D_Omega_m_Omega_lambda, Omega_m, Omega_lambda, 'Omega_m', 'Omega_lambda', 'marginalized over H0')
    L2D_H0_Omega_lambda = np.sum(Grid, axis=1)  # marginalize over Omega_m
    likelihood.plot_2D_likelihood(L2D_H0_Omega_lambda, H0, Omega_lambda, 'H0', 'Omega_lambda', 'marginalized over Omega_m')

    # 1D marginalized likelihoods
    L1D_Omega_m = np.sum(Grid, axis=(0,2))
    L1D_Omega_lambda = np.sum(Grid, axis=(0,1))
    L1D_H0 = np.sum(Grid, axis=(1,2))

    likelihood.plot_1D_likelihood(L1D_Omega_m, Omega_m, 'Omega_m', '1D marginalized over H0 & Omega_lambda')
    likelihood.plot_1D_likelihood(L1D_Omega_lambda, Omega_lambda, 'Omega_lambda', '1D marginalized over H0 & Omega_m')
    likelihood.plot_1D_likelihood(L1D_H0, H0, 'H0', '1D marginalized over Omega_m & Omega_lambda')


def task_42():
    ## Metropolis algorithm analysis
    # Initial conditions
    theta0 = [72, 0.3, 0.8]   
    bounds = [(50, 90), (0.0, 1.0), (0.0, 1.0)]
    M = -19.3     
    sigmas = [0.15, 0.03, 0.05]
    param_names = ['H0', 'Omega_m', 'Omega_lambda']
    n_steps = 5000
    burn_in = 300
    # Run MCMC
    likelihood = Likelihood(pantheon_data="/Users/quinnbaldwin/Unit 1 computer modelling/Unit 4/pantheon_data.txt", M=M)
    mcmc = Metropolis(likelihood, theta0, bounds, sigmas, param_names, n_steps)
    mcmc.run()
    mcmc.likelihood_1D(bins=30, burn=burn_in)
    mcmc.likelihood_2D(bins=30, burn=burn_in)
    mcmc.likelihood_3D(burn=burn_in)
    mcmc.statistics(burn_in=burn_in)



task = "__task_42__"

if task == "__task_42__":
    task_42()
elif task == "__task_41__":
    task_41()
