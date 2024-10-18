import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon

# Use ggplot style for nicer plots
plt.style.use('ggplot')

# Parameters
T = 10  # Final time period
num_simulations = 40  # Number of simulations

# General function to compute mood shocks from any distribution
def generate_mood_shocks(distribution, T, **kwargs):
    if distribution == "normal":
        mu = kwargs.get("mu", 0)
        sigma = kwargs.get("sigma", 1)
        return np.random.normal(mu, sigma, T)
    elif distribution == "uniform":
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 1)
        return np.random.uniform(low, high, T)
    elif distribution == "exponential":
        scale = kwargs.get("scale", 1)
        return np.random.exponential(scale, T)
    elif distribution == "multimodal":
        mu1, sigma1 = kwargs.get("mu1", -2), kwargs.get("sigma1", 1)
        mu2, sigma2 = kwargs.get("mu2", 2), kwargs.get("sigma2", 1)
        data = np.concatenate([
            np.random.normal(mu1, sigma1, T // 2),
            np.random.normal(mu2, sigma2, T // 2)
        ])
        np.random.shuffle(data)  # Shuffle to mix the modes
        return data
    else:
        raise ValueError("Unsupported distribution")

# Compute B(t) recursively
def compute_B(T, mu, sigma):
    B = np.zeros(T + 1)  # Initialize B(t) for t = 0 to T

    # Set B(T) = 0 since there is no future utility
    B[T] = 0

    # Value function iteration (backwards)
    for t in range(T - 1, -1, -1):
        # Define a function for the integral E_t[max{epsilon_t, B(t+1)}]
        B_next = B[t + 1]
        F_eps_next = norm.cdf(B_next, mu, sigma)
        E_eps_greater_next = mu + sigma * (norm.pdf(B_next, mu, sigma) / (1 - F_eps_next))
        B[t] = B_next * F_eps_next + E_eps_greater_next * (1 - F_eps_next)
    
    return B

# Function for the stopping rule
def stopping_rule(B):
    stopping_boundaries = B[1:]  # B(t+1) for each t < T
    return stopping_boundaries

# Simulate mood shock trajectories
def simulate_cake_eating(num_simulations, T, stopping_boundaries, distribution, **kwargs):
    results = []

    for sim in range(num_simulations):
        # Generate mood shocks for each simulation
        shocks = generate_mood_shocks(distribution, T, **kwargs)

        # Determine the day the cake is eaten based on the stopping rule
        for t in range(T):
            if shocks[t] > stopping_boundaries[t]:
                eat_day = t + 1  # Days are indexed from 1 to T
                break
        else:
            eat_day = T  # Eat on the last day if no earlier stopping condition is met

        # Truncate the shocks after the eating day
        truncated_shocks = shocks.copy()
        truncated_shocks[eat_day:] = np.nan  # Set shocks after eating to NaN for plotting

        results.append((truncated_shocks, eat_day))

    return results

# Function to plot the boundary and trajectories
def plot_trajectories_and_boundary(ax, stopping_boundaries, simulation_results, title):
    # Plot B(t+1), the boundary for eating the cake
    ax.plot(range(1, T + 1), stopping_boundaries, '--', color='black', label='B(t+1): Stopping Boundary')

    for shocks, eat_day in simulation_results:
        ax.plot(range(1, T + 1), shocks, color='blue')
        ax.plot(eat_day, shocks[eat_day - 1], 'ro')  # Highlight the day the cake was eaten

    # Add title, labels, and legend
    ax.set_title(title)
    ax.set_xlabel('Day')
    ax.set_ylabel('Mood Shocks')
    ax.grid(True)

# Run the simulation for each distribution and generate the plots in a 2x2 grid
def run_simulations():
    # Compute B(t) for the normal distribution
    mu, sigma = 0, 1
    B = compute_B(T, mu, sigma)
    stopping_boundaries = stopping_rule(B)

    # Prepare 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) Normal Distribution (Current Example)
    sim_results_normal = simulate_cake_eating(num_simulations, T, stopping_boundaries, "normal", mu=mu, sigma=sigma)
    plot_trajectories_and_boundary(axes[0, 0], stopping_boundaries, sim_results_normal, "Normal Distribution (μ=0, σ=1)")

    # (2) Exponential Distribution
    sim_results_exp = simulate_cake_eating(num_simulations, T, stopping_boundaries, "exponential", scale=1)
    plot_trajectories_and_boundary(axes[0, 1], stopping_boundaries, sim_results_exp, "Exponential Distribution")

    # (3) Uniform Distribution
    sim_results_uniform = simulate_cake_eating(num_simulations, T, stopping_boundaries, "uniform", low=-1, high=1)
    plot_trajectories_and_boundary(axes[1, 0], stopping_boundaries, sim_results_uniform, "Uniform Distribution [-1, 1]")

    # (4) Multimodal Distribution
    sim_results_multi = simulate_cake_eating(num_simulations, T, stopping_boundaries, "multimodal", mu1=-2, sigma1=0.5, mu2=2, sigma2=0.5)
    plot_trajectories_and_boundary(axes[1, 1], stopping_boundaries, sim_results_multi, "Multimodal Distribution (μ1=-2, μ2=2)")

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Run all simulations and plots in a 2x2 grid
run_simulations()
