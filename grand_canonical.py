import numpy as np
from numba import njit
from scipy.constants import physical_constants
from pandas import DataFrame
import matplotlib.pyplot as plt
from tqdm import tqdm

k_B = physical_constants["Boltzmann constant in eV/K"][0]  # Boltzmann constant in eV/K


def point_cloud(box_size, num_points):
    return np.random.uniform(0, box_size, size=(num_points, 3))


def get_distance_matrix(points):
    num_points = points.shape[0]
    distance_matrix = np.zeros((num_points, num_points), dtype=np.float64)
    
    for i in np.arange(num_points):
        for j in np.arange(i + 1, num_points):
            dist = np.linalg.norm(points[i] - points[j])
            # Only one of these is not necessary, but I can't remember which is the upper or lower triangle.
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
            
    return distance_matrix


def reciprocal(d, decay=2):
    return 1/(np.power(d, decay))

def two_body_potential(points, decay = -1):
    flattened = np.tril(get_distance_matrix(points)).flatten()
    flattened = flattened[flattened > 0]  # Remove zero distances
    return np.sum(reciprocal(flattened, decay))

def logsumexp2(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def get_counts(num_systems, population_ratios):
    total = np.sum(population_ratios)
    populations = np.ones(len(population_ratios), dtype=np.int64)  # Initialize populations with ones
    for idx, ratio in enumerate(population_ratios):
        populations[idx] = int(ratio * num_systems // total)
    return populations

def partition(energies, populations, mu, T):
    assert energies.shape == populations.shape, "Energies and populations must have the same shape."
    B = 1 / (k_B * T)
    Ei = energies - (populations * mu)
    lnZ = logsumexp2(Ei)
    w = -(Ei*B)
    Z = np.exp(lnZ)
    print(f"Partition function Z: {Z}, lnZ: {lnZ}")
    P = np.exp(w - lnZ)
    
    return Z, P, Ei

def var(x, p):
    mean_x = np.dot(x, p)
    mean_x2 = np.dot(np.power(x, 2), p)
    return mean_x2 - np.power(mean_x, 2)

def covar(x, y, p):
    mean_x = np.dot(x, p)
    mean_y = np.dot(y, p)
    xy = np.multiply(x, y)
    mean_xy = np.dot(xy, p)
    return mean_xy - (mean_x * mean_y)


def grand_cv(energies, populations, T, mu):
    Z, P, Ei = partition(energies, populations, mu, T)
    var_E = var(Ei, P)
    var_N = var(populations, P)
    cov_EN = covar(Ei, populations, P)
    return (1/(k_B * np.power(T, 2)))*(
        var_E - ((np.power(cov_EN, 2))/(var_N + 1e-15))  # Adding a small value to avoid division by zero
    )

mu = 1.0
total_systems = 5000
N_particles = np.array([10, 20, 30, 40, 50])
population_ratios = np.array([1, 3, 5, 7, 9])
counts = get_counts(total_systems, population_ratios)

df = DataFrame()
energies = []
particles = []

for idx, count in enumerate(counts):
    N = N_particles[idx]
    print(f"Group {idx}: {count} systems with N = {N} particles")
    for i in tqdm(range(count)):
        points = point_cloud(10.0, N)
        energies.append((two_body_potential(points, 1.5) + (mu* N))/N)
        particles.append(N)

energies = np.array(energies)
particles = np.array(particles)

temperatures = np.linspace(0.0001, 500.0, 2500)
Cvs = np.zeros(len(temperatures))

for idx, T in enumerate(temperatures):
    Cvs[idx] = grand_cv(energies, particles, T, mu)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(212)
ax1.plot(temperatures, Cvs, label='Grand Canonical Heat Capacity', color='blue')
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel('Heat Capacity (eV/K)')
ax1.set_title('Grand Canonical Heat Capacity vs Temperature')
ax1.legend()
ax1.grid()

ax2 = fig.add_subplot(211)
energies_sorted = np.sort(energies)
ax2.plot(np.arange(len(energies_sorted)), energies_sorted, label='Energy Distribution', color='orange')
ax2.set_xlabel('Ranking')
ax2.set_ylabel('Energy (eV) per particle')
ax2.set_title('Energy Distribution')
ax2.legend()
ax2.grid()
fig.suptitle('Grand Canonical Heat Capacity and Energy Distribution')
plt.tight_layout()
plt.show()