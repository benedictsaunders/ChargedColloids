import numpy as np
from numba import njit
from .finite_point_cloud import logsumexp2
from scipy.constants import physical_constants

k_B = physical_constants["Boltzmann constant in eV/K"][0]  # Boltzmann constant in eV/K

def point_cloud(box_size, num_points):
    return np.random.uniform(0, box_size, size=(num_points, 3))

def potential():
    None

def canonical_partition(energies, populations, potential, T):
    assert energies.shape == populations.shape, "Energies and populations must have the same shape."
    B = 1 / (k_B * T)
    Ei = energies - (populations * potential)
    lnZ = logsumexp2(Ei)
    w = -(Ei*B)
    Z = np.exp(lnZ)
    P = np.exp(Ei - lnZ)
    return Z, P, Ei

@njit(parallel=True)
def var():
    None

@njit(parallel=True)
def covar():
    None

def grand_cv(energies, populations, T):

    return (1/(k_B*np.power(T, 2)))*(

    )


populations = None
points = None

total_systems = 5000
energies = np.zeros(total_systems)

for idx, (sys, pop) in enumerate(zip(range(total_systems), populations)):
    system = point_cloud(20, pop)
    energies[idx] = potential(system)

