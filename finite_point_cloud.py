import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from scipy.special import logsumexp

k_B = physical_constants["Boltzmann constant in eV/K"][0]  # Boltzmann constant in eV/K

@njit(parallel=True)
def finite_point_cloud(box_size, num_points):
    """
    Generate a finite point cloud within a cubic box of given size.

    Parameters:
    box_size (float): The size of the cubic box.
    num_points (int): The number of points to generate.

    Returns:
    np.ndarray: An array of shape (num_points, 3) containing the coordinates of the points.
    """
    return np.random.uniform(0, box_size, size=(num_points, 3))

def get_distance_matrix(points):
    """
    Calculate the distance matrix for a set of points.

    Parameters:
    points (np.ndarray): An array of shape (num_points, 3) containing the coordinates of the points.

    Returns:
    np.ndarray: A square matrix of shape (num_points, num_points) containing the distances between each pair of points.
    """
    num_points = points.shape[0]
    distance_matrix = np.zeros((num_points, num_points), dtype=np.float64)
    
    for i in tqdm(range(num_points), desc="Calculating distance matrix", leave=False):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(points[i] - points[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
            
    return distance_matrix

def potential_energy(points, cutoff, zero = 0, onebody = 1.0, two_body_init = 1.0, two_body_decay = -2., show_potential = False):
    """
    Calculate the potential energy of a system of points in a cubic box.

    Parameters:
    points (np.ndarray): An array of shape (num_points, 3) containing the coordinates of the points.
    box_size (float): The size of the cubic box.

    Returns:
    float: The total potential energy of the system.
    """

    def potential(d):
        """
        Calculate the potential energy for a given distance.

        Parameters:
        d (float): The distance between two points.

        Returns:
        float: The potential energy for the given distance.
        """
        # return two_body_init * np.exp(two_body_decay * d)
        return 1/(np.power(d, two_body_decay))

        
    E = 0.0
    flattened = np.tril(get_distance_matrix(points)).flatten()
    flattened = flattened[flattened > 0]  # Remove zero distances

    one_body_terms = onebody * len(points)  # One-body potential energy term
    two_body_terms = potential(flattened)  # Two-body potential energy terms

    E = one_body_terms + np.sum(two_body_terms)

    if show_potential:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(flattened, potential(flattened), label='Potential Energy', color='blue')
        plt.show()

    return E

@njit(parallel=True)
def logsumexp2(x):
    """
    Compute the log-sum-exp of an array for numerical stability.

    Parameters:
    x (np.ndarray): Input array.

    Returns:
    float: The log-sum-exp of the input array.
    """
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


@njit(parallel=True)
def canonical_partition_function(energies, temperature):
    """
    Calculate the canonical partition function for a set of energies at a given temperature.

    Parameters:
    energies (np.ndarray): An array of energies.
    temperature (float): The temperature in Kelvin.

    Returns:
    float: The canonical partition function and the Boltzmann probabilities.
    """
    # energies = energies/10000
    B = 1 / (k_B * temperature)
    w = -(energies*B)
    lnZ = logsumexp2(w)  # Log-sum-exp for numerical stability
    Z, P = np.exp(lnZ), np.exp(w - lnZ)
    return Z, P

def canonical_heat_capacity(energies, probabilities, temperature):
    """
    Calculate the canonical heat capacity for a set of energies and their probabilities at a given temperature.

    Parameters:
    energies (np.ndarray): An array of energies.
    probabilities (np.ndarray): An array of probabilities corresponding to each energy.z
    temperature (float): The temperature in Kelvin.

    Returns:
    float: The canonical heat capacity.
    """

    # Calculate the average energy
    U = np.dot(energies, probabilities)
    
    # Fluctuations in energy
    fluctuations = -np.dot(np.power(energies, 2), probabilities) + np.power(U, 2)
    # Heat capacity formula
    heat_capacity = fluctuations / -(k_B * np.power(temperature, 2))
    return heat_capacity

num_systems = 5000  # Number of times to run the simulation
num_particles = 40  # Number of particles in each system

energies = np.zeros(num_systems)
for idx in tqdm(range(num_systems)):
    points = finite_point_cloud(15.0, num_particles)
    energies[idx] = potential_energy(points, cutoff=10, onebody=0., two_body_init=1., two_body_decay=-.5, show_potential=False)/num_particles

# Define a range of temperatures for the simulation
T = np.linspace(0.1, 15000.0, 500)  #
Cv = np.zeros(len(T))

for idx, t in tqdm(enumerate(T), desc="Calculating heat capacity", total=len(T)):
    #print(energies)
    Z, P = canonical_partition_function(energies, temperature=t)
    #print(f"Partition function Z at T={t} K: {Z}")
    Cv[idx] = canonical_heat_capacity(energies, P, temperature=t)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(T, Cv, label='Heat Capacity')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Heat Capacity (ev / K / particle)')
ax.set_title('Canonical Heat Capacity vs Temperature')
ax.legend()
ax2 = fig.add_subplot(212)
ax2.plot(np.arange(len(energies)), np.sort(energies), label='Heat Capacity', color='orange')
ax2.set_xlabel('Ranking')
ax2.set_ylabel('Energy (eV) per particle')

fig.suptitle('Canonical Heat Capacity and Energy Distribution')
fig.tight_layout()

fig.savefig('heat_capacity.pdf', dpi=300)
plt.show()





