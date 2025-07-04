# Charged Colloids

This repository contains various models for determining thermodynamic properties of charged colloid (and other custom potential) systems by brute forcing the statistical mechanical partitions functions for various ensembles.

## `finite_point_cloud.py`
Creates a point cloud in `n` dimensions without PBC.

### Usage
`python finite_point_cloud.py`

  `-h`, `--help`            show this help message and exit
  `--systems SYSTEMS`, `-s SYSTEMS`
                        Number of systems to run the simulation
  `--particles PARTICLES`, `-p PARTICLES`
                        Number of particles in each system
  `--max_temp MAX_TEMP`, `-t MAX_TEMP`
                        Maximum temperature in Kelvin
  `--force_linearity`, `-l`
                        Force linearity in energy distribution for numerical stability
  `--potential` {exponential, reciprocal, grample}
                        Type of potential to use: "exponential", "reciprocal" or "grample"
