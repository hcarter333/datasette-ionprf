#!/usr/bin/env python3
"""
Example Python script to extract E region data (90–150 km) from a COSMIC‐2 IONOPRF NetCDF file.
"""

import netCDF4
import numpy as np
import matplotlib.pyplot as plt

# Define the path to your NetCDF file
filename = 'your_ionprf_file.nc'

# Open the NetCDF file for reading
ds = netCDF4.Dataset(filename, 'r')

# List all variables in the dataset (for inspection)
print("Variables in the dataset:")
for var in ds.variables:
    print(f"  {var}")

# -------------------------------------------------------------------------
# Note: Adjust these variable names based on the actual file structure.
# Common names might be:
#   - 'altitude' or 'z': altitude (in meters)
#   - 'electron_density' or 'Ne': electron density
#
# Here we assume:
#   - 'altitude': Either a 1D array (common to all profiles) or a 2D array (profiles x levels)
#   - 'electron_density': 2D array with shape (n_profiles, n_levels)
# -------------------------------------------------------------------------

# For example, suppose the altitude variable is 'altitude' and electron density is 'electron_density'
# Check dimensions and shape
alt_var = ds.variables['altitude']
ne_var = ds.variables['electron_density']

print(f"\n'altitude' variable shape: {alt_var.shape}")
print(f"'electron_density' variable shape: {ne_var.shape}")

# Read in the data
altitudes = alt_var[:]         # in meters; shape could be (n_levels,) or (n_profiles, n_levels)
electron_density = ne_var[:]   # shape expected to be (n_profiles, n_levels)

# Define the altitude range for the E region in meters
e_region_min = 90e3   # 90 km in meters
e_region_max = 150e3  # 150 km in meters

# Two cases: 
# (A) Altitude is a 1D array common to all profiles.
# (B) Altitude is a 2D array with each profile having its own altitude levels.

# Case A: 1D altitude array (common to all profiles)
if altitudes.ndim == 1:
    # Create a boolean mask for the E region altitudes
    e_mask = (altitudes >= e_region_min) & (altitudes <= e_region_max)
    e_altitudes = altitudes[e_mask]  # Altitudes within the E region
    
    # Apply the mask to all profiles: assuming axis=1 corresponds to altitude
    e_electron_density = electron_density[:, e_mask]
    
    print(f"\nExtracted E region data (common altitude array):")
    print(f"Altitudes in E region: {e_altitudes}")
    print(f"Shape of electron density in E region: {e_electron_density.shape}")

# Case B: 2D altitude array (each profile has its own altitude levels)
else:
    n_profiles = altitudes.shape[0]
    e_altitudes_list = []       # to store altitude arrays for each profile in the E region
    e_electron_density_list = []  # to store electron density arrays for each profile in the E region

    for i in range(n_profiles):
        alt_profile = altitudes[i, :]   # Altitude profile for index i
        ne_profile = electron_density[i, :]  # Corresponding electron density
        
            # Create a mask for the E region for this profile
        mask = (alt_profile >= e_region_min) & (alt_profile <= e_region_max)
        
        # Only add the profile if there is data in the E region
        if np.any(mask):
            e_altitudes_list.append(alt_profile[mask])
            e_electron_density_list.append(ne_profile[mask])
    
    print(f"\nExtracted E region data for {len(e_altitudes_list)} profiles out of {n_profiles} total profiles.")

# Optionally, plot the first profile's E region data (if available)
if altitudes.ndim == 1:
    # Plot using the common altitude array
    plt.figure(figsize=(6, 8))
    plt.plot(e_electron_density[0, :], e_altitudes/1e3, marker='o')
    plt.xlabel("Electron Density")
    plt.ylabel("Altitude (km)")
    plt.title("E Region Electron Density Profile (Profile 0)")
    plt.grid(True)
    plt.show()
elif len(e_altitudes_list) > 0:
    plt.figure(figsize=(6, 8))
    plt.plot(e_electron_density_list[0], e_altitudes_list[0]/1e3, marker='o')
    plt.xlabel("Electron Density")
    plt.ylabel("Altitude (km)")
    plt.title("E Region Electron Density Profile (First available profile)")
    plt.grid(True)
    plt.show()

# Close the dataset when done
ds.close()
