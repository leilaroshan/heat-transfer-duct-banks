import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import matplotlib.patches as patches
from matplotlib import colors as mcolors
from functools import partial

# Number of time steps
ntime_steps = 200000

class HeatTransfer:
  """Initializes the class with the given parameters and calculates additional properties.

Args:
    soil_properties (dict): A dictionary containing the properties of the soil. 
    backfill_properties (dict): A dictionary containing the properties of the backfill.
    cable_offset (float): The offset of the cable.
    ntime_steps (int, optional): The number of time steps. Defaults to 200000.

Attributes:
    soil_properties (dict): A dictionary containing the properties of the soil. 
    backfill_properties (dict): A dictionary containing the properties of the backfill.
    cable_offset (float): The offset of the cable.
    ntime_steps (int): The number of time steps.

Methods:
    calculate_soil_properties: Calculates additional soil properties.
    calculate_backfill_properties: Calculates additional backfill properties.
    conduction_convection: Simulates heat transfer using finite difference methods.
    _plot_heat_distribution: Plots the heat distribution.
    save_temperature_to_csv: Saves temperature data to a CSV file.
    run_simulation: Runs the complete heat transfer simulation.
"""
    def __init__(self, soil_properties, backfill_properties, cable_offset, ntime_steps=100000):
        """Initialize the HeatTransfer class with given parameters."""

        self.soil_properties = soil_properties
        self.backfill_properties = backfill_properties
        self.cable_offset = cable_offset
        self.ntime_steps = ntime_steps

        # Calculate additional properties
        self.calculate_soil_properties()
        self.calculate_backfill_properties()

    def calculate_soil_properties(self):
        """Calculate additional soil properties."""

        n = self.soil_properties['n']
        lambda_soil = self.soil_properties['lambda_soil']
        cp_soil = self.soil_properties['cp_soil']
        rhoS = self.soil_properties['rhoS']

        self.soil_properties['lambda_medium'] = lambda_soil * (1 - n)
        self.soil_properties['cp'] = cp_soil * (1 - n)
        self.soil_properties['alpha_soil'] = lambda_soil / (rhoS * cp_soil)

    def calculate_backfill_properties(self):
        """Calculate additional backfill properties."""

        n_backfill = self.backfill_properties['n_backfill']
        lambda_backfill = self.backfill_properties['lambda_backfill']
        cp_backfill = self.backfill_properties['cp_backfill']
        rho_backfill = self.backfill_properties['rho_backfill']

        self.backfill_properties['alpha_backfill'] = lambda_backfill / (rho_backfill * cp_backfill)

    def conduction_convection(self):
        """Simulate heat transfer using finite difference methods."""

        porosity = self.soil_properties['n']
        permeability_soil = self.soil_properties['permeability_soil']
        alpha_soil = self.soil_properties['alpha_soil']

        permeability_backfill = self.backfill_properties['permeability_backfill']
        alpha_backfill = self.backfill_properties['alpha_backfill']

        offset = self.cable_offset

        # Define the properties of the domain
        w = h = 1           # Width and height of the domain (meters)
        L = 1.0             # Length of the domain (meters)
        Nx = 200            # Number of grid points in the x direction
        dx = dy = w / Nx    # Grid spacing in the x direction
        rhow = 980          # Density of water (kg/m^3)
        mu = 1.00E-03       # Dynamic viscosity of water (Pa.s)
        g = 9.81            # Acceleration due to gravity (m/s^2)
        beta = 8.80E-05     # Coefficient of volumetric thermal expansion (1/K)
        conduction = 1.     # Conduction coefficient
        convection = 1.     # Convection coefficient
        Tcool= 0            # Boundary cool temperature
        Thot = 30           # Boundary hot temperature
        pr, px, py = 0.025, 0.5, 0.5  # Initial for coordinate and radius of the cable
        pr2 = pr**2         # Square of the radius of the cable
        nx, ny = int(w / dx), int(h / dy)   # Number of grid points in x and y directions
        dx2, dy2 = dx * dx, dy * dy         # Square of the grid spacing
        dt = 0.01                   # Time step
        nsteps = self.ntime_steps   # Number of time steps


        # Initial conditions and parameters for the simulation
        u0 = jnp.zeros((nx, ny))          # Initial temperature distribution matrix
        mask_cable = np.zeros((nx, ny))   # Mask for the cable region
        offset = jnp.array(offset)        # Offset for the cable

        # Calculate coordinates for the cables
        cable_y = np.arange(nx * ny).reshape(nx, ny) // nx * dy
        cable_x = (np.arange(nx * ny).reshape(nx, ny) // ny * dy).T
        epsilon = 1e-6
        con1 = jnp.where(((cable_x - px - offset)**2 + (cable_y - py)**2) <= (pr2 + epsilon), True, False)
        con2 = jnp.where(((cable_x - px + offset)**2 + (cable_y - py)**2) <= (pr2 + epsilon), True, False)
        mask_cable = con1 | con2
        mask_cable = jnp.asarray(mask_cable)

        # Set initial temperature values for the cable region
        u0 = mask_cable * Thot
        mask_cable_transform = 1 - mask_cable
        mask_backfill = np.zeros((nx, ny))

        # Set values for the backfill region
        backfill_size = 0.5
        for i in range(int((nx // 2 - backfill_size / dx / 2)), int((nx // 2 + backfill_size / dx / 2))):
            for j in range(int((ny // 2 - backfill_size / dy / 2)), int((ny // 2 + backfill_size / dy / 2))):
                mask_backfill[i, j] = 1.0
        mask_backfill = jnp.asarray(mask_backfill)
        mask_soil = 1 - mask_backfill

        # Set properties based on the masks for backfill and soil
        alpha = jnp.where(mask_backfill, alpha_backfill, alpha_soil)
        permeability = jnp.where(mask_backfill, permeability_backfill, permeability_soil)

        # Create a mask for the boundaries
        mask_boundaries = np.ones((nx, ny))
        mask_boundaries[:, 0] = 0.0
        mask_boundaries[:, nx - 1] = 0.0
        mask_boundaries[0, :] = 0.0
        mask_boundaries[nx - 1, :] = 0.0

        # Apply the backfill mask to the boundaries mask
        mask_boundaries = jnp.where(mask_backfill, mask_boundaries, 1.0)
        u0 = jnp.multiply(mask_boundaries, u0)

        # Initialize the temperature distribution variable
        u = u0

        # Calculate the convection factor based on parameters
        convection_factor = convection * dt * permeability * (1 / (porosity * mu) * g * rhow) / dy

        # Define a step function for the simulation
        def step(i, carry):
          u0, u, alpha, permeability = carry
          uip = jnp.roll(u0, 1, axis=0)
          ujp = jnp.roll(u0, 1, axis=1)
          uin = jnp.roll(u0, -1, axis=0)
          ujn = jnp.roll(u0, -1, axis=1)

          # Update the temperature distribution using conduction and convection terms
          conduction_term = jnp.where(mask_cable_transform, (uin - 2 * u0 + uip) / dy2 + (ujn - 2 * u0 + ujp) / dx2, 0)
          u = u0 + conduction * dt * alpha * conduction_term + (uip - u0) * convection_factor * (1 - beta * u0)

          u = jnp.multiply(u, mask_cable_transform) + mask_cable * Thot
          u = jnp.multiply(u, mask_boundaries)

          # Update the initial temperature distribution for the next iteration
          u0 = u

          return (u0, u, alpha, permeability)  # Corrected return statement

        # Iterate (updating)
        u0, u, _, _ = jax.lax.fori_loop(0, nsteps, step, (u0, u, alpha, permeability))
        u_forward = u0
        return {
          'temperature_distribution': jnp.array(u_forward, dtype=jnp.float32),
          'mask_backfill': mask_backfill,
          'mask_cable': mask_cable
        }


    def _plot_heat_distribution(self, u_forward):
        """Plot the heat distribution."""

        # Plotting heat distribution
        ny, nx = u_forward.shape
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))

        pcm_forward = axes.pcolormesh(u_forward, cmap=plt.cm.get_cmap("jet"), vmin=u_forward.min(), vmax=u_forward.max())
        axes.set_title('Forward FD Model')
        axes.add_patch(patches.Rectangle((nx // 4, ny // 4), nx // 2, ny // 2, linewidth=2, edgecolor='yellow', facecolor='none', label='Backfill Box'))
        axes.set_xlabel('L (m)')
        axes.set_ylabel('H (m)')
        axes.set_xticks([0, nx // 4, nx // 2, 3 * nx // 4, nx - 1])
        axes.set_yticks([0, ny // 4, ny // 2, 3 * ny // 4, ny - 1])
        axes.set_xticks([0, 40, 80, 120, 160, 200])
        axes.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes.set_yticks([0, 40, 80, 120, 160, 200])
        axes.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes.set_xlim([0, 200])
        axes.set_ylim([0, 200])

        cbar_forward = fig.colorbar(pcm_forward, ax=axes, label='Temperature', shrink=1.0)

        axes.add_patch(patches.Rectangle(
            (nx // 4, ny // 4), nx // 2, ny // 2,
            linewidth=2, edgecolor='yellow', facecolor='none', label='Backfill Box'
        ))

        im1 = axes.imshow(u_forward, cmap='jet', origin='lower')
        plt.tight_layout()
        plt.show()

    def save_temperature_to_csv(self, u):
        """Save temperature data to a CSV file."""

        np.savetxt('output_u.csv', u, delimiter=',')

    def run_simulation(self):
        """Run the complete heat transfer simulation."""

        results_dict = self.conduction_convection()
        u_forward = results_dict['temperature_distribution']
        mask_backfill = results_dict['mask_backfill']
        mask_cable = results_dict['mask_cable']

        average_temperature_inside_box = jnp.mean(u_forward[mask_backfill.astype(bool)])
        max_temperature_inside_box = jnp.max(u_forward[mask_backfill.astype(bool)])
        min_temperature_inside_box = jnp.min(u_forward[mask_backfill.astype(bool)])
        print("Forward offset:", self.cable_offset)
        print('avg: ', average_temperature_inside_box)
        print('min: ', min_temperature_inside_box)
        print('max: ', max_temperature_inside_box)
        self._plot_heat_distribution(u_forward)

        self.save_temperature_to_csv(u_forward)

if __name__ == "__main__":

    soil_properties = {
        "n": 0.6,                     # Porosity
        "lambda_soil": 1.5,           # Thermal conductivity (W/m-K)
        "cp_soil": 2000,              # Specific heat capacity (J/kg-K)
        "rhoS": 1850,                 # Density (kg/m^3)
        "permeability_soil": 1e-16,   # Permeability (m^2)
    }

    backfill_properties = {
        "n_backfill": 0.4,                   # Porosity
        "lambda_backfill": 1.0,              # Thermal conductivity (W/m-K)
        "cp_backfill": 800,                  # Specific heat capacity (J/kg-K)
        "rho_backfill": 1900,                # Density (kg/m^3)
        "permeability_backfill": 1.0E-11,    # Permeability (m^2)
    }

    cable_offset = 0.05

    heat_transfer_sim = HeatTransfer(soil_properties, backfill_properties, cable_offset, ntime_steps)
    heat_transfer_sim.run_simulation()
