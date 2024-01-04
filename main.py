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

