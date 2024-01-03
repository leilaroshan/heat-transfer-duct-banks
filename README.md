# Heat dissipation in underground power cable duct banks using JAX
This project aims to develop a simulation for designing both cable spacing and backfill material properties.
The first step in the process is the forward problem where we obtain the temperature distribution across the soil volume resulting from a known initial temperature distribution, soil thermal properties and permeability. We use a 2D Finite Difference model to solve the forward problem.  We consider both conductive and convective heat transfer mechanisms by solving the partial differential equations for the heat transfer and the time-independent coupled fluid flow.

Here is some examples of simulation running with JAX and cable spacing of 0.05m.

<img width="633" alt="Screenshot 2024-01-03 at 11 50 05 AM" src="https://github.com/geoelements-dev/heat-transfer-duct-banks/assets/118838742/5e83bb93-c7c2-44c5-b08c-ad8ff785df70">
