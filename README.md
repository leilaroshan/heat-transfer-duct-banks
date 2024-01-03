# heat-transfer-duct-banks
Heat dissipation in underground power cable duct banks using finite-difference forward method
# Underground power cable duct banks simulation using JAX
This study aims to develop a novel optimization algorithm for designing both cable spacing and backfill material properties.
The first step in the process is the forward problem where we obtain the temperature distribution across the soil volume resulting from a known initial temperature distribution, soil thermal properties and permeability. We use a 2D Finite Difference model to solve the forward problem.  We consider both conductive and convective heat transfer mechanisms by solving the partial differential equations for the heat transfer and the time-independent coupled fluid flow.
Here are some examples of simulation running using JAX, with the cable spacing of 0.05m.

<img width="633" alt="Screenshot 2024-01-03 at 11 50 05 AM" src="https://github.com/geoelements-dev/heat-transfer-duct-banks/assets/118838742/fea67a9d-b2e0-47ae-8dea-02f1deacec4a">
