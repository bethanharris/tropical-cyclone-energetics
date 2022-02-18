Code for computing closed budget of moist APE density for an axisymmetric tropical cyclone model, as described in:
Harris et al., 2022: "A Moist Available Potential Energy Budget for an Axisymmetric Tropical Cyclone".
All figures/equations refer to that paper. 
Files described here in the order they should be run: later scripts rely on saved output of e.g. APE density computation.

RE87_jordan_30pt3.f is the version of the Rotunno & Emanuel (1987) model that was used to simulate the axisymmetric TC.

model_reader.py contains code to read the output of this axisymmetric run into a python dictionary.

intensity_plot.py creates time series of TC intensity (Fig. 1).

kinetic_elastic_energy_budgets.py computes and saves all terms in the kinetic energy budgets (u, v, and w components) and
the available elastic energy budget, for all parcels at each time step of the output. See Eqs. 6, 9 for budget details.

ape_density.py computes and saves moist APE density, reference height and APE production coefficients (G_{\theta_{ei}} and G_{rt})
for each parcel. This is set up to run with multiprocessing on 8 cores, because computation for each parcel is independent.
Note the code is set up to work with different reference state profiles in future, but currently only 'initial' is supported as reference state argument.

ape_budget.py computes and saves the terms in the APE budget (Eq. 33) for each parcel.

energy_budget_timeseries.py plots time series of volume-integrated energy budgets (e.g. vertical kinetic energy,
total kinetic energy, available elastic energy, APE). Includes verification plots for budget closure. Produces Figs 2, 8, 13, A3.

ape_import_export.py looks at the flux into/out of the centre of the model domain, as in Fig. 9.

ape_production_timeseries.py plots time series of volume integrals of APE production (e.g. Figs 11, A4).
Includes verification plots for budget closure. Also includes breakdown of surface flux production into components (Fig. 12).

spatial_plots.py creates various plots showing r-z cross-section snapshots at a model time step. Used for Figs 5, 6, 7, 10.

zr_jump_plots.py contains code to illustrate the discontinuous behaviour of moist APE density in the atmosphere (as in Fig. 3).
This works with continuous_ape.py, which contains a version of the APE density computation that numerically integrates
an interpolated buoyancy profile rather than summing over discretised vertical levels as in Eq. A2 (this prevents negative values of APE, which may occur in the discretised version, as discussed in the paper).