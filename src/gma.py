# -*- coding: utf-8 -*-
"""
Gibbs Moist Air Library: Functions to calculate various thermodynamic
properties of moist air. Formulae according to Ambaum "Thermal Physics of
the Atmosphere" (Wiley-Blackwell, 2010).

Inputs to all functions should be given in units:
    Temperature: C
    Pressures: mb
    Mixing ratios/specific humidities: kg/kg

Remi Tailleux/Bethan Harris, University of Reading, 11th November 2016
"""
import numpy as np

# ==============================================================================
# Define useful thermodynamic constants
# ==============================================================================

t0 = 273.15  # 0 degrees Celsius in K
t25_k = 25. + 273.15  # 25 degrees Celsius in K
cl = 4179.9  # specific heat capacity for liquid water (J/kg/K)
cpd = 1004.5  # specific heat at constant pressure for dry air (J/kg/K)
cpv = 1865.1  # specific heat capacity at constant pressure for water vapour (J/kg/K)
ci = 1960.  # specific heat capacity at constant pressure for ice (J/kg/K)
e0_mb = 10.  # reference partial vapour pressure for water vapour (mb)
es0_mb = 31.6743  # reference saturation vapour pressure(mb)
es0_mb_ice = 6.108  # saturation vapour pressure over ice at T=0C(mb)
alv0 = 2.444e6  # latent heat of evaporation at T=25C (J/kg)
sub0 = 2.834e6  # latent heat of sublimation at T=0C (J/kg)
p0_mb = 1000.  # reference pressure (mb)
pd0_mb = p0_mb - e0_mb  # reference partial pressure of dry air (mb)
Rd = 287.  # specific gas constant for dry air (J/kg/K)
Rv = 461.5  # specific gas constant for water vapour (J/kg/K)
eeps = Rd / Rv
chi = Rd / cpd
cpvmcl = cl - cpv
alpha = cpvmcl / Rv
alpha_ice = (ci - cpv) / Rv


def saturation_vapour_pressure(t_c):
    """Compute saturation water vapour pressure (mb)
    INPUT: temperature (C)"""
    t_k = t_c + 273.15
    dt_k = t_k - t25_k
    es_mb = (es0_mb * np.exp((alv0 / Rv) * dt_k / (t_k * t25_k)) * (t25_k / t_k) ** alpha * np.exp(alpha * dt_k / t_k))
    return es_mb


def saturation_vapour_pressure_ice(t_c):
    """Compute saturation water vapour pressure (mb)
    INPUT: temperature (C)"""
    t_k = t_c + 273.15
    dt_k = t_k - t0
    es_mb = (es0_mb_ice * np.exp((sub0 / Rv) * dt_k / (t_k * t0)) * (t0 / t_k) ** alpha_ice * np.exp(
        alpha_ice * dt_k / t_k))
    return es_mb


def saturation_mixing_ratio(t_c, p_mb):
    """Compute saturation mixing ratio (kg/kg)
    INPUT: temperature (C), pressure(mb)"""
    es_mb = saturation_vapour_pressure(t_c)
    # Compute saturation mixing ratio as if everything was fine
    rvs_kgkg = eeps * es_mb / (p_mb - es_mb)
    # Set negative values to a very high values that should guarantee
    # lack of saturation
    rvs_kgkg[np.where(rvs_kgkg < 0.)] = 1e6
    return rvs_kgkg


def saturation_mixing_ratio_ice(t_c, p_mb):
    """Compute saturation mixing ratio (kg/kg)
    INPUT: temperature (C), pressure(mb)"""
    es_mb = saturation_vapour_pressure_ice(t_c)
    # Compute saturation mixing ratio as if everything was fine
    rvs_kgkg = eeps * es_mb / (p_mb - es_mb)
    # Set negative values to a very high values that should guarantee
    # lack of saturation
    rvs_kgkg[np.where(rvs_kgkg < 0.)] = 1e6
    return rvs_kgkg


def latent_heat(t_c):
    """Compute latent heat of evaporation (J/kg)
    INPUT: temperature (C)"""
    return alv0 - cpvmcl * (t_c - 25.)


def latent_heat_sub(t_c):
    """Compute latent heat of sublimation (J/kg)
    INPUT: temperature (C)"""
    return sub0 - (ci - cpv) * t_c


def rt_from_qt(qt_kgkg):
    """Compute total mixing ratio (kg/kg)
    INPUT: total specific humidity (kg/kg)"""
    return qt_kgkg / (1 - qt_kgkg)


def qt_from_rt(rt_kgkg):
    """Compute total specific humidity (kg/kg)
    INPUT: total mixing ratio (kg/kg)"""
    return rt_kgkg / (1 + rt_kgkg)


def partial_vapour_pressure(t_c, p_mb, qt_kgkg):
    """Compute the partial pressure of water vapour (mb)
    INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)"""
    rt_kgkg = rt_from_qt(qt_kgkg)
    e_mb = rt_kgkg * p_mb / (eeps + rt_kgkg)
    es_mb = saturation_vapour_pressure(t_c)
    e_mb = np.minimum(e_mb, es_mb)
    return e_mb


def entropy_dry_air(t_c, pd_mb):
    """Compute the partial specific entropy for dry (J/kg/K)
     INPUT: temperature (C), partial dry air pressure (mb)"""
    t_k = t_c + t0
    eta_d = cpd * np.log(t_k / t0) - Rd * np.log(pd_mb / pd0_mb)
    return eta_d


def entropy_water_vapour(t_c, e_mb):
    """Compute the partial specific entropy for water vapour (J/kg/K)
     INPUT: temperature (C), partial water vapour pressure (mb)"""
    if isinstance(t_c, np.float32) or isinstance(t_c, np.float64) or isinstance(t_c, float):
        t_c = np.array([t_c])
        e_mb = np.array([e_mb])
    t_k = t_c + t0
    e_mb_term = Rv * np.log(e_mb / e0_mb)
    eta_v = cpv * np.log(t_k / t0) - e_mb_term
    # Set -Inf values to 0
    eta_v[np.where(np.isinf(e_mb_term))] = 0.
    return eta_v


def moist_entropy(t_c, p_mb, qt_kgkg):
    """Compute the moist specific entropy (J/kg/K) 
     INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)
     assuming phase liquid equilibrium"""
    t_k = t_c + t0
    rl_kgkg = liquid_mixing_ratio(t_c, p_mb, qt_kgkg)
    ql_kgkg = rl_kgkg * (1. - qt_kgkg)
    qv_kgkg = qt_kgkg - ql_kgkg
    e_mb = partial_vapour_pressure(t_c, p_mb, qt_kgkg)
    pd_mb = p_mb - e_mb
    lv = latent_heat(t_c)
    cp_eff = (1. - qt_kgkg) * cpd + qt_kgkg * cl
    eta_ma = cp_eff * np.log(t_k) - (1. - qt_kgkg) * Rd * np.log(
        pd_mb * 100.) + lv * qv_kgkg / t_k - qv_kgkg * Rv * np.log(e_mb / saturation_vapour_pressure(t_c))
    return eta_ma


def cp_moist_air_exact(t_c, p_mb, qt_kgkg):
    """Compute effective specific heat capacity (J/kg/K) for moist air
     INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)"""
    rl_kgkg = liquid_mixing_ratio(t_c, p_mb, qt_kgkg);
    ql_kgkg = rl_kgkg * (1 - qt_kgkg)
    qv_kgkg = qt_kgkg - ql_kgkg

    e_mb = partial_vapour_pressure(t_c, p_mb, qt_kgkg)
    pd_mb = p_mb - e_mb

    tk = t_c + t0
    lv = latent_heat(t_c)
    qvlvdrdt = (qv_kgkg / Rv) * (p_mb / pd_mb) * (lv / tk) ** 2

    cp_moist = cpd * (1 - qt_kgkg) + cpv * qv_kgkg
    isat = np.where(rl_kgkg > 0.)
    cp_moist[isat] = cp_moist[isat] + cl * ql_kgkg[isat] + qvlvdrdt[isat]
    return cp_moist


def liquid_mixing_ratio(t_c, p_mb, qt_kgkg):
    """Compute the liquid water mixing ratio (kg/kg) for saturated moist air
    INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)"""
    # Compute partial pressure assuming that air is unsaturated
    rt_kgkg = rt_from_qt(qt_kgkg)
    e_mb = rt_kgkg * p_mb / (eeps + rt_kgkg)
    # Compute saturation water pressure
    es_mb = saturation_vapour_pressure(t_c)
    # Define partial water vapour pressure as the mininum of the two pressure
    e_mb = np.minimum(e_mb, es_mb)
    # Compute water vapour and liquid mixing ratio
    rv_kgkg = eeps * e_mb / (p_mb - e_mb)
    rl_kgkg = rt_kgkg - rv_kgkg
    return rl_kgkg


def saturation_vapour_pressure_ice_approx(t_c):
    t_K = t_c + t0
    log_e_sat = 23.33086 - 6111.72784 / t_K + 0.15215 * np.log(t_K)
    e_sat_mb = np.exp(log_e_sat)
    return e_sat_mb


def all_mixing_ratios(t_c, p_mb, rt_kgkg):
    """Compute the liquid water and ice mixing ratios (kg/kg) for saturated moist air
    INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)"""
    # Compute partial pressure assuming that air is unsaturated
    e_mb = rt_kgkg * p_mb / (eeps + rt_kgkg)
    # Compute saturation water pressure
    es_mb_liq = saturation_vapour_pressure(t_c)
    es_mb_ice = saturation_vapour_pressure_ice(t_c)
    frozen = t_c < 0.
    # Define partial water vapour pressure as the mininum of the two pressure
    e_mb = np.minimum(e_mb, es_mb_liq) * np.logical_not(frozen) + np.minimum(e_mb, es_mb_ice) * frozen
    # Compute water vapour and liquid mixing ratio
    rv_kgkg = eeps * e_mb / (p_mb - e_mb)
    rl_kgkg = (rt_kgkg - rv_kgkg) * np.logical_not(frozen)
    ri_kgkg = (rt_kgkg - rv_kgkg) * frozen
    return rv_kgkg, rl_kgkg, ri_kgkg


def spec_vol(t_c, p_mb, qt_kgkg):
    """Compute the specific volume (m^3/kg) of moist air
    INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)"""
    t_k = t_c + t0
    # Compute partial pressure of dry air
    pd_mb = p_mb - partial_vapour_pressure(t_c, p_mb, qt_kgkg)
    pd_ppa = pd_mb * 100
    # Compute specific volume according to alpha = (1-qt)*alpha_d
    spec_vol = (1 - qt_kgkg) * Rd * t_k / pd_ppa
    return spec_vol


def moist_enthalpy(t_c, p_mb, qt_kgkg):
    """Compute specific moist enthalpy (J/kg/K) assuming phase liquid equilibrium
    INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg) 
    """
    # Compute mixing ratio and specific humidity for the liquid part
    rl_kgkg = liquid_mixing_ratio(t_c, p_mb, qt_kgkg)
    ql_kgkg = rl_kgkg * (1 - qt_kgkg)
    qv_kgkg = qt_kgkg - ql_kgkg
    # Compute latent heat
    lv = latent_heat(t_c)
    # Estimate moist entropy as weighted partial entropies, accounting for
    # liquid mixing ratio contribution separately for saturated case
    cp_eff = cpd * (1 - qt_kgkg) + cl * qt_kgkg
    h_ma = cp_eff * (t_c + t0) + qv_kgkg * lv
    return h_ma


def spec_hum_from_rel_hum(t_c, p_mb, rh):
    """Compute specific humidity (kg/kg)
    INPUT: temperature (C), pressure (mb), relative humidity (fraction 0 to 1)"""
    es_mb = saturation_vapour_pressure(t_c)
    e_mb = rh * es_mb
    spec_hum = e_mb / (e_mb + (p_mb - e_mb) / eeps)
    return spec_hum


def gibbs_dry_air(t_c, pd_mb):
    """Compute Gibbs function for dry air (J/kg) 
    INPUT: temperature (C), partial dry air pressure (mb)"""
    t_k = t_c + t0
    gd = cpd * (t_k - t0 - t_k * np.log(t_k / t0)) + Rd * t_k * np.log(pd_mb / pd0_mb)
    return gd


def gibbs_water_vapour(t_c, e_mb):
    """Compute Gibbs function for water vapour (J/kg)
    INPUT: temperature (C), partial water vapour pressure (mb)"""
    if isinstance(t_c, np.float32) or isinstance(t_c, np.float64) or isinstance(t_c, float):
        t_c = np.array([t_c])
        e_mb = np.array([e_mb])
    t_k = t_c + t0
    gv_temperature = cpv * (t_k - t0 - t_k * np.log(t_k / t0))
    gv_pressure = Rv * t_k * np.log(e_mb / e0_mb)
    gv = gv_temperature + gv_pressure
    gv[np.where(np.isinf(gv_pressure))] = 0.
    return gv


def chemical_potential(t_c, p_mb, qt_kgkg):
    """Compute chemical potential of moist air (J/kg)
    INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)"""
    e_mb = partial_vapour_pressure(t_c, p_mb, qt_kgkg)
    pd_mb = p_mb - e_mb
    mu = gibbs_water_vapour(t_c, e_mb) - gibbs_dry_air(t_c, pd_mb)
    return mu


def dry_potential_temperature(t_c, p_mb):
    """Compute potential temperature (K) 
    INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)"""
    theta = (t_c + t0) * (p0_mb / p_mb) ** (Rd / cpd)
    return theta


def equivalent_potential_temperature(t_c, p_mb, qt_kgkg):
    """Compute equivalent potential temperature (K)
    INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)"""
    theta = dry_potential_temperature(t_c, p_mb)
    rt_kgkg = rt_from_qt(qt_kgkg)
    rl_kgkg = liquid_mixing_ratio(t_c, p_mb, qt_kgkg)
    rv_kgkg = rt_kgkg - rl_kgkg
    es_mb = saturation_vapour_pressure(t_c)
    a = np.exp(alv0 * rv_kgkg / (cpd * (t_c + t0)))
    b = ((t_c + t0) / t0) ** (rl_kgkg * cl / cpd)
    c = (1. - es_mb / p_mb) ** (-Rd / cpd)
    theta_e_K = theta * a * b * c
    return theta_e_K


def saturated_equivalent_potential_temperature(t_c, p_mb, qt_kgkg):
    """Compute saturated equivalent potential temperature (K)
    INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)"""
    theta = dry_potential_temperature(t_c, p_mb)
    rvs_kgkg = saturation_mixing_ratio(t_c, p_mb)
    rl_kgkg = liquid_mixing_ratio(t_c, p_mb, qt_kgkg)
    es = saturation_vapour_pressure(t_c)
    a = np.exp(alv0 * rvs_kgkg / (cpd * (t_c + t0)))
    b = ((t_c + t0) / t0) ** (rl_kgkg * cl / cpd)
    c = (1 - es / p_mb) ** (-Rd / cpd)
    theta_es = theta * a * b * c
    return theta_es


def dmu_dt(t_c, p_mb, qt_kgkg):
    """Compute partial derivative of chemical potential with respect to temperature
    INPUT: temperature (C), pressure (mb), total specific humidity (kg/kg)"""
    t_k = t_c + t0
    rl_kgkg = liquid_mixing_ratio(t_c, p_mb, qt_kgkg)

    e_mb = partial_vapour_pressure(t_c, p_mb, qt_kgkg)
    pd_mb = p_mb - e_mb
    lv = latent_heat(t_c)

    dmu_dt = (entropy_dry_air(t_c, pd_mb)
              - entropy_water_vapour(t_c, e_mb)
              + (rl_kgkg > 1.e-10) * (1. + eeps * e_mb / pd_mb) * lv / t_k)
    return dmu_dt
