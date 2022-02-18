import numpy as np
from model_reader import read_fortran_output
from scipy import interpolate as sint
import thermo
import gma
import scipy.io as sio
import multiprocessing
from utils import safe_log, file_cleanup


def initial_state_interp(hurr):
    z_list = np.arange(0, hurr['dz'] * hurr['n'], hurr['dz']) + 0.5 * hurr['dz']
    z_list = np.hstack(([0], z_list))
    theta_data = np.hstack((hurr['theta_surface_initial'], hurr['theta_initial']))
    rv_data = np.hstack((hurr['rv_surface_initial'], hurr['rv_initial']))
    pi_surface = hurr['pi_surface_initial']
    pi_data = np.hstack((pi_surface, hurr['pi_initial']))
    theta_initial_interp = sint.InterpolatedUnivariateSpline(z_list, theta_data)
    rv_initial_interp = sint.InterpolatedUnivariateSpline(z_list, rv_data)
    pi_initial_interp = sint.InterpolatedUnivariateSpline(z_list, pi_data)
    z_list_fine = np.arange(27500)
    theta_rho_initial_fine = thermo.theta_rho_exact(theta_initial_interp(z_list_fine), rv_initial_interp(z_list_fine),
                                                    np.zeros_like(z_list_fine), np.zeros_like(z_list_fine))
    pi_initial_fine = pi_initial_interp(z_list_fine)
    initial_spec_vol_fine = thermo.spec_vol(pi_initial_fine, theta_rho_initial_fine)
    spec_vol_initial_interp = sint.InterpolatedUnivariateSpline(z_list_fine, initial_spec_vol_fine)
    pressure_mb_initial_interp = sint.InterpolatedUnivariateSpline(z_list_fine, thermo.pressure_from_PI(pi_initial_fine))
    return spec_vol_initial_interp, pressure_mb_initial_interp


def buoyancy_interp_parcel(hurr, theta_e, rt, ref_interps):
    spec_vol_initial_interp, pressure_mb_initial_interp = ref_interps
    z_list_fine = np.arange(27500)
    pressure_mb_initial_fine = pressure_mb_initial_interp(z_list_fine)
    spec_vol_initial_fine = spec_vol_initial_interp(z_list_fine)

    rt_kgkg_list = np.ones_like(spec_vol_initial_fine) * rt
    qt_kgkg_list = thermo.specific_humidity_from_mixing_ratio(rt_kgkg_list)

    theta_e_list = np.ones_like(spec_vol_initial_fine) * theta_e

    spec_vol_fine, t_K_fine = thermo.spec_vol_from_thetae_rt(theta_e_list, rt_kgkg_list, pressure_mb_initial_fine)
    buoyancy = thermo.buoyancy_spec_vol(spec_vol_fine, spec_vol_initial_fine)
    buoyancy_interp = sint.InterpolatedUnivariateSpline(z_list_fine, buoyancy)
    t_K_interp = sint.InterpolatedUnivariateSpline(z_list_fine, t_K_fine)
    e_mb_fine = gma.partial_vapour_pressure(t_K_fine-gma.t0, pressure_mb_initial_fine, qt_kgkg_list)
    es_mb_fine = gma.saturation_vapour_pressure(t_K_fine - gma.t0)
    rv_fine = np.minimum(rt_kgkg_list, gma.saturation_mixing_ratio(t_K_fine - gma.t0, pressure_mb_initial_fine))
    rl_fine = rt_kgkg_list - rv_fine
    rel_hum_fine = e_mb_fine / es_mb_fine
    pd_mb_fine = pressure_mb_initial_fine - e_mb_fine
    pd_mb_interp = sint.InterpolatedUnivariateSpline(z_list_fine, pd_mb_fine)
    rel_hum_interp = sint.InterpolatedUnivariateSpline(z_list_fine, rel_hum_fine)
    rv_interp = sint.InterpolatedUnivariateSpline(z_list_fine, rv_fine)
    rl_interp = sint.InterpolatedUnivariateSpline(z_list_fine, rl_fine)

    return buoyancy_interp, t_K_interp, pd_mb_interp, rel_hum_interp, pressure_mb_initial_interp, rv_interp, rl_interp, spec_vol_fine


def parcel_lnb(hurr, z_m, buoyancy_interp):
    lnb = None
    lnb_list = buoyancy_interp.roots()
    if buoyancy_interp(z_m) == 0.:
        lnb = z_m
    elif buoyancy_interp(z_m) < 0.:
        lnbs_below = lnb_list[np.where(lnb_list < z_m)]
        if lnbs_below.size > 0:
            lnb = np.max(lnbs_below)
        else:
            lnb = 0.
    elif buoyancy_interp(z_m) > 0:
        lnbs_above = lnb_list[np.where(lnb_list > z_m)]
        if lnbs_above.size > 0:
            lnb = np.min(lnbs_above)
        else:
            lnb = hurr['dz']*hurr['n']
    return lnb


def ape_density_parcel(hurr, theta_e, rt, j, ref_interps):
    buoyancy_interp, t_K_interp, pd_mb_interp, rel_hum_interp, pressure_mb_initial_interp, _, _, _ = buoyancy_interp_parcel(hurr, theta_e, rt, ref_interps)
    height = (j+0.5) * hurr['dz']
    reference_height = parcel_lnb(hurr, height, buoyancy_interp)
    if np.abs(height - reference_height) < 1e-4:
        ape_density = 0.
    else:
        ape_density = buoyancy_interp.integral(height, reference_height)
    t_K_at_ref_height = t_K_interp(reference_height)
    pd_mb_at_ref_height = pd_mb_interp(reference_height)
    rel_hum_at_ref_height = rel_hum_interp(reference_height)

    t_K_at_p0 = t_K_interp(height)
    pd_mb_at_p0 = pd_mb_interp(height)
    rel_hum_at_p0 = rel_hum_interp(height)

    cp_eff = gma.cpd + rt * gma.cl
    eff_theta_e = cp_eff * (t_K_at_p0 - t_K_at_ref_height) / theta_e

    eff_rt = (t_K_at_p0 * ((gma.cl - gma.cpd) * (1. + np.log(theta_e / t_K_at_p0)) + gma.rd * np.log(
        1000. / pd_mb_at_p0) + gma.rv*safe_log(rel_hum_at_p0)) - t_K_at_ref_height * (
                         (gma.cl - gma.cpd) * (1. + np.log(theta_e / t_K_at_ref_height)) + gma.rd * np.log(
                     1000. / pd_mb_at_ref_height)+ gma.rv*safe_log(rel_hum_at_ref_height)))/(1. + rt)**2

    if ape_density < 0.:
        print('APE density negative. theta_e = %f, rt = %f, j = %f, APE = %f, z_r = %f'%(theta_e, rt, j, ape_density, reference_height))

    return ape_density, reference_height, eff_theta_e, eff_rt

