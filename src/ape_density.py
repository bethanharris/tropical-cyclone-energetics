import numpy as np
import sys
import multiprocessing
from model_reader import read_fortran_output
import thermo
import gma
from utils import file_cleanup, construct_full_profile, multiply_by_rho


def initial_ref_state(hurr):
    theta_ref = construct_full_profile(hurr, hurr['theta_initial'], hurr['theta_surface_initial'],
                                       hurr['theta_top_initial'])
    rv_ref = construct_full_profile(hurr, hurr['rv_initial'], hurr['rv_surface_initial'], hurr['rv_top_initial'])
    pi_ref = construct_full_profile(hurr, hurr['pi_initial'], hurr['pi_surface_initial'], hurr['pi_top_initial'])
    pressure_mb_ref = thermo.pressure_from_PI(pi_ref)
    return theta_ref, rv_ref, pressure_mb_ref


def theta_e_fix_exner(hurr, t_c, p_mb, rt_kgkg, fixed_exner):
    rv_kgkg, rl_kgkg, ri_kgkg = gma.all_mixing_ratios(t_c, p_mb, rt_kgkg)
    theta_term = (t_c + gma.t0) * (1000. / p_mb) ** (hurr['Rd'] / hurr['cp'])
    rv_term = hurr['Ls'] * rv_kgkg / (hurr['cp'] * fixed_exner)
    rl_term = hurr['Lf'] * rl_kgkg / (hurr['cp'] * fixed_exner)
    theta_e = theta_term + rv_term + rl_term
    return theta_e


def decompose_theta_e_rt(hurr, theta_e, rt, p_mb, fixed_exner):
    if isinstance(theta_e, np.float32) or isinstance(theta_e, np.float64) or isinstance(theta_e, float):
        theta_e = np.array([theta_e])
        p_mb = np.array([p_mb])
        rt = np.array([rt])

    dt_bound = 1.e-8  # desired precision of final lifted temperature

    lower_bound = -274.
    upper_bound = 500.  # Upper and lower ends of initial range to bisect

    low = np.full(theta_e.shape, lower_bound)
    high = np.full(theta_e.shape, upper_bound)

    # Find number of bisection steps required to reach desired accuracy
    steps = np.ceil(np.log2((upper_bound - lower_bound) / dt_bound))

    # Bisect
    for _ in np.arange(0, steps):
        mid = (low + high) / 2.
        theta_e_mid = theta_e_fix_exner(hurr, mid, p_mb, rt, fixed_exner)
        delta_theta_e_mid = theta_e_mid - theta_e
        pos = np.where(delta_theta_e_mid > 0.)
        neg = np.where(delta_theta_e_mid < 0.)
        low[neg] = mid[neg]
        high[pos] = mid[pos]
        mid_new = (low + high) / 2.

    t2_c = mid_new

    t_K = t2_c + gma.t0
    theta = thermo.theta_from_T(t_K, p_mb)
    rv_kgkg, rl_kgkg, ri_kgkg = gma.all_mixing_ratios(t2_c, p_mb, rt)
    saturated = rl_kgkg + ri_kgkg > 1.e-7
    frozen = ri_kgkg > 1.e-7
    return theta, rv_kgkg, rl_kgkg, ri_kgkg, t_K, saturated, frozen


def buoyancy_lifted_parcel(hurr, theta_e, rt, fixed_exner, theta_ref, rv_ref, pressure_mb_ref):
    rt_kgkg_list = np.ones_like(rv_ref) * rt
    theta_e_list = np.ones_like(rv_ref) * theta_e
    fixed_exner_list = np.ones_like(rv_ref) * fixed_exner

    theta_lift, rv_lift, rl_lift, ri_lift, t_K_lift, saturated, frozen = decompose_theta_e_rt(hurr, theta_e_list,
                                                                                              rt_kgkg_list,
                                                                                              pressure_mb_ref,
                                                                                              fixed_exner_list)
    buoyancy_lift = hurr['g'] * ((theta_lift - theta_ref) / theta_ref + 0.61 * (rv_lift - rv_ref) - rl_lift - ri_lift)

    saturated_levels = np.where(saturated)[0]
    frozen_levels = np.where(frozen)[0]
    z_list = np.arange(0, hurr['dz'] * hurr['n'] + 1, hurr['dz'] / 2.)
    if saturated_levels.size == 0:
        saturation_level = hurr['n'] * hurr['dz']
    else:
        saturation_level = z_list[saturated_levels[0]]
    if frozen_levels.size == 0:
        freezing_level = hurr['n'] * hurr['dz']
    else:
        freezing_level = z_list[frozen_levels[0]]
    return buoyancy_lift, saturation_level, freezing_level, t_K_lift, theta_lift, rv_lift


def parcel_lnb(hurr, buoyancy, j):
    # j should be on v grid as only v grid parcels get APE calculate
    height_idx = j
    lnb = None
    z_list = np.arange(0, hurr['dz'] * hurr['n'], hurr['dz']) + 0.5 * hurr['dz']
    z_m = z_list[height_idx]
    buoyancy_v = buoyancy[1::2]
    sign_changes = np.where(np.diff(np.sign(buoyancy_v)) != 0)
    lnb_list = np.array([z_list[level] - (z_list[level + 1] - z_list[level]) * buoyancy_v[level] / (
                buoyancy_v[level + 1] - buoyancy_v[level]) for level in sign_changes])
    if buoyancy_v[height_idx] == 0.:
        lnb = z_m
    elif buoyancy_v[height_idx] < 0.:
        lnbs_below = lnb_list[np.where(lnb_list < z_m)]
        if lnbs_below.size > 0:
            lnb = np.max(lnbs_below)
        else:
            lnb = 0.
    elif buoyancy_v[height_idx] > 0:
        lnbs_above = lnb_list[np.where(lnb_list > z_m)]
        if lnbs_above.size > 0:
            lnb = np.min(lnbs_above)
        else:
            lnb = hurr['dz'] * hurr['n']
    return lnb


def weighted_average_exner(hurr):
    hurr['exner'] = np.dstack([np.tile(hurr['pi_initial'], (hurr['m'], 1)).transpose()] * hurr['timesteps']) + hurr[
        'pi']
    rho = multiply_by_rho(hurr, np.ones_like(hurr['v']))
    hurr['mean_exner'] = np.average(hurr['exner'], weights=rho)


def prepare_hurr(hurr, timestep):
    weighted_average_exner(hurr)
    if timestep == 'at':
        hurr['approx_theta_e'] = hurr['theta'] + hurr['Ls'] * hurr['rv'] / (hurr['cp'] * hurr['mean_exner']) + hurr[
            'Lf'] * (hurr['rl'] + hurr['rr']) / (hurr['cp'] * hurr['mean_exner'])
        hurr['rt'] = hurr['rv'] + hurr['rl'] + hurr['rr'] + hurr['ri']
    else:
        hurr['approx_theta_e'] = hurr['theta_' + timestep] + hurr['Ls'] * hurr['rv_' + timestep] / (
                    hurr['cp'] * hurr['mean_exner']) + hurr['Lf'] * (
                                             hurr['rl_' + timestep] + hurr['rr_' + timestep]) / (
                                             hurr['cp'] * hurr['mean_exner'])
        hurr['rt'] = hurr['rv_' + timestep] + hurr['rl_' + timestep] + hurr['rr_' + timestep] + hurr['ri_' + timestep]


def ape_density_parcel(hurr, j, i, t, ref_profiles, ref_state, nudge_theta_e=0., nudge_rt=0., fix_params=None,
                       fix_exner=None):
    if fix_params is not None:
        theta_e, rt = fix_params
    else:
        theta_e = hurr['approx_theta_e'][j, i, t]
        rt = hurr['rt'][j, i, t]
    theta_e += nudge_theta_e
    rt += nudge_rt  # nudges/fixing for testing purposes only
    if fix_exner is not None:
        fixed_exner = fix_exner
    else:
        fixed_exner = hurr['mean_exner']
    if ref_state == 'initial':
        theta_ref, rv_ref, pressure_mb_ref = ref_profiles
    else:
        raise KeyError('Invalid reference state type. Only initial reference state currently supported.')

    (buoyancy_lift, saturation_height, freezing_height,
     t_K_lift, theta_lift, rv_lift) = buoyancy_lifted_parcel(hurr, theta_e, rt, fixed_exner, theta_ref, rv_ref,
                                                             pressure_mb_ref)
    height = (j + 0.5) * hurr['dz']
    height_idx = 2 * j + 1
    reference_height = parcel_lnb(hurr, buoyancy_lift, j)
    ape_density = 0.
    eff_theta_e = 0.
    eff_rt = 0.
    if height < reference_height:
        level_height = height
        while level_height + hurr['dz'] < reference_height:
            ape_density += hurr['dz'] * buoyancy_lift[height_idx + 1]
            if level_height + 0.5 * hurr['dz'] < saturation_height:
                eff_theta_e += hurr['dz'] * hurr['g'] / theta_ref[height_idx + 1]
                eff_rt += hurr['g'] * (0.61 - hurr['Ls'] / (
                        hurr['cp'] * theta_ref[height_idx + 1] * fixed_exner)) * hurr['dz']
            elif level_height + 0.5 * hurr['dz'] < freezing_height:
                t_K = t_K_lift[height_idx + 1]
                theta = theta_lift[height_idx + 1]
                rv = rv_lift[height_idx + 1]
                num = 1. + hurr['Lv'] * rv / (hurr['Rd'] * t_K)
                denom = 1 + 0.622 * (hurr['Lv'] ** 2 * rv) / (hurr['cp'] * hurr['Rd'] * t_K * theta * fixed_exner)
                saturation_factor = num / denom
                eff_theta_e += hurr['dz'] * saturation_factor * hurr['g'] / theta_ref[height_idx + 1]
                eff_rt += -hurr['g'] * hurr['dz'] * (1. + hurr['Lf'] * saturation_factor / (
                            hurr['cp'] * fixed_exner * theta_ref[height_idx + 1]))
            else:
                t_K = t_K_lift[height_idx + 1]
                theta = theta_lift[height_idx + 1]
                rv = rv_lift[height_idx + 1]
                num = 1. + hurr['Ls'] * rv / (hurr['Rd'] * t_K)
                denom = 1 + 0.622 * (hurr['Ls'] ** 2 * rv) / (hurr['cp'] * hurr['Rd'] * t_K * theta * fixed_exner)
                frozen_factor = num / denom
                eff_theta_e += hurr['dz'] * frozen_factor * hurr['g'] / theta_ref[height_idx + 1]
                eff_rt += -hurr['g'] * hurr['dz']
            level_height += hurr['dz']
            height_idx += 2
        height_fraction = (reference_height - level_height) / hurr['dz']
        ape_density += buoyancy_lift[height_idx + 1] * height_fraction * hurr['dz']
        if level_height + 0.5 * hurr['dz'] < saturation_height:
            eff_theta_e += height_fraction * hurr['dz'] * hurr['g'] / theta_ref[height_idx + 1]
            eff_rt += height_fraction * hurr['g'] * (0.61 - hurr['Ls'] / (
                    hurr['cp'] * theta_ref[height_idx + 1] * fixed_exner)) * hurr['dz']
        elif level_height + 0.5 * hurr['dz'] < freezing_height:
            t_K = t_K_lift[height_idx + 1]
            theta = theta_lift[height_idx + 1]
            rv = rv_lift[height_idx + 1]
            num = 1. + hurr['Lv'] * rv / (hurr['Rd'] * t_K)
            denom = 1 + 0.622 * (hurr['Lv'] ** 2 * rv) / (hurr['cp'] * hurr['Rd'] * t_K * theta * fixed_exner)
            saturation_factor = num / denom
            eff_theta_e += height_fraction * hurr['dz'] * saturation_factor * hurr['g'] / theta_ref[height_idx + 1]
            eff_rt += -height_fraction * hurr['g'] * hurr['dz'] * (1. + hurr['Lf'] * saturation_factor / (
                    hurr['cp'] * fixed_exner * theta_ref[height_idx + 1]))
        else:
            t_K = t_K_lift[height_idx + 1]
            theta = theta_lift[height_idx + 1]
            rv = rv_lift[height_idx + 1]
            num = 1. + hurr['Ls'] * rv / (hurr['Rd'] * t_K)
            denom = 1 + 0.622 * (hurr['Ls'] ** 2 * rv) / (hurr['cp'] * hurr['Rd'] * t_K * theta * fixed_exner)
            frozen_factor = num / denom
            eff_theta_e += height_fraction * hurr['dz'] * frozen_factor * hurr['g'] / theta_ref[height_idx + 1]
            eff_rt += -height_fraction * hurr['g'] * hurr['dz']

    elif height > reference_height:
        level_height = height
        while level_height - hurr['dz'] > reference_height:
            ape_density -= hurr['dz'] * buoyancy_lift[height_idx - 1]
            if level_height - 0.5 * hurr['dz'] < saturation_height:
                eff_theta_e -= hurr['dz'] * hurr['g'] / theta_ref[height_idx - 1]
                eff_rt -= hurr['g'] * (0.61 - hurr['Ls'] / (
                        hurr['cp'] * theta_ref[height_idx - 1] * fixed_exner)) * hurr['dz']
            elif level_height - 0.5 * hurr['dz'] < freezing_height:
                t_K = t_K_lift[height_idx - 1]
                theta = theta_lift[height_idx - 1]
                rv = rv_lift[height_idx - 1]
                num = 1. + hurr['Lv'] * rv / (hurr['Rd'] * t_K)
                denom = 1 + 0.622 * (hurr['Lv'] ** 2 * rv) / (hurr['cp'] * hurr['Rd'] * t_K * theta * fixed_exner)
                saturation_factor = num / denom
                eff_theta_e -= hurr['dz'] * saturation_factor * hurr['g'] / theta_ref[height_idx - 1]
                eff_rt -= -hurr['g'] * hurr['dz'] * (1. + hurr['Lf'] * saturation_factor / (
                            hurr['cp'] * fixed_exner * theta_ref[height_idx - 1]))
            else:
                t_K = t_K_lift[height_idx - 1]
                theta = theta_lift[height_idx - 1]
                rv = rv_lift[height_idx - 1]
                num = 1. + hurr['Ls'] * rv / (hurr['Rd'] * t_K)
                denom = 1 + 0.622 * (hurr['Ls'] ** 2 * rv) / (hurr['cp'] * hurr['Rd'] * t_K * theta * fixed_exner)
                frozen_factor = num / denom
                eff_theta_e -= hurr['dz'] * frozen_factor * hurr['g'] / theta_ref[height_idx - 1]
                eff_rt -= -hurr['g'] * hurr['dz']
            level_height -= hurr['dz']
            height_idx -= 2
        height_fraction = (level_height - reference_height) / hurr['dz']
        ape_density -= buoyancy_lift[height_idx - 1] * height_fraction * hurr['dz']
        if level_height - 0.5 * hurr['dz'] < saturation_height:
            eff_theta_e -= height_fraction * hurr['dz'] * hurr['g'] / theta_ref[height_idx - 1]
            eff_rt -= height_fraction * hurr['g'] * (0.61 - hurr['Ls'] / (
                    hurr['cp'] * theta_ref[height_idx - 1] * fixed_exner)) * hurr['dz']
        elif level_height - 0.5 * hurr['dz'] < freezing_height:
            t_K = t_K_lift[height_idx - 1]
            theta = theta_lift[height_idx - 1]
            rv = rv_lift[height_idx - 1]
            num = 1. + hurr['Lv'] * rv / (hurr['Rd'] * t_K)
            denom = 1 + 0.622 * (hurr['Lv'] ** 2 * rv) / (hurr['cp'] * hurr['Rd'] * t_K * theta * fixed_exner)
            saturation_factor = num / denom
            eff_theta_e -= height_fraction * hurr['dz'] * saturation_factor * hurr['g'] / theta_ref[height_idx - 1]
            eff_rt -= -height_fraction * hurr['g'] * hurr['dz'] * (1. + hurr['Lf'] * saturation_factor / (
                    hurr['cp'] * fixed_exner * theta_ref[height_idx - 1]))
        else:
            t_K = t_K_lift[height_idx - 1]
            theta = theta_lift[height_idx - 1]
            rv = rv_lift[height_idx - 1]
            num = 1. + hurr['Ls'] * rv / (hurr['Rd'] * t_K)
            denom = 1 + 0.622 * (hurr['Ls'] ** 2 * rv) / (hurr['cp'] * hurr['Rd'] * t_K * theta * fixed_exner)
            frozen_factor = num / denom
            eff_theta_e -= height_fraction * hurr['dz'] * frozen_factor * hurr['g'] / theta_ref[height_idx - 1]
            eff_rt -= -height_fraction * hurr['g'] * hurr['dz']

    return ape_density, reference_height, eff_theta_e, eff_rt


def ape_from_indices(params):
    global hurr
    global ref_profiles
    global reference_state
    j, i, t = params
    ape_density, reference_height, eff_theta_e, eff_rt = ape_density_parcel(hurr, j, i, t, ref_profiles,
                                                                            reference_state)
    return ape_density, reference_height, eff_theta_e, eff_rt


def get_all_ape(hurr, directory, timestep, run_id, ref_state):
    for t in range(hurr['timesteps']):
        ape_dict = {
            'ape_density': np.zeros((hurr['n'], hurr['m'])),
            'z_r': np.zeros((hurr['n'], hurr['m'])),
            'G_theta_e': np.zeros((hurr['n'], hurr['m'])),
            'G_rt': np.zeros((hurr['n'], hurr['m']))
        }
        i_list = range(hurr['m'])
        j_list = range(hurr['n'])
        J, I = np.meshgrid(j_list, i_list)
        T = np.ones_like(J) * t
        params = zip(J.ravel(), I.ravel(), T.ravel())
        pool = multiprocessing.Pool(8)
        ape_data = pool.map(ape_from_indices, params)
        pool.terminate()
        pool.join()
        pool.close()
        all_data = np.array(ape_data)
        ape_dict['ape_density'] = np.reshape(all_data[:, 0], (hurr['m'], hurr['n'])).transpose()
        ape_dict['z_r'] = np.reshape(all_data[:, 1], (hurr['m'], hurr['n'])).transpose()
        ape_dict['G_theta_e'] = np.reshape(all_data[:, 2], (hurr['m'], hurr['n'])).transpose()
        ape_dict['G_rt'] = np.reshape(all_data[:, 3], (hurr['m'], hurr['n'])).transpose()
        np.savez(f'{directory}/ape_{ref_state}_ref_{timestep}_time_{t}_{run_id}.npz', **ape_dict)


def compile_timestep_files(hurr, directory, timestep, run_id, ref_state):
    ape_all = {
        'ape_density': np.zeros((hurr['n'], hurr['m'], hurr['timesteps'])),
        'z_r': np.zeros((hurr['n'], hurr['m'], hurr['timesteps'])),
        'G_theta_e': np.zeros((hurr['n'], hurr['m'], hurr['timesteps'])),
        'G_rt': np.zeros((hurr['n'], hurr['m'], hurr['timesteps']))
    }
    for t in range(hurr['timesteps']):
        data = np.load(f'{directory}/ape_{ref_state}_ref_{timestep}_time_{t}_{run_id}.npz')
        for key in ape_all.keys():
            ape_all[key][:, :, t] = data[key]

    np.savez(f'{directory}/ape_{ref_state}_ref_{timestep}_timesteps_{run_id}.npz', **ape_all)
    file_cleanup(directory, [f'ape_{ref_state}_ref_{timestep}_time_'])


def run_ape(directory, timestep, run_id, ref_state):
    global hurr
    global ref_profiles
    prepare_hurr(hurr, timestep)
    get_all_ape(hurr, directory, timestep, run_id, ref_state)
    compile_timestep_files(hurr, directory, timestep, run_id, ref_state)


if __name__ == '__main__':
    reference_state = str(sys.argv[1])  # e.g. run <<python ape_density.py initial>> to compute for initial ref state
    print(reference_state)
    data_dir = '../data/J30pt3'
    run_id = 'J30pt3'
    hurr = read_fortran_output(data_dir)
    if reference_state == 'initial':
        ref_profiles = initial_ref_state(hurr)
    else:
        raise KeyError('Invalid reference state type. Only initial reference state currently supported.')
    run_ape(data_dir, 'before', run_id, reference_state)
    run_ape(data_dir, 'at', run_id, reference_state)
    run_ape(data_dir, 'after', run_id, reference_state)
