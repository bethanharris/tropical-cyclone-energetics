import numpy as np
from tqdm import tqdm
from pathlib import Path
from model_reader import read_fortran_output
from ape_density import prepare_hurr
from diabatic_budgets import advection_term
from utils import multiply_by_rho


def get_ape_data(directory, run_id, ref_state, offsets=True):
    ape_at = np.load(f'{directory}/ape_{ref_state}_ref_at_timesteps_{run_id}.npz')
    if offsets:
        ape_after = np.load(f'{directory}/ape_{ref_state}_ref_after_timesteps_{run_id}.npz')
        ape_before = np.load(f'{directory}/ape_{ref_state}_ref_before_timesteps_{run_id}.npz')
        return ape_at, ape_before, ape_after
    else:
        return ape_at


def zr_variations_as_residual(directory, hurr, run_id, ref_state, ape_deriv, budget_sum):
    ape_at, ape_before, ape_after = get_ape_data(directory, run_id, ref_state)
    z_r = ape_at['z_r']
    t_change_zr = np.abs(ape_after['z_r'] - ape_before['z_r'])
    ape_at.close()
    ape_before.close()
    ape_after.close()

    z_change_zr = np.zeros_like(hurr['v'])
    r_change_zr = np.zeros_like(hurr['v'])
    for t in tqdm(range(hurr['timesteps']), desc='discontinuities'):
        for i in range(hurr['m']):
            z_change_zr[0, i, t] = np.abs(z_r[0, i, t] - z_r[1, i, t])
            z_change_zr[hurr['n'] - 1, i, t] = np.abs(z_r[hurr['n'] - 1, i, t] - z_r[hurr['n'] - 2, i, t])
            for j in range(1, hurr['n'] - 1):
                z_change_zr[j, i, t] = max(np.abs(z_r[j, i, t] - z_r[j - 1, i, t]),
                                           np.abs(z_r[j, i, t] - z_r[j + 1, i, t]))
        for j in range(hurr['n']):
            r_change_zr[j, 0, t] = np.abs(z_r[j, 0, t] - z_r[j, 1, t])
            r_change_zr[j, hurr['m'] - 1, t] = np.abs(z_r[j, hurr['m'] - 1, t] - z_r[j, hurr['m'] - 2, t])
            for i in range(1, hurr['m'] - 1):
                r_change_zr[j, i, t] = max(np.abs(z_r[j, i, t] - z_r[j, i - 1, t]),
                                           np.abs(z_r[j, i, t] - z_r[j, i + 1, t]))

    time_discontinuous = t_change_zr > hurr['dz']
    space_discontinuous = np.logical_or(r_change_zr > 10. * hurr['dz'], z_change_zr > 10. * hurr['dz'])
    budget_residual = ape_deriv - budget_sum
    time_discontinuity = time_discontinuous * budget_residual
    budget_residual -= time_discontinuity
    space_discontinuity = space_discontinuous * budget_residual
    return time_discontinuity, space_discontinuity


def compute_budget_terms(directory, hurr, run_id, ref_state):
    ape_at, ape_before, ape_after = get_ape_data(directory, run_id, ref_state)

    ape_density = ape_at['ape_density']
    G_theta_e = ape_at['G_theta_e']
    G_rt = ape_at['G_rt']
    ape_at.close()
    ape_deriv = multiply_by_rho(hurr, (ape_after['ape_density'] - ape_before['ape_density']) / (2. * hurr['dt']))
    ape_before.close()
    ape_after.close()

    prepare_hurr(hurr, 'at')
    hurr['rt'] = hurr['rv'] + hurr['rl'] + hurr['rr'] + hurr['ri']
    hurr['rt_before'] = hurr['rv_before'] + hurr['rl_before'] + hurr['rr_before'] + hurr['ri_before']
    hurr['rt_after'] = hurr['rv_after'] + hurr['rl_after'] + hurr['rr_after'] + hurr['ri_after']

    theta_deriv = (hurr['theta_after'] - hurr['theta_before']) / (2. * hurr['dt'])
    rv_deriv = (hurr['rv_after'] - hurr['rv_before']) / (2. * hurr['dt'])
    rl_deriv = (hurr['rl_after'] - hurr['rl_before']) / (2. * hurr['dt'])
    rr_deriv = (hurr['rr_after'] - hurr['rr_before']) / (2. * hurr['dt'])
    rt_deriv = (hurr['rt_after'] - hurr['rt_before']) / (2. * hurr['dt'])

    theta_advection = np.zeros_like(hurr['v'])
    rv_advection = np.zeros_like(hurr['v'])
    rl_advection = np.zeros_like(hurr['v'])
    rr_advection = np.zeros_like(hurr['v'])
    ri_advection = np.zeros_like(hurr['v'])
    ape_src = np.zeros_like(hurr['v'])
    ape_flux = np.zeros_like(hurr['v'])

    for t in tqdm(range(hurr['timesteps']), desc='budget timesteps'):
        for i in range(hurr['m'] - 1):
            for j in range(hurr['n'] - 1):
                theta_advection[j, i, t] = advection_term(hurr, hurr['theta'], j, i, t)
                rv_advection[j, i, t] = advection_term(hurr, hurr['rv'], j, i, t)
                rl_advection[j, i, t] = advection_term(hurr, hurr['rl'], j, i, t)
                rr_advection[j, i, t] = advection_term(hurr, hurr['rr'], j, i, t)
                ri_advection[j, i, t] = advection_term(hurr, hurr['ri'], j, i, t)
                ape_src[j, i, t] = ape_density[j, i, t] * (hurr['rho_initial_v'][j] * (
                            hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] - hurr['r_list_u'][i] * hurr['u'][
                        j, i, t]) / (hurr['r_list_v'][i] * hurr['dr']) + (hurr['rho_initial_w'][j + 1] * hurr['w'][
                    j + 1, i, t] - hurr['rho_initial_w'][j] * hurr['w'][j, i, t]) / hurr['dz'])
                ape_flux_r = hurr['rho_initial_v'][j] * (hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] * (
                            ape_density[j, i + 1, t] + ape_density[j, i, t]) - hurr['r_list_u'][i] * hurr['u'][
                                                             j, i, t] * (ape_density[j, i, t] + ape_density[
                    j, i - 1, t])) / (2. * hurr['r_list_v'][i] * hurr['dr'])
                ape_flux_z = (hurr['rho_initial_w'][j + 1] * hurr['w'][j + 1, i, t] * (
                            ape_density[j + 1, i, t] + ape_density[j, i, t]) - hurr['rho_initial_w'][j] * hurr['w'][
                                  j, i, t] * (ape_density[j, i, t] + ape_density[j - 1, i, t])) / (2. * hurr['dz'])
                ape_flux[j, i, t] = ape_flux_r + ape_flux_z

    max_rl_advection = (hurr['rl'] + hurr['subgrid_rl']) / (2. * hurr['dt'])
    masked_rl_advection = np.minimum(rl_advection, max_rl_advection)

    max_rr_advection = (hurr['rr'] + hurr['subgrid_rr']) / (2. * hurr['dt'])
    masked_rr_advection = np.minimum(rr_advection, max_rr_advection)
    max_ri_advection = (hurr['ri'] + hurr['subgrid_ri']) / (2. * hurr['dt'])
    masked_ri_advection = np.minimum(ri_advection, max_ri_advection)
    theta_e_deriv = theta_deriv + hurr['Ls'] * rv_deriv / (hurr['cp'] * hurr['mean_exner']) + hurr['Lf'] * (
                rl_deriv + rr_deriv) / (hurr['cp'] * hurr['mean_exner'])
    theta_e_advection = theta_advection + hurr['Ls'] * rv_advection / (hurr['cp'] * hurr['mean_exner']) + hurr['Lf'] * (
                masked_rl_advection + masked_rr_advection) / (hurr['cp'] * hurr['mean_exner'])
    theta_e_lagr = multiply_by_rho(hurr, theta_e_deriv + theta_e_advection)
    rt_lagr = multiply_by_rho(hurr,
                              rt_deriv + rv_advection + masked_rl_advection + masked_rr_advection + masked_ri_advection)

    theta_e_production = G_theta_e * theta_e_lagr
    rt_production = G_rt * rt_lagr

    if ref_state == 'initial':
        with np.load(f'{directory}/kinetic_elastic_energy_budgets_{run_id}.npz') as budget_data:
            buoyancy_flux = budget_data['buoyancy']
    else:
        raise KeyError('Invalid reference state type. Only initial reference state currently supported.')

    budget_sum = theta_e_production + rt_production + ape_src - ape_flux - buoyancy_flux
    time_change_zr, spatial_change_zr = zr_variations_as_residual(directory, hurr, run_id,
                                                                  ref_state, ape_deriv, budget_sum)
    return (theta_e_production, rt_production, ape_deriv, ape_src, ape_flux, buoyancy_flux,
                time_change_zr, spatial_change_zr)


def save_budget(directory, hurr, run_id, ref_state):
    (theta_e_production, rt_production, ape_deriv, ape_src, ape_flux, buoyancy_flux,
     time_change_zr, spatial_change_zr) = compute_budget_terms(directory, hurr, run_id, ref_state)
    ape_budget = {'ape_deriv': ape_deriv,
                  'ape_src': ape_src,
                  'ape_flux': ape_flux,
                  'theta_e_prod': theta_e_production,
                  'rt_prod': rt_production,
                  'buoyancy_flux': buoyancy_flux,
                  'zr_change_time': time_change_zr,
                  'zr_change_space': spatial_change_zr}
    np.savez(f'{directory}/ape_{ref_state}_ref_budget_{run_id}.npz', **ape_budget)


if __name__=='__main__':
    data_dir = '../data/J30pt3'
    run_id = 'J30pt3'
    hurr = read_fortran_output(data_dir)
    save_budget(data_dir, hurr, run_id, 'initial')
