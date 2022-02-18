import numpy as np


def advection_term(hurr, data, j, i, t):
    # advection for any v-grid variable
    r_advection = (hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] * (data[j, i + 1, t] - data[j, i, t]) +
                   hurr['r_list_u'][i] * hurr['u'][j, i, t] * (data[j, i, t] - data[j, i - 1, t])) / (
                          2. * hurr['r_list_v'][i] * hurr['dr'])
    z_advection = (hurr['rho_initial_w'][j + 1] * hurr['w'][j + 1, i, t] * (data[j + 1, i, t] - data[j, i, t]) +
                   hurr['rho_initial_w'][j] * hurr['w'][j, i, t] * (data[j, i, t] - data[j - 1, i, t])) / (
                          2. * hurr['rho_initial_v'][j] * hurr['dz'])
    advection = r_advection + z_advection
    return advection


def theta_budget(hurr):
    t_deriv = (hurr['theta_after'] - hurr['theta_before']) / (2. * hurr['dt'])

    advection = np.zeros_like(hurr['v'])
    for t in range(hurr['timesteps']):
        for i in range(hurr['m'] - 1):
            for j in range(hurr['n'] - 1):
                advection[j, i, t] = advection_term(hurr, hurr['theta'], j, i, t)

    lagrangian_deriv = t_deriv + advection

    return lagrangian_deriv, hurr['micro_theta'] / (2. * hurr['dt']), hurr['precip_theta'] / (2. * hurr['dt']), hurr[
        'subgrid_theta'] / (2. * hurr['dt']), hurr['radcool'] / (2. * hurr['dt'])


def rv_budget(hurr):
    t_deriv = (hurr['rv_after'] - hurr['rv_before']) / (2. * hurr['dt'])

    advection = np.zeros_like(hurr['v'])
    for t in range(hurr['timesteps']):
        for i in range(hurr['m'] - 1):
            for j in range(hurr['n'] - 1):
                advection[j, i, t] = advection_term(hurr, hurr['rv'], j, i, t)

    lagrangian_deriv = t_deriv + advection

    return lagrangian_deriv, hurr['micro_rv'] / (2. * hurr['dt']), hurr['precip_rv'] / (2. * hurr['dt']), hurr[
        'subgrid_rv'] / (2. * hurr['dt'])


def rl_budget(hurr):
    t_deriv = (hurr['rl_after'] - hurr['rl_before']) / (2. * hurr['dt'])

    advection = np.zeros_like(hurr['v'])
    for t in range(hurr['timesteps']):
        for i in range(hurr['m'] - 1):
            for j in range(hurr['n'] - 1):
                advection[j, i, t] = advection_term(hurr, hurr['rl'], j, i, t)

    subgrid = hurr['subgrid_rl'] / (2. * hurr['dt'])
    max_advection = (hurr['rl'] / (2. * hurr['dt']) + subgrid)
    masked_advection = np.minimum(advection, max_advection)
    lagrangian_deriv = t_deriv + masked_advection

    return lagrangian_deriv, -hurr['micro_rv'] / (2. * hurr['dt']), hurr['precip_rl'] / (2. * hurr['dt']), subgrid


def rr_budget(hurr):
    t_deriv = (hurr['rr_after'] - hurr['rr_before']) / (2. * hurr['dt'])

    advection = np.zeros_like(hurr['v'])
    for t in range(hurr['timesteps']):
        for i in range(hurr['m'] - 1):
            for j in range(hurr['n'] - 1):
                advection[j, i, t] = advection_term(hurr, hurr['rr'], j, i, t)

    subgrid = hurr['subgrid_rr'] / (2. * hurr['dt'])
    max_advection = (hurr['rr'] / (2. * hurr['dt']) + subgrid)
    masked_advection = np.minimum(advection, max_advection)
    lagrangian_deriv = t_deriv + masked_advection

    return lagrangian_deriv, hurr['precip_rr'] / (2. * hurr['dt']), subgrid


def ri_budget(hurr):
    t_deriv = (hurr['ri_after'] - hurr['ri_before']) / (2. * hurr['dt'])

    advection = np.zeros_like(hurr['v'])
    for t in range(hurr['timesteps']):
        for i in range(hurr['m'] - 1):
            for j in range(hurr['n'] - 1):
                advection[j, i, t] = advection_term(hurr, hurr['ri'], j, i, t)

    subgrid = hurr['subgrid_ri'] / (2. * hurr['dt'])
    max_advection = (hurr['ri'] / (2. * hurr['dt']) + subgrid)
    masked_advection = np.minimum(advection, max_advection)
    lagrangian_deriv = t_deriv + masked_advection

    return lagrangian_deriv, hurr['precip_ri'] / (2. * hurr['dt']), subgrid


def theta_ei_budget(hurr, fixed_exner=None):
    lagr_theta, micro_theta, precip_theta, subgrid_theta, radcool_theta = theta_budget(hurr)
    lagr_rv, micro_rv, precip_rv, subgrid_rv = rv_budget(hurr)

    if fixed_exner is None:
        exner = np.dstack([np.tile(hurr['pi_initial'], (hurr['m'], 1)).transpose()] * hurr['timesteps']) + hurr['pi']
    else:
        exner = fixed_exner

    lagr_rl, micro_rl, precip_rl, subgrid_rl = rl_budget(hurr)
    lagr_rr, precip_rr, subgrid_rr = rr_budget(hurr)
    rv_coeff = hurr['Ls'] / (hurr['cp'] * exner)
    rl_coeff = hurr['Lf'] / (hurr['cp'] * exner)
    lagr_theta_ei = lagr_theta + rv_coeff * lagr_rv + rl_coeff * (lagr_rl + lagr_rr)
    micro_theta_ei = micro_theta + rv_coeff * micro_rv + rl_coeff * micro_rl
    precip_theta_ei = precip_theta + rv_coeff * precip_rv + rl_coeff * (precip_rl + precip_rr)
    subgrid_theta_ei = subgrid_theta + rv_coeff * subgrid_rv + rl_coeff * (subgrid_rl + subgrid_rr)

    vertical_flux_theta_e = hurr['vertical_flux_theta'][:-1, :, :] + rv_coeff * hurr['vertical_flux_rv'][:-1, :, :]
    # make same shape, will only use bottom level anyway

    return lagr_theta_ei, micro_theta_ei, precip_theta_ei, subgrid_theta_ei, vertical_flux_theta_e, radcool_theta


def rt_budget(hurr):
    lagr_v, _, precip_v, subgrid_v = rv_budget(hurr)
    lagr_l, _, precip_l, subgrid_l = rl_budget(hurr)
    lagr_r, precip_r, subgrid_r = rr_budget(hurr)
    lagr_i, precip_i, subgrid_i = ri_budget(hurr)
    lagr = lagr_v + lagr_l + lagr_r + lagr_i
    precip = precip_v + precip_l + precip_r + precip_i
    subgrid = subgrid_v + subgrid_l + subgrid_r + subgrid_i
    return lagr, precip, subgrid
