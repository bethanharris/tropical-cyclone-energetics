from model_reader import read_fortran_output
import numpy as np
from tqdm import tqdm


def budget_parcel_at_time(hurr, i, j, t):
    buoyancy = 0.5 * hurr['g'] * (
            hurr['rho_initial_w'][j + 1] * hurr['w'][j + 1, i, t] * (
            (hurr['theta'][j + 1, i, t] + hurr['theta'][j, i, t]) / (
            hurr['theta_initial'][j + 1] + hurr['theta_initial'][j]) - 1 + 0.305 * (
                    hurr['rv'][j + 1, i, t] + hurr['rv'][j, i, t] - hurr['rv_initial'][j + 1] - hurr['rv_initial'][j])
            - 0.5 * (hurr['rl'][j + 1, i, t] + hurr['rl'][j, i, t] + hurr['ri'][j + 1, i, t] + hurr['ri'][j, i, t] +
                     hurr['rr'][j + 1, i, t] + hurr['rr'][j, i, t])
    )
            + hurr['rho_initial_w'][j] * hurr['w'][j, i, t] * (
                    (hurr['theta'][j, i, t] + hurr['theta'][j - 1, i, t]) / (
                    hurr['theta_initial'][j] + hurr['theta_initial'][j - 1]) - 1
                    + 0.305 * (hurr['rv'][j, i, t] + hurr['rv'][j - 1, i, t] - hurr['rv_initial'][j] -
                               hurr['rv_initial'][
                                   j - 1])
                    - 0.5 * (hurr['rl'][j, i, t] + hurr['rl'][j - 1, i, t] + hurr['ri'][j, i, t] + hurr['ri'][
                j - 1, i, t] +
                             hurr['rr'][j, i, t] + hurr['rr'][j - 1, i, t])
            )
    )

    time_change_ek_u = (0.5 / hurr['dt']) * (
            0.5 * hurr['rho_initial_v'][j] * (
            hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] * (
            hurr['u_after'][j, i + 1, t] - hurr['u_before'][j, i + 1, t])
            + hurr['r_list_u'][i] * hurr['u'][j, i, t] * (hurr['u_after'][j, i, t] - hurr['u_before'][j, i, t])) /
            hurr['r_list_v'][i])

    time_change_ek_v = (0.5 / hurr['dt']) * (
            hurr['rho_initial_v'][j] * hurr['v'][j, i, t] * (hurr['v_after'][j, i, t] - hurr['v_before'][j, i, t]))

    time_change_ek_w = (0.5 / hurr['dt']) * (0.5 * (
            hurr['rho_initial_w'][j + 1] * hurr['w'][j + 1, i, t] * (
            hurr['w_after'][j + 1, i, t] - hurr['w_before'][j + 1, i, t])
            + hurr['rho_initial_w'][j] * hurr['w'][j, i, t] * (
                    hurr['w_after'][j, i, t] - hurr['w_before'][j, i, t]))
                                             )
    theta_v_initial = hurr['theta_initial'][j] * (1. + 0.61 * hurr['rv_initial'][j])
    aee_coeff = hurr['rho_initial_v'][j] * hurr['cp'] ** 2 * (theta_v_initial) ** 2 / hurr['c2']

    time_change_ee = aee_coeff * hurr['pi'][j, i, t] * (hurr['pi_after'][j, i, t] - hurr['pi_before'][j, i, t]) / (
            2. * hurr['dt'])

    pressure_div_r = hurr['cp'] * hurr['rho_initial_v'][j] * theta_v_initial * (
            hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] * (hurr['pi'][j, i + 1, t] + hurr['pi'][j, i, t]) -
            hurr['r_list_u'][i] * hurr['u'][j, i, t] * (hurr['pi'][j, i, t] + hurr['pi'][j, i - 1, t])
    ) / (2. * hurr['r_list_v'][i] * hurr['dr'])
    pressure_div_z = hurr['cp'] * (
            hurr['rho_thetav_initial_w'][j + 1] * hurr['w'][j + 1, i, t] * (
            hurr['pi'][j + 1, i, t] + hurr['pi'][j, i, t]) -
            hurr['rho_thetav_initial_w'][j] * hurr['w'][j, i, t] * (hurr['pi'][j, i, t] + hurr['pi'][j - 1, i, t])
    ) / (2. * hurr['dz'])
    pressure_div = pressure_div_r + pressure_div_z

    pressure_grad_u = hurr['cp'] * hurr['rho_initial_v'][j] * theta_v_initial * (
            hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] * (hurr['pi'][j, i + 1, t] - hurr['pi'][j, i, t]) +
            hurr['r_list_u'][i] * hurr['u'][j, i, t] * (hurr['pi'][j, i, t] - hurr['pi'][j, i - 1, t])
    ) / (2. * hurr['r_list_v'][i] * hurr['dr'])

    pressure_grad_w = hurr['cp'] * (
            hurr['rho_thetav_initial_w'][j + 1] * hurr['w'][j + 1, i, t] * (
            hurr['pi'][j + 1, i, t] - hurr['pi'][j, i, t]) +
            hurr['rho_thetav_initial_w'][j] * hurr['w'][j, i, t] * (hurr['pi'][j, i, t] - hurr['pi'][j - 1, i, t])
    ) / (2. * hurr['dz'])

    dthetav_subgrid = (hurr['dts'] * hurr['c2']) / (
                2. * hurr['dt'] * hurr['cp'] * (hurr['theta_initial'][j] + 0.61 * hurr['rv_initial'][j]) ** 2) * (
                                  (1. + 0.61 * hurr['rv'][j, i, t]) * (
                                      hurr['radcool'][j, i, t] + hurr['subgrid_theta'][j, i, t]) + 0.61 * hurr['theta'][
                                      j, i, t] * hurr['subgrid_rv'][j, i, t])

    pressure_dthetav = 2. * (hurr['dt'] / hurr['dts']) * aee_coeff * hurr['pi'][j, i, t] * (
                hurr['dthetav'][j, i, t] + dthetav_subgrid) / (2. * hurr['dt'])

    mass_source_u = (hurr['rho_initial_v'][j] * (
            hurr['u'][j, i + 1, t] * hurr['u'][j, i + 1, t] * (
            hurr['r_list_u'][i + 2] * hurr['u'][j, i + 2, t] - hurr['r_list_u'][i] * hurr['u'][j, i, t]) +
            hurr['u'][j, i, t] * hurr['u'][j, i, t] * (
                    hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] - hurr['r_list_u'][i - 1] * hurr['u'][
                j, i - 1, t])) / (4. * hurr['dr'])

                     + (hurr['rho_initial_w'][j + 1] * (hurr['u'][j, i + 1, t] * hurr['u'][j, i + 1, t] * (
                    hurr['r_list_v'][i + 1] * hurr['w'][j + 1, i + 1, t] + hurr['r_list_v'][i] * hurr['w'][
                j + 1, i, t]) + hurr['u'][j, i, t] * hurr['u'][j, i, t] * (
                                                                hurr['r_list_v'][i] * hurr['w'][j + 1, i, t] +
                                                                hurr['r_list_v'][
                                                                    i - 1] * hurr['w'][
                                                                    j + 1, i - 1, t])
                                                        )
                        - hurr['rho_initial_w'][j] * (
                                hurr['u'][j, i + 1, t] * hurr['u'][j, i + 1, t] * (
                                hurr['r_list_v'][i + 1] * hurr['w'][j, i + 1, t] + hurr['r_list_v'][i] * hurr['w'][
                            j, i, t])
                                + hurr['u'][j, i, t] * hurr['u'][j, i, t] * (
                                        hurr['r_list_v'][i] * hurr['w'][j, i, t] + hurr['r_list_v'][i - 1] * hurr['w'][
                                    j, i - 1, t])
                        )
                        ) / (4. * hurr['dz'])

                     ) / (2. * hurr['r_list_v'][i])

    mass_source_v = 0.5 * hurr['v'][j, i, t] * hurr['v'][j, i, t] * (
            hurr['rho_initial_v'][j] * (
            hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] - hurr['r_list_u'][i] * hurr['u'][j, i, t]) / (
                    hurr['r_list_v'][i] * hurr['dr'])
            + (hurr['rho_initial_w'][j + 1] * hurr['w'][j + 1, i, t] - hurr['rho_initial_w'][j] * hurr['w'][j, i, t]) /
            hurr['dz']
    )

    mass_source_w_r = (
                              hurr['w'][j + 1, i, t] * hurr['w'][j + 1, i, t] * (
                              hurr['r_list_u'][i + 1] * (
                              hurr['rho_initial_v'][j + 1] * hurr['u'][j + 1, i + 1, t] + hurr['rho_initial_v'][j] *
                              hurr['u'][j, i + 1, t])
                              - hurr['r_list_u'][i] * (
                                      hurr['rho_initial_v'][j + 1] * hurr['u'][j + 1, i, t] + hurr['rho_initial_v'][j] *
                                      hurr['u'][j, i, t])
                      )
                              + hurr['w'][j, i, t] * hurr['w'][j, i, t] * (
                                      hurr['r_list_u'][i + 1] * (
                                      hurr['rho_initial_v'][j] * hurr['u'][j, i + 1, t] + hurr['rho_initial_v'][j - 1] *
                                      hurr['u'][j - 1, i + 1, t])
                                      - hurr['r_list_u'][i] * (
                                              hurr['rho_initial_v'][j] * hurr['u'][j, i, t] + hurr['rho_initial_v'][
                                          j - 1] * hurr['u'][j - 1, i, t])
                              )
                      ) / (8. * hurr['r_list_v'][i] * hurr['dr'])
    mass_source_w_z = (
                              hurr['w'][j + 1, i, t] * hurr['w'][j + 1, i, t] * (
                              hurr['rho_initial_w'][j + 2] * hurr['w'][j + 2, i, t] - hurr['rho_initial_w'][j] *
                              hurr['w'][j, i, t])
                              + hurr['w'][j, i, t] * hurr['w'][j, i, t] * (
                                      hurr['rho_initial_w'][j + 1] * hurr['w'][j + 1, i, t] - hurr['rho_initial_w'][
                                  j - 1] * hurr['w'][
                                          j - 1, i, t])
                      ) / (8. * hurr['dz'])

    mass_source_w = mass_source_w_r + mass_source_w_z

    subgrid_u = 20. * hurr['rho_initial_v'][j] * (
            hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] * hurr['subgrid_u'][j, i + 1, t] + hurr['r_list_u'][i] *
            hurr['u'][j, i, t] * hurr['subgrid_u'][j, i, t]) / (2. * hurr['r_list_v'][i]) / (2. * hurr['dt'])
    subgrid_v = hurr['rho_initial_v'][j] * hurr['v'][j, i, t] * hurr['subgrid_v'][j, i, t] / (2. * hurr['dt'])
    subgrid_w = 20. * 0.5 * (hurr['rho_initial_w'][j + 1] * hurr['w'][j + 1, i, t] * hurr['subgrid_w'][j + 1, i, t] +
                             hurr['rho_initial_w'][j] * hurr['w'][j, i, t] * hurr['subgrid_w'][j, i, t]) / (
                        2. * hurr['dt'])

    ek_flux_r_u = 0.25 * hurr['rho_initial_v'][j] * (
            hurr['u'][j, i + 2, t] * hurr['u'][j, i + 1, t] * (
            hurr['r_list_u'][i + 2] * hurr['u'][j, i + 2, t]
            + hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t])
            - hurr['u'][j, i, t] * hurr['u'][j, i - 1, t] * (hurr['r_list_u'][i] * hurr['u'][j, i, t]
                                                             + hurr['r_list_u'][i - 1] * hurr['u'][
                                                                 j, i - 1, t])) / (
                          2. * hurr['r_list_v'][i] * hurr['dr'])

    ek_flux_r_v = hurr['rho_initial_v'][j] * hurr['v'][j, i, t] * (
            hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] * hurr['v'][j, i + 1, t]
            - hurr['r_list_u'][i] * hurr['u'][j, i, t] * hurr['v'][j, i - 1, t]) / (
                          2. * hurr['r_list_v'][i] * hurr['dr'])

    ek_flux_r_w = (
                          hurr['r_list_u'][i + 1] * (
                          hurr['rho_initial_v'][j + 1] * hurr['u'][j + 1, i + 1, t] + hurr['rho_initial_v'][
                      j] * hurr['u'][j, i + 1, t]) * (hurr['w'][j + 1, i + 1, t] * hurr['w'][j + 1, i, t])
                          - hurr['r_list_u'][i] * (hurr['rho_initial_v'][j + 1] * hurr['u'][j + 1, i, t] +
                                                   hurr['rho_initial_v'][j] * hurr['u'][j, i, t]) * (
                                  hurr['w'][j + 1, i, t] * hurr['w'][j + 1, i - 1, t])
                          + hurr['r_list_u'][i + 1] * (
                                  hurr['rho_initial_v'][j] * hurr['u'][j, i + 1, t] + hurr['rho_initial_v'][
                              j - 1] * hurr['u'][j - 1, i + 1, t]) * (
                                  hurr['w'][j, i + 1, t] * hurr['w'][j, i, t])
                          - hurr['r_list_u'][i] * (
                                  hurr['rho_initial_v'][j] * hurr['u'][j, i, t] + hurr['rho_initial_v'][
                              j - 1] * hurr['u'][j - 1, i, t]) * (hurr['w'][j, i, t] * hurr['w'][j, i - 1, t])
                  ) / (8. * hurr['r_list_v'][i] * hurr['dr'])

    ek_flux_z_u = (
                          hurr['rho_initial_w'][j + 1] * (
                          hurr['u'][j + 1, i + 1, t] * hurr['u'][j, i + 1, t] * (
                          hurr['r_list_v'][i + 1] * hurr['w'][j + 1, i + 1, t] + hurr['r_list_v'][i] * hurr['w'][
                      j + 1, i, t])
                          + hurr['u'][j + 1, i, t] * hurr['u'][j, i, t] * (
                                  hurr['r_list_v'][i] * hurr['w'][j + 1, i, t] + hurr['r_list_v'][i - 1] *
                                  hurr['w'][j + 1, i - 1, t])
                  ) / (4. * hurr['r_list_v'][i])

                          - hurr['rho_initial_w'][j] * (
                                  hurr['u'][j, i + 1, t] * hurr['u'][j - 1, i + 1, t] * (
                                  hurr['r_list_v'][i + 1] * hurr['w'][j, i + 1, t] + hurr['r_list_v'][i] *
                                  hurr['w'][
                                      j, i, t])
                                  + hurr['u'][j, i, t] * hurr['u'][j - 1, i, t] * (
                                          hurr['r_list_v'][i] * hurr['w'][j, i, t] + hurr['r_list_v'][i - 1] *
                                          hurr['w'][
                                              j, i - 1, t])
                          ) / (4. * hurr['r_list_v'][i])) / (2. * hurr['dz'])

    ek_flux_z_v = hurr['v'][j, i, t] * (
            hurr['rho_initial_w'][j + 1] * hurr['w'][j + 1, i, t] * hurr['v'][j + 1, i, t]
            - hurr['rho_initial_w'][j] * hurr['w'][j, i, t] * hurr['v'][j - 1, i, t]) / (2. * hurr['dz'])

    ek_flux_z_w = 0.25 * (
            hurr['w'][j + 2, i, t] * hurr['w'][j + 1, i, t] * (
            hurr['rho_initial_w'][j + 2] * hurr['w'][j + 2, i, t]
            + hurr['rho_initial_w'][j + 1] * hurr['w'][j + 1, i, t])
            - hurr['w'][j, i, t] * hurr['w'][j - 1, i, t] * (
                    hurr['rho_initial_w'][j] * hurr['w'][j, i, t]
                    + hurr['rho_initial_w'][j - 1] * hurr['w'][j - 1, i, t])
    ) / (2. * hurr['dz'])

    ek_flux_u = ek_flux_r_u + ek_flux_z_u
    ek_flux_v = ek_flux_r_v + ek_flux_z_v
    ek_flux_w = ek_flux_r_w + ek_flux_z_w

    coriolis_u = 0.25 * hurr['rho_initial_v'][j] * (
            hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] * (
            hurr['v'][j, i + 1, t] ** 2 / hurr['r_list_v'][i + 1] + hurr['f'] * hurr['v'][j, i + 1, t] + hurr['v'][
        j, i, t] ** 2 / hurr['r_list_v'][i] + hurr['f'] * hurr['v'][j, i, t])
            + hurr['r_list_u'][i] * hurr['u'][j, i, t] * (
                    hurr['v'][j, i, t] ** 2 / hurr['r_list_v'][i] + hurr['f'] * hurr['v'][j, i, t] + hurr['v'][
                j, i - 1, t] ** 2 / hurr['r_list_v'][i - 1] + hurr['f'] * hurr['v'][j, i - 1, t])
    ) / hurr['r_list_v'][i]

    coriolis_v = 0.5 * hurr['rho_initial_v'][j] * (
            hurr['v'][j, i, t] ** 2 / hurr['r_list_v'][i] + hurr['f'] * hurr['v'][j, i, t]) * (
                         hurr['r_list_u'][i + 1] * hurr['u'][j, i + 1, t] + hurr['r_list_u'][i] * hurr['u'][
                     j, i, t]) / hurr['r_list_v'][i]

    return [time_change_ek_u, time_change_ek_v, time_change_ek_w, time_change_ee,
            pressure_div, pressure_grad_u, pressure_grad_w, pressure_dthetav,
            ek_flux_u, ek_flux_v, ek_flux_w,
            mass_source_u, mass_source_v, mass_source_w,
            subgrid_u, subgrid_v, subgrid_w,
            buoyancy, coriolis_u, coriolis_v]


def all_parcels_all_time(data_directory, hurr, run_id):
    time_change_ek_u = np.zeros_like(hurr['v'])
    time_change_ek_v = np.zeros_like(hurr['v'])
    time_change_ek_w = np.zeros_like(hurr['v'])
    time_change_ee = np.zeros_like(hurr['v'])
    pressure_div = np.zeros_like(hurr['v'])
    pressure_grad_u = np.zeros_like(hurr['v'])
    pressure_grad_w = np.zeros_like(hurr['v'])
    pressure_dthetav = np.zeros_like(hurr['v'])
    ek_flux_u = np.zeros_like(hurr['v'])
    ek_flux_v = np.zeros_like(hurr['v'])
    ek_flux_w = np.zeros_like(hurr['v'])
    mass_source_u = np.zeros_like(hurr['v'])
    mass_source_v = np.zeros_like(hurr['v'])
    mass_source_w = np.zeros_like(hurr['v'])
    subgrid_u = np.zeros_like(hurr['v'])
    subgrid_v = np.zeros_like(hurr['v'])
    subgrid_w = np.zeros_like(hurr['v'])
    buoyancy = np.zeros_like(hurr['v'])
    coriolis_u = np.zeros_like(hurr['v'])
    coriolis_v = np.zeros_like(hurr['v'])

    for t in tqdm(range(hurr['timesteps'])):
        for j in range(hurr['n'] - 1):
            for i in range(hurr['m'] - 1):
                [time_change_ek_u[j, i, t], time_change_ek_v[j, i, t], time_change_ek_w[j, i, t],
                 time_change_ee[j, i, t],
                 pressure_div[j, i, t], pressure_grad_u[j, i, t], pressure_grad_w[j, i, t], pressure_dthetav[j, i, t],
                 ek_flux_u[j, i, t], ek_flux_v[j, i, t], ek_flux_w[j, i, t],
                 mass_source_u[j, i, t], mass_source_v[j, i, t], mass_source_w[j, i, t],
                 subgrid_u[j, i, t], subgrid_v[j, i, t], subgrid_w[j, i, t],
                 buoyancy[j, i, t], coriolis_u[j, i, t], coriolis_v[j, i, t]] = budget_parcel_at_time(hurr, i, j, t)

    energy_budget = {}
    energy_budget['time_change_ek_u'] = time_change_ek_u
    energy_budget['time_change_ek_v'] = time_change_ek_v
    energy_budget['time_change_ek_w'] = time_change_ek_w
    energy_budget['time_change_ee'] = time_change_ee
    energy_budget['pressure_div'] = pressure_div
    energy_budget['pressure_grad_u'] = pressure_grad_u
    energy_budget['pressure_grad_w'] = pressure_grad_w
    energy_budget['pressure_dthetav'] = pressure_dthetav
    energy_budget['ek_flux_u'] = ek_flux_u
    energy_budget['ek_flux_v'] = ek_flux_v
    energy_budget['ek_flux_w'] = ek_flux_w
    energy_budget['mass_source_u'] = mass_source_u
    energy_budget['mass_source_v'] = mass_source_v
    energy_budget['mass_source_w'] = mass_source_w
    energy_budget['subgrid_u'] = subgrid_u
    energy_budget['subgrid_v'] = subgrid_v
    energy_budget['subgrid_w'] = subgrid_w
    energy_budget['buoyancy'] = buoyancy
    energy_budget['coriolis_u'] = coriolis_u
    energy_budget['coriolis_v'] = coriolis_v

    np.savez(f'{data_directory}/kinetic_elastic_energy_budgets_{run_id}.npz', **energy_budget)


if __name__ == '__main__':
    data_directory = '../data/J30pt3'
    run_id = 'J30pt3'
    hurr = read_fortran_output(data_directory)
    all_parcels_all_time(data_directory, hurr, run_id)
