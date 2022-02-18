import numpy as np
from model_reader import read_fortran_output
import matplotlib.pyplot as plt
from volume_integral import vol_int
from utils import clean_filename, multiply_by_rho


def precip_variables():
    # QRFALL,QIFALL,THNUCL,QNUCL,QLNUCL,QINUCL,THRIM,QLRIM,
    # QIRIM,THDEPSUB,QDEPSUB,QLDEPSUB,QIDEPSUB,THCAPT,QRCAPT,QICAPT,
    # THSNEVAP,QSNEVAP,QISNEVAP,THMELT,QRMELT,QIMELT,THEVAP,QEVAP,
    # QREVAP,QLACCR,QRACCR,QLAUTO,QRAUTO
    output_variables = ['r_fallout', 'i_fallout', 'theta_nucleation', 'v_nucleation', 'l_nucleation', 'i_nucleation',
                        'theta_riming', 'l_riming', 'i_riming', 'theta_depsub', 'v_depsub', 'l_depsub', 'i_depsub',
                        'theta_capture', 'r_capture', 'i_capture', 'theta_snow_evap', 'v_snow_evap', 'i_snow_evap',
                        'theta_melt', 'r_melt', 'i_melt', 'theta_evap', 'v_evap', 'r_evap', 'l_accretion',
                        'r_accretion', 'l_autoconversion', 'r_autoconversion']
    return output_variables


def read_budget(directory, hurr):
    m = hurr['m']
    n = hurr['n']
    timesteps = hurr['timesteps']
    precip_file = directory + '/fort.19'
    precip_data = {key: np.zeros((n, m, timesteps)) for key in precip_variables()}
    with open(precip_file, 'rb') as datafile:
        _ = np.fromfile(datafile, dtype=np.float32, count=1)
        for t in range(timesteps):
            for variable in precip_variables():
                data = np.fromfile(datafile, dtype=np.float32, count=m * n)
                precip_data[variable][:, :, t] = np.reshape(data, (n, m))
            _ = np.fromfile(datafile, dtype=np.float32, count=2)
    return precip_data


def precip_budgets(directory, hurr, region_key='non-sponge', end_time=None, save=False):
    precip_data = read_budget(directory, hurr)
    theta_keys = [key for key in precip_variables() if key.startswith('theta_')]
    rv_keys = [key for key in precip_variables() if key.startswith('v_')]
    rl_keys = [key for key in precip_variables() if key.startswith('l_')]
    rr_keys = [key for key in precip_variables() if key.startswith('r_')]
    ri_keys = [key for key in precip_variables() if key.startswith('i_')]

    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.
    if end_time and end_time > time_list.size:
        end_time = None
    time_list = time_list[0:end_time]

    if isinstance(region_key, str):
        region_key = [region_key]

    for region in region_key:
        theta_sum = np.zeros_like(hurr['v'])
        rv_sum = np.zeros_like(hurr['v'])
        rl_sum = np.zeros_like(hurr['v'])
        rr_sum = np.zeros_like(hurr['v'])
        ri_sum = np.zeros_like(hurr['v'])

        fig = plt.figure(figsize=(10, 7.5))
        plt.title(f'Microphysical contributions to $\\theta$, {region} region', fontsize=14)
        for key in theta_keys:
            plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, precip_data[key]) / (2. * hurr['dt']),
                                        region_key=region)[0:end_time], linewidth=2, label=key)
            theta_sum += precip_data[key]
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, theta_sum) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], 'k', linewidth=2, label='sum')
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, hurr['precip_theta']) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], '--', color='gray', linewidth=2, label='net precip')

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('time (h)', fontsize=20)
        ax.set_ylabel('(kgK/s)', fontsize=24)
        if save:
            fig.savefig(f'../results/precip_diagnostics/theta_{clean_filename(region)}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        fig = plt.figure(figsize=(10, 7.5))
        plt.title(f'Microphysical contributions to $r_v$, {region} region', fontsize=14)
        for key in rv_keys:
            plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, precip_data[key]) / (2. * hurr['dt']),
                                        region_key=region)[0:end_time], linewidth=2, label=key)
            rv_sum += precip_data[key]
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, rv_sum) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], 'k', linewidth=2, label='sum')
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, hurr['precip_rv']) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], '--', color='gray', linewidth=2, label='net precip')

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('time (h)', fontsize=20)
        ax.set_ylabel('(kg/s)', fontsize=24)
        if save:
            fig.savefig(f'../results/precip_diagnostics/rv_{clean_filename(region)}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        fig = plt.figure(figsize=(10, 7.5))
        plt.title(f'Microphysical contributions to $r_l$, {region} region', fontsize=14)
        for key in rl_keys:
            plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, precip_data[key]) / (2. * hurr['dt']),
                                        region_key=region)[0:end_time], linewidth=2, label=key)
            rl_sum += precip_data[key]
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, rl_sum) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], 'k', linewidth=2, label='sum')
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, hurr['precip_rl']) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], '--', color='gray', linewidth=2, label='net precip')

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('time (h)', fontsize=20)
        ax.set_ylabel('(kg/s)', fontsize=24)
        if save:
            fig.savefig(f'../results/precip_diagnostics/rl_{clean_filename(region)}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        fig = plt.figure(figsize=(10, 7.5))
        plt.title(f'Microphysical contributions to $r_r$, {region} region', fontsize=14)
        for key in rr_keys:
            plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, precip_data[key]) / (2. * hurr['dt']),
                                        region_key=region)[0:end_time], linewidth=2, label=key)
            rr_sum += precip_data[key]
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, rr_sum) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], 'k', linewidth=2, label='sum')
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, hurr['precip_rr']) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], '--', color='gray', linewidth=2, label='net precip')

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('time (h)', fontsize=20)
        ax.set_ylabel('(kg/s)', fontsize=24)
        if save:
            fig.savefig(f'../results/precip_diagnostics/rr_{clean_filename(region)}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        fig = plt.figure(figsize=(10, 7.5))
        plt.title(f'Microphysical contributions to $r_i$, {region} region', fontsize=14)
        for key in ri_keys:
            plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, precip_data[key]) / (2. * hurr['dt']),
                                        region_key=region)[0:end_time], linewidth=2, label=key)
            ri_sum += precip_data[key]
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, ri_sum) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], 'k', linewidth=2, label='sum')
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, hurr['precip_ri']) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], '--', color='gray', linewidth=2, label='net precip')

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('time (h)', fontsize=20)
        ax.set_ylabel('(kg/s)', fontsize=24)
        if save:
            fig.savefig(f'../results/precip_diagnostics/ri_{clean_filename(region)}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        fig = plt.figure(figsize=(10, 7.5))
        plt.title(f'Microphysical contributions to $r_v + r_l$, {region} region', fontsize=14)
        for key in (rv_keys + rl_keys):
            plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, precip_data[key]) / (2. * hurr['dt']),
                                        region_key=region)[0:end_time], linewidth=2, label=key)
        rv_rl_sum = rv_sum + rl_sum
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, rv_rl_sum) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], 'k', linewidth=2, label='sum')
        plt.plot(time_list,
                 vol_int(hurr, multiply_by_rho(hurr, hurr['precip_rv'] + hurr['precip_rl']) / (2. * hurr['dt']),
                         region_key=region)[0:end_time], '--', color='gray', linewidth=2, label='net precip')

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('time (h)', fontsize=20)
        ax.set_ylabel('(kg/s)', fontsize=24)
        if save:
            fig.savefig(f'../results/precip_diagnostics/rv_plus_rl_{clean_filename(region)}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        fig = plt.figure(figsize=(10, 7.5))
        plt.title(f'Microphysical contributions to $r_t$, {region} region', fontsize=14)
        for key in (rv_keys + rl_keys + rr_keys + ri_keys):
            plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, precip_data[key]) / (2. * hurr['dt']),
                                        region_key=region)[0:end_time], linewidth=2, label=key)
        rt_sum = rv_sum + rl_sum + rr_sum + ri_sum
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, rt_sum) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], 'k', linewidth=2, label='sum')
        all_precip_terms = hurr['precip_rv'] + hurr['precip_rl'] + hurr['precip_rr'] + hurr['precip_ri']
        plt.plot(time_list, vol_int(hurr, multiply_by_rho(hurr, all_precip_terms) / (2. * hurr['dt']),
                                    region_key=region)[0:end_time], '--', color='gray', linewidth=2, label='net precip')

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('time (h)', fontsize=20)
        ax.set_ylabel('(kg/s)', fontsize=24)
        if save:
            fig.savefig(f'../results/precip_diagnostics/rt_{clean_filename(region)}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def theta_e_conservation(directory, hurr, region_key='non-sponge'):
    precip_data = read_budget(directory, hurr)

    hurr['exner'] = np.dstack([np.tile(hurr['pi_initial'], (hurr['m'], 1)).transpose()] * hurr['timesteps']) + hurr[
        'pi']
    Lf = 3.34e5
    rv_coeff = hurr['Lv'] / (hurr['cp'] * hurr['exner'])
    ri_coeff = Lf / (hurr['cp'] * hurr['exner'])

    theta_keys = [key for key in precip_variables() if key.startswith('theta_')]
    rv_keys = [key for key in precip_variables() if key.startswith('v_')]
    ri_keys_no_fallout = [key for key in precip_variables() if key.startswith('i_') and not key.endswith('fallout')]

    theta_sum = np.zeros_like(hurr['v'])
    rv_sum = np.zeros_like(hurr['v'])
    ri_no_fallout_sum = np.zeros_like(hurr['v'])

    for key in theta_keys:
        theta_sum += precip_data[key]
    for key in rv_keys:
        rv_sum += precip_data[key]
    for key in ri_keys_no_fallout:
        ri_no_fallout_sum += precip_data[key]

    theta_e_sum = theta_sum + rv_coeff * rv_sum
    theta_ei_sum = theta_sum + rv_coeff * rv_sum - ri_coeff * ri_no_fallout_sum

    plt.figure()
    plt.plot(vol_int(hurr, theta_e_sum, region_key=region_key), label='theta_e')
    plt.plot(vol_int(hurr, theta_ei_sum, region_key=region_key), label='theta_ei')
    plt.plot(vol_int(hurr, theta_ei_sum - ri_coeff * precip_data['i_fallout'], region_key=region_key),
             label='theta_ei with ice fallout')
    plt.legend(loc='best')
    plt.show()


def theta_ei_withrain(directory, hurr, region_key='non-sponge'):
    precip_data = read_budget(directory, hurr)

    hurr['exner'] = np.dstack([np.tile(hurr['pi_initial'], (hurr['m'], 1)).transpose()] * hurr['timesteps']) + hurr[
        'pi']
    Lf = 3.34e5
    Ls = hurr['Lv'] + Lf
    old_rv_coeff = hurr['Lv'] / (hurr['cp'] * hurr['exner'])
    rv_coeff = Ls / (hurr['cp'] * hurr['exner'])
    rl_coeff = Lf / (hurr['cp'] * hurr['exner'])

    theta_keys = [key for key in precip_variables() if key.startswith('theta_')]
    rv_keys = [key for key in precip_variables() if key.startswith('v_')]
    rl_keys = [key for key in precip_variables() if key.startswith(('l_', 'r_'))]

    theta_sum = np.zeros_like(hurr['v'])
    rv_sum = np.zeros_like(hurr['v'])
    rl_sum = np.zeros_like(hurr['v'])

    for key in theta_keys:
        theta_sum += precip_data[key]
    for key in rv_keys:
        rv_sum += precip_data[key]
    for key in rl_keys:
        rl_sum += precip_data[key]

    theta_e_sum = theta_sum + old_rv_coeff * rv_sum
    theta_ei_sum = theta_sum + rv_coeff * rv_sum + rl_coeff * rl_sum

    plt.figure()
    plt.plot(vol_int(hurr, theta_e_sum, region_key=region_key), label='theta_e')
    plt.plot(vol_int(hurr, theta_ei_sum, region_key=region_key), label='theta_ei')
    plt.plot(vol_int(hurr, rl_coeff * precip_data['r_fallout'], region_key=region_key), '--', label='r fallout')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    data_dir = '../data'
    hurr = read_fortran_output(data_dir)
    precip_budgets(data_dir, hurr, region_key='r<500', save=True)
