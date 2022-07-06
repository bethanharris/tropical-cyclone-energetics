import numpy as np
import matplotlib as mpl
from model_reader import read_fortran_output
import matplotlib.pyplot as plt
from volume_integral import vol_int, integrate_area_surface, mask_array
from utils import clean_filename, multiply_by_rho, safe_div, running_mean, column_areas
from energy_budget_timeseries import time_jump_masker, plot_time_series
from precip_budget import read_budget as read_precip_budget
from diabatic_budgets import *


def ape_production_budget_linestyles():
    colours = {
        'mixing': '#54FD70',
        'ice': '#A6CEE3',
        'precip': '#2F8FD0',
        'surface_flux': '#40770E',
        'radcool': '#5105A9'
    }
    markers = {
        'mixing': 'None',
        'ice': 'None',
        'precip': 'None',
        'surface_flux': 'None',
        'radcool': 'None'
    }
    return colours, markers


def plot_theta_ei_budget(directory, hurr, run_id, region_key='non-sponge', fixed_exner=None, end_time=None, save=False,
                         normalise=False, title=False, legend_on_error_plot=False):
    (lagr_theta_ei, _, precip_theta_ei, subgrid_theta_ei,
     vertical_flux_theta_e, radcool_theta) = theta_ei_budget(hurr, fixed_exner=fixed_exner)

    colours, shapes = ape_production_budget_linestyles()
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.

    if end_time and end_time > time_list.size:
        end_time = None
    time_list = time_list[0:end_time]

    if isinstance(region_key, str):
        region_key = [region_key]

    for region in region_key:
        lagr_integral = vol_int(hurr, multiply_by_rho(hurr, lagr_theta_ei), region_key=region)
        subgrid_integral = vol_int(hurr, multiply_by_rho(hurr, subgrid_theta_ei), region_key=region)
        radcool_integral = vol_int(hurr, multiply_by_rho(hurr, radcool_theta), region_key=region)
        precip_data = read_precip_budget(directory, hurr)
        exner = np.dstack([np.tile(hurr['pi_initial'], (hurr['m'], 1)).transpose()] * hurr['timesteps']) + hurr['pi']
        rl_coeff = hurr['Lf'] / (hurr['cp'] * exner)  # should this be fixed_exner?
        r_fallout_theta_ei = rl_coeff * precip_data['r_fallout'] / (2. * hurr['dt'])
        r_fallout_integral = vol_int(hurr, multiply_by_rho(hurr, r_fallout_theta_ei), region_key=region)
        surface_flux = hurr['rho_initial_v'][0] * integrate_area_surface(hurr, -vertical_flux_theta_e,
                                                                         region_key=region)
        if normalise:
            rho_grid = np.dstack([np.tile(hurr['rho_initial_v'], (hurr['m'], 1)).transpose()] * hurr['timesteps'])
            region_mass = vol_int(hurr, rho_grid, region_key=region)
            lagr_integral = safe_div(lagr_integral, region_mass)
            subgrid_integral = safe_div(subgrid_integral, region_mass)
            radcool_integral = safe_div(radcool_integral, region_mass)
            r_fallout_integral = safe_div(r_fallout_integral, region_mass)
            surface_flux = safe_div(surface_flux, region_mass)

        mixing_integral = subgrid_integral - surface_flux

        fig = plt.figure(figsize=(8, 5))
        plt.plot(time_list, lagr_integral[0:end_time], 'k--', linewidth=2,
                 label=r'$\overline{\rho}\frac{D\theta_{ei}}{Dt}$')
        plt.plot(time_list, mixing_integral[0:end_time], color=colours['mixing'], marker=shapes['mixing'],
                 linewidth=2, label='mixing')
        plt.plot(time_list, radcool_integral[0:end_time], color=colours['radcool'], marker=shapes['radcool'],
                 linewidth=2, label='radiative cooling')
        plt.plot(time_list, r_fallout_integral[0:end_time], color=colours['precip'], marker=shapes['precip'],
                 linewidth=2, label='rain fallout')
        plt.plot(time_list, surface_flux[0:end_time], color=colours['surface_flux'], marker=shapes['surface_flux'],
                 linewidth=2, label='surface flux')

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlim([0, 250])
        ax.set_xlabel('time (h)', fontsize=20)
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_fontsize(16)
        if normalise:
            ax.set_ylabel(r'$\mathregular{\left(Ks^{-1}\right)}$', fontsize=20)
        else:
            ax.set_ylabel(r'$\mathregular{\left(kgKs^{-1}\right)}$', fontsize=20)
        if title:
            theta_ei_label = r'$\theta_{ei}$'
            plot_title = f'{theta_ei_label} budget - {run_id} {region}'
            ax.set_title(plot_title, fontsize=24)
        if save:
            save_name = f'{clean_filename(region)}_theta_ei_budget_{run_id}'
            if normalise:
                save_name += '_normalised'
            fig.savefig(f'../results/ape_production_timeseries/{save_name}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        estimated_derivative = r_fallout_theta_ei + subgrid_theta_ei + radcool_theta

        fig = plt.figure(figsize=(6, 4.5))
        if title:
            plt.title(plot_title, fontsize=16)
        plt.plot(time_list, vol_int(hurr, lagr_theta_ei, region_key=region)[0:end_time], 'k', linewidth=2,
                 label=r'$\overline{\rho}\frac{D\theta_{ei}}{Dt}$')
        plt.plot(time_list, vol_int(hurr, estimated_derivative, region_key=region)[0:end_time], '--',
                 color='orange', dashes=(5, 5), linewidth=2, label=r'$\mathrm{sum}$')

        ax = plt.gca()
        if legend_on_error_plot:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlim([0, 250])
        ax.set_xlabel('time (h)', fontsize=20)
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_fontsize(16)
        if normalise:
            ax.set_ylabel(r'$\mathregular{\left(Ks^{-1}\right)}$', fontsize=20)
        else:
            ax.set_ylabel(r'$\mathregular{\left(kgKs^{-1}\right)}$', fontsize=20)
        if save:
            extra = (lgd,) if legend_on_error_plot else None
            fig.savefig(f'../results/ape_production_timeseries/error/{save_name}.png',
                        bbox_inches='tight', bbox_extra_artists=extra)
            plt.close()
        else:
            plt.show()


def plot_rt_budget(directory, hurr, run_id, region_key='non-sponge', end_time=None, save=False,
                   normalise=False, title=False, legend_on_error_plot=False):
    lagr, precip, subgrid = rt_budget(hurr)

    colours, shapes = ape_production_budget_linestyles()
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.

    if end_time and end_time > time_list.size:
        end_time = None
    time_list = time_list[0:end_time]

    if isinstance(region_key, str):
        region_key = [region_key]

    for region in region_key:
        lagr_integral = vol_int(hurr, multiply_by_rho(hurr, lagr), region_key=region)
        subgrid_integral = vol_int(hurr, multiply_by_rho(hurr, subgrid), region_key=region)
        precip_data = read_precip_budget(directory, hurr)
        r_fallout = precip_data['r_fallout'] / (2. * hurr['dt'])
        i_fallout = precip_data['i_fallout'] / (2. * hurr['dt'])
        r_fallout_integral = vol_int(hurr, multiply_by_rho(hurr, r_fallout), region_key=region)
        i_fallout_integral = vol_int(hurr, multiply_by_rho(hurr, i_fallout), region_key=region)
        surface_flux = hurr['rho_initial_v'][0] * integrate_area_surface(hurr, -hurr['vertical_flux_rv'],
                                                                         region_key=region)
        if normalise:
            rho_grid = np.dstack([np.tile(hurr['rho_initial_v'], (hurr['m'], 1)).transpose()] * hurr['timesteps'])
            region_mass = vol_int(hurr, rho_grid, region_key=region)
            lagr_integral = safe_div(lagr_integral, region_mass)
            subgrid_integral = safe_div(subgrid_integral, region_mass)
            r_fallout_integral = safe_div(r_fallout_integral, region_mass)
            i_fallout_integral = safe_div(i_fallout_integral, region_mass)
            surface_flux = safe_div(surface_flux, region_mass)

        mixing_integral = subgrid_integral - surface_flux

        fig = plt.figure(figsize=(8, 5))
        plt.plot(time_list, lagr_integral[0:end_time], 'k--', linewidth=2, label=r'$\overline{\rho}\frac{Dr_t}{Dt}$')
        plt.plot(time_list, mixing_integral[0:end_time], color=colours['mixing'], marker=shapes['mixing'],
                 linewidth=2, label='mixing')
        plt.plot(time_list, r_fallout_integral[0:end_time], color=colours['precip'], marker=shapes['precip'],
                 linewidth=2, label='r fallout')
        plt.plot(time_list, i_fallout_integral[0:end_time],
                 color=colours['ice'], marker=shapes['ice'], linewidth=2, label='i fallout')
        plt.plot(time_list, surface_flux[0:end_time], color=colours['surface_flux'], marker=shapes['surface_flux'],
                 linewidth=2, label='surface flux')

        ax = plt.gca()
        ax.set_xlim([0, 250])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('time (h)', fontsize=20)
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_fontsize(16)
        if normalise:
            ax.set_ylabel(r'$\mathregular{\left(s^{-1}\right)}$', fontsize=20)
        else:
            ax.set_ylabel(r'$\mathregular{\left(kgs^{-1}\right)}$', fontsize=20)
        if title:
            plot_title = f'$r_t$ budget - {run_id} {region}'
            ax.set_title(plot_title, fontsize=24)
        if save:
            save_name = f'{clean_filename(region)}_rt_budget_{run_id}'
            if normalise:
                save_name += '_normalised'
            fig.savefig(f'../results/ape_production_timeseries/{save_name}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        estimated_derivative = r_fallout_integral + i_fallout_integral + subgrid_integral

        fig = plt.figure(figsize=(6, 4.5))
        if title:
            plt.title(plot_title, fontsize=16)
        plt.plot(time_list, lagr_integral[0:end_time], 'k', linewidth=2, label=r'$\overline{\rho}\frac{Dr_t}{Dt}$')
        plt.plot(time_list, estimated_derivative[0:end_time], '--', color='orange',
                 dashes=(5, 5), linewidth=2, label=r'$\mathrm{sum}$')

        ax = plt.gca()
        if legend_on_error_plot:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlim([0, 250])
        ax.set_xlabel('time (h)', fontsize=20)
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_fontsize(16)
        if normalise:
            ax.set_ylabel(r'$\mathregular{\left(s^{-1}\right)}$', fontsize=20)
        else:
            ax.set_ylabel(r'$\mathregular{\left(kgs^{-1}\right)}$', fontsize=20)
        if save:
            extra = (lgd,) if legend_on_error_plot else None
            fig.savefig(f'../results/ape_production_timeseries/error/{save_name}.png',
                        bbox_inches='tight', bbox_extra_artists=extra)
            plt.close()
        else:
            plt.show()


def plot_ape_production_theta_ei_budget(directory, hurr, run_id, ref_state, region_key='non-sponge', save=False,
                                        end_time=None, normalise=False, title=False, fixed_exner=None,
                                        legend_on_error_plot=False):
    ape = np.load(f'{directory}/ape_{ref_state}_ref_at_timesteps_{run_id}.npz')
    (lagr_theta_ei, _, _, subgrid_theta_ei,
     vertical_flux_theta_e, radcool_theta) = theta_ei_budget(hurr, fixed_exner=fixed_exner)

    lagr_prod = multiply_by_rho(hurr, ape['G_theta_e'] * lagr_theta_ei)
    subgrid_prod = multiply_by_rho(hurr, ape['G_theta_e'] * subgrid_theta_ei)
    radcool_prod = multiply_by_rho(hurr, ape['G_theta_e'] * radcool_theta)

    colours, shapes = ape_production_budget_linestyles()
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.

    if end_time and end_time > time_list.size:
        end_time = None
    time_list = time_list[0:end_time]

    theta_e_prod_label = r'$\overline{\rho}\varepsilon_{\theta_{ei}}c_p\Pi\frac{D\theta_{ei}}{Dt}$'

    if isinstance(region_key, str):
        region_key = [region_key]

    for region in region_key:

        lagr_integral = vol_int(hurr, lagr_prod, region_key=region)
        subgrid_integral = vol_int(hurr, subgrid_prod, region_key=region)
        radcool_integral = vol_int(hurr, radcool_prod, region_key=region)
        exner = np.dstack([np.tile(hurr['pi_initial'], (hurr['m'], 1)).transpose()] * hurr['timesteps']) + hurr[
            'pi']
        rl_coeff = hurr['Lf'] / (hurr['cp'] * exner)
        precip_data = read_precip_budget(directory, hurr)
        r_fallout_prod = multiply_by_rho(hurr, ape['G_theta_e'] * rl_coeff * precip_data['r_fallout']) / (
                    2. * hurr['dt'])
        r_fallout_integral = vol_int(hurr, r_fallout_prod, region_key=region)
        surface_flux = hurr['rho_initial_v'][0] * integrate_area_surface(hurr,
                                                                         ape['G_theta_e'] * -vertical_flux_theta_e,
                                                                         region_key=region)
        if normalise:
            rho_grid = np.dstack([np.tile(hurr['rho_initial_v'], (hurr['m'], 1)).transpose()] * hurr['timesteps'])
            region_mass = vol_int(hurr, rho_grid, region_key=region)
            lagr_integral = safe_div(lagr_integral, region_mass)
            subgrid_integral = safe_div(subgrid_integral, region_mass)
            radcool_integral = safe_div(radcool_integral, region_mass)
            r_fallout_integral = safe_div(r_fallout_integral, region_mass)
            surface_flux = safe_div(surface_flux, region_mass)

        mixing_integral = subgrid_integral - surface_flux

        fig = plt.figure(figsize=(8, 5))
        plt.plot(time_list, lagr_integral[0:end_time], 'k--', linewidth=2, label=theta_e_prod_label)
        plt.plot(time_list, mixing_integral[0:end_time], color=colours['mixing'], marker=shapes['mixing'],
                 linewidth=2, label='mixing')
        plt.plot(time_list, radcool_integral[0:end_time], color=colours['radcool'], marker=shapes['radcool'],
                  linewidth=2, label='radiative cooling')
        plt.plot(time_list, r_fallout_integral[0:end_time], color=colours['precip'], marker=shapes['precip'],
                     linewidth=2, label='rain fallout')
        plt.plot(time_list, surface_flux[0:end_time], color=colours['surface_flux'], marker=shapes['surface_flux'],
                 linewidth=2, label='surface flux')

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_fontsize(16)
        ax.set_xlabel('time (h)', fontsize=20)
        if normalise:
            ax.set_ylabel('(W/kg)', fontsize=20)
        else:
            ax.set_ylabel('(W)', fontsize=20)
        theta_str = r'$\theta_{ei}$'
        if title:
            plot_title = f'{theta_str} APE production budget - {run_id} {region}'
            ax.set_title(plot_title, fontsize=18)
        if save:
            save_name = f'{clean_filename(region)}_theta_ei_production_budget_{ref_state}_ref_{run_id}'
            if normalise:
                save_name += '_normalised'
            fig.savefig(f'../results/ape_production_timeseries/{save_name}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        ape_budget = np.load(f'{directory}/ape_{ref_state}_ref_budget_{run_id}.npz')
        estimated_derivative = r_fallout_integral + subgrid_integral + radcool_integral
        ape_budget = time_jump_masker(ape_budget)

        theta_e_production = vol_int(hurr, ape_budget['theta_e_prod'], region_key=region)
        if normalise:
            theta_e_production = safe_div(theta_e_production, region_mass)

        fig = plt.figure(figsize=(6, 4.5))
        if title:
            plt.title(plot_title, fontsize=16)
        plt.plot(time_list, theta_e_production[0:end_time], 'k', linewidth=2, label=theta_e_prod_label)
        plt.plot(time_list, estimated_derivative[0:end_time], '--', color='orange',
                 linewidth=2, label=r'$\mathrm{sum}$')

        ax = plt.gca()
        if legend_on_error_plot:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_fontsize(16)
        ax.set_xlabel('time (h)', fontsize=20)
        if normalise:
            ax.set_ylabel('(W/kg)', fontsize=20)
        else:
            ax.set_ylabel('(W)', fontsize=20)
        if save:
            extra = (lgd,) if legend_on_error_plot else None
            fig.savefig(f'../results/ape_production_timeseries/error/{save_name}.png',
                        bbox_extra_artists=extra, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_ape_production_rt_budget(directory, hurr, run_id, ref_state, region_key='non-sponge', save=False,
                                  end_time=None, normalise=False, title=False, legend_on_error_plot=False):
    ape = np.load(f'{directory}/ape_{ref_state}_ref_at_timesteps_{run_id}.npz')
    lagr, precip, subgrid = rt_budget(hurr)

    lagr_prod = multiply_by_rho(hurr, ape['G_rt'] * lagr)
    subgrid_prod = multiply_by_rho(hurr, ape['G_rt'] * subgrid)

    colours, shapes = ape_production_budget_linestyles()
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.

    if end_time and end_time > time_list.size:
        end_time = None
    time_list = time_list[0:end_time]

    rt_prod_label = r'$\overline{\rho}\varepsilon_{r_t}L_s\frac{Dr_t}{Dt}$'

    if isinstance(region_key, str):
        region_key = [region_key]

    for region in region_key:

        lagr_integral = vol_int(hurr, lagr_prod, region_key=region)
        subgrid_integral = vol_int(hurr, subgrid_prod, region_key=region)
        precip_data = read_precip_budget(directory, hurr)
        r_fallout_prod = multiply_by_rho(hurr, ape['G_rt'] * precip_data['r_fallout']) / (2. * hurr['dt'])
        i_fallout_prod = multiply_by_rho(hurr, ape['G_rt'] * precip_data['i_fallout']) / (2. * hurr['dt'])
        r_fallout_integral = vol_int(hurr, r_fallout_prod, region_key=region)
        i_fallout_integral = vol_int(hurr, i_fallout_prod, region_key=region)
        surface_flux = (hurr['rho_initial_v'][0] * integrate_area_surface(hurr, (
                        ape['G_rt'] * -hurr['vertical_flux_rv'][:-1, :, :]), region_key=region))

        if normalise:
            rho_grid = np.dstack([np.tile(hurr['rho_initial_v'], (hurr['m'], 1)).transpose()] * hurr['timesteps'])
            region_mass = vol_int(hurr, rho_grid, region_key=region)
            lagr_integral = safe_div(lagr_integral, region_mass)
            subgrid_integral = safe_div(subgrid_integral, region_mass)
            r_fallout_integral = safe_div(r_fallout_integral, region_mass)
            i_fallout_integral = safe_div(i_fallout_integral, region_mass)
            surface_flux = safe_div(surface_flux, region_mass)

        mixing_integral = subgrid_integral - surface_flux

        fig = plt.figure(figsize=(8, 5))
        plt.plot(time_list, lagr_integral[0:end_time], 'k--', linewidth=2, label=rt_prod_label)
        plt.plot(time_list, mixing_integral[0:end_time], color=colours['mixing'], marker=shapes['mixing'],
                 linewidth=2, label='mixing')
        plt.plot(time_list, r_fallout_integral[0:end_time], color=colours['precip'], marker=shapes['precip'],
                 linewidth=2, label='r fallout')
        plt.plot(time_list, i_fallout_integral[0:end_time], color=colours['ice'], marker=shapes['ice'],
                 linewidth=2, label='i fallout')
        plt.plot(time_list, surface_flux[0:end_time], color=colours['surface_flux'], marker=shapes['surface_flux'],
                 linewidth=2, label='surface flux')

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_fontsize(16)
        ax.set_xlabel('time (h)', fontsize=20)
        if normalise:
            ax.set_ylabel('(W/kg)', fontsize=20)
        else:
            ax.set_ylabel('(W)', fontsize=20)
        if title:
            plot_title = f'$r_t$ APE production budget - {run_id} {region}'
            ax.set_title(plot_title, fontsize=18)
        if save:
            save_name = f'{clean_filename(region)}_rt_production_budget_{ref_state}_ref_{run_id}'
            if normalise:
                save_name += '_normalised'
            fig.savefig(f'../results/ape_production_timeseries/{save_name}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        ape_budget = np.load(f'{directory}/ape_{ref_state}_ref_budget_{run_id}.npz')
        estimated_derivative = r_fallout_integral + i_fallout_integral + subgrid_integral
        ape_budget = time_jump_masker(ape_budget)

        rt_production = vol_int(hurr, ape_budget['rt_prod'], region_key=region)
        if normalise:
            rt_production = safe_div(rt_production, region_mass)

        fig = plt.figure(figsize=(6, 4.5))
        if title:
            plt.title(plot_title, fontsize=16)
        plt.plot(time_list, rt_production[0:end_time], 'k', linewidth=2, label=rt_prod_label)
        plt.plot(time_list, estimated_derivative[0:end_time], '--', color='orange', linewidth=2, label=r'$\mathrm{sum}$')

        ax = plt.gca()
        if legend_on_error_plot:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_fontsize(16)
        ax.set_xlabel('time (h)', fontsize=20)
        if normalise:
            ax.set_ylabel('(W/kg)', fontsize=20)
        else:
            ax.set_ylabel('(W)', fontsize=20)
        if save:
            extra = (lgd,) if legend_on_error_plot else None
            fig.savefig(f'../results/ape_production_timeseries/error/{save_name}.png',
                        bbox_extra_artists=extra, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_ape_production_total_budget(directory, hurr, run_id, ref_state, region_key='non-sponge', save=False,
                                     end_time=None, normalise=False, title=False, smooth=False, fixed_exner=None,
                                     legend_on_error_plot=False):
    ape = np.load(f'{directory}/ape_{ref_state}_ref_at_timesteps_{run_id}.npz')
    lagr_rt, precip_rt, subgrid_rt = rt_budget(hurr)
    (lagr_theta_ei, _, _, subgrid_theta_ei,
     vertical_flux_theta_e, radcool_theta) = theta_ei_budget(hurr, fixed_exner=fixed_exner)

    lagr_prod_theta_e = multiply_by_rho(hurr, ape['G_theta_e'] * lagr_theta_ei)
    subgrid_prod_theta_e = multiply_by_rho(hurr, ape['G_theta_e'] * subgrid_theta_ei)
    radcool_prod_theta_e = multiply_by_rho(hurr, ape['G_theta_e'] * radcool_theta)

    lagr_prod_rt = multiply_by_rho(hurr, ape['G_rt'] * lagr_rt)
    subgrid_prod_rt = multiply_by_rho(hurr, ape['G_rt'] * subgrid_rt)

    colours, shapes = ape_production_budget_linestyles()
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.

    if end_time and end_time > time_list.size:
        end_time = None
    time_list = time_list[0:end_time]

    total_prod_label = r'$\overline{\rho}\left(\varepsilon_{\theta_{ei}}c_p\Pi\frac{D\theta_{ei}}{Dt} + \varepsilon_{r_t}L_s\frac{Dr_t}{Dt}\right)$'
    if isinstance(region_key, str):
        region_key = [region_key]

    for region in region_key:

        lagr_integral = vol_int(hurr, lagr_prod_theta_e+lagr_prod_rt, region_key=region)
        subgrid_integral = vol_int(hurr, subgrid_prod_theta_e+subgrid_prod_rt, region_key=region)
        radcool_integral = vol_int(hurr, radcool_prod_theta_e, region_key=region)
        precip_data = read_precip_budget(directory, hurr)
        r_fallout_prod_rt = multiply_by_rho(hurr, ape['G_rt'] * precip_data['r_fallout']) / (2. * hurr['dt'])
        i_fallout_prod_rt = multiply_by_rho(hurr, ape['G_rt'] * precip_data['i_fallout']) / (2. * hurr['dt'])
        if fixed_exner is None:
            exner = np.dstack([np.tile(hurr['pi_initial'], (hurr['m'], 1)).transpose()] * hurr['timesteps']) + hurr[
              'pi']
        else:
            exner = fixed_exner
        rl_coeff = hurr['Lf'] / (hurr['cp'] * exner)
        r_fallout_prod_theta_e = multiply_by_rho(hurr, ape['G_theta_e'] * rl_coeff * precip_data['r_fallout']) / (
                    2. * hurr['dt'])
        r_fallout_integral = vol_int(hurr, r_fallout_prod_rt + r_fallout_prod_theta_e, region_key=region)
        i_fallout_integral = vol_int(hurr, i_fallout_prod_rt, region_key=region)
        surface_flux_theta_e = hurr['rho_initial_v'][0] * integrate_area_surface(hurr,
                                                                         ape['G_theta_e'] * -vertical_flux_theta_e,
                                                                         region_key=region)
        surface_flux_rt = hurr['rho_initial_v'][0] * integrate_area_surface(hurr,
                                                                         ape['G_rt'] * -hurr['vertical_flux_rv'][:-1, :, :], region_key=region)
        surface_flux = surface_flux_theta_e + surface_flux_rt

        if normalise:
            rho_grid = np.dstack([np.tile(hurr['rho_initial_v'], (hurr['m'], 1)).transpose()] * hurr['timesteps'])
            region_mass = vol_int(hurr, rho_grid, region_key=region)
            lagr_integral = safe_div(lagr_integral, region_mass)
            subgrid_integral = safe_div(subgrid_integral, region_mass)
            radcool_integral = safe_div(radcool_integral, region_mass)
            r_fallout_integral = safe_div(r_fallout_integral, region_mass)
            i_fallout_integral = safe_div(i_fallout_integral, region_mass)
            surface_flux = safe_div(surface_flux, region_mass)

        if smooth:
            window = 3
            time_list = running_mean(time_list, window)
            lagr_integral = running_mean(lagr_integral, window)
            subgrid_integral = running_mean(subgrid_integral, window)
            radcool_integral = running_mean(radcool_integral, window)
            r_fallout_integral = running_mean(r_fallout_integral, window)
            i_fallout_integral = running_mean(i_fallout_integral, window)
            surface_flux = running_mean(surface_flux, window)

        mixing_integral = subgrid_integral - surface_flux

        fig = plt.figure(figsize=(8, 5))
        plt.plot(time_list, lagr_integral[0:end_time], 'k', linestyle=(0, (4, 2)), linewidth=2, label=total_prod_label)
        plt.plot(time_list, mixing_integral[0:end_time], color=colours['mixing'], marker=shapes['mixing'],
                 linewidth=2, label=r'$\mathregular{mixing}$')
        plt.plot(time_list, radcool_integral[0:end_time], color=colours['radcool'], marker=shapes['radcool'],
                 linewidth=2, label=r'$\mathregular{radiative\; cooling}$')
        plt.plot(time_list, i_fallout_integral[0:end_time], color=colours['ice'], marker=shapes['ice'],
                 linewidth=2, label=r'$\mathregular{ice\; fallout}$')
        plt.plot(time_list, r_fallout_integral[0:end_time], color=colours['precip'], marker=shapes['precip'],
                 linewidth=2, label=r'$\mathregular{rain\; fallout}$')
        plt.plot(time_list, surface_flux[0:end_time], color=colours['surface_flux'], marker=shapes['surface_flux'],
                 linewidth=2, label=r'$\mathregular{surface\; flux}$')

        ax = plt.gca()
        ax.axhline(0, color='k', linewidth=1)
        ax.set_xlim([0, 250])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('time (h)', fontsize=20)
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_fontsize(16)
        if normalise:
            ax.set_ylabel(r'$\mathregular{\left(Wkg^{-1}\right)}$', fontsize=20)
        else:
            ax.set_ylabel(r'$\mathregular{\left(W\right)}$', fontsize=20)
        if title:
            plot_title = f'total APE production budget - {run_id} {region}'
            ax.set_title(plot_title, fontsize=18)
        if save:
            save_name = f'{clean_filename(region)}_total_production_budget_{ref_state}_ref_{run_id}'
            if normalise:
                save_name += '_normalised'
            if fixed_exner is not None:
                save_name += '_fixexner'
            fig.savefig(f'../results/ape_production_timeseries/{save_name}.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            fig.savefig(f'../results/ape_production_timeseries/{save_name}.pdf',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        ape_budget = np.load(f'{directory}/ape_{ref_state}_ref_budget_{run_id}.npz')
        estimated_derivative = r_fallout_integral + i_fallout_integral + subgrid_integral + radcool_integral
        ape_budget = time_jump_masker(ape_budget)

        total_production = vol_int(hurr, ape_budget['theta_e_prod'] + ape_budget['rt_prod'], region_key=region)
        if normalise:
            total_production = safe_div(total_production, region_mass)
        if smooth:
            total_production = running_mean(total_production, window)

        fig = plt.figure(figsize=(6, 4.5))
        if title:
            plt.title(plot_title, fontsize=16)
        plt.plot(time_list, estimated_derivative[0:end_time], 'k', linewidth=1.5, label=total_prod_label)
        plt.plot(time_list, total_production[0:end_time], color='orange', linestyle=(0, (3, 3)),
                 linewidth=1.5,
                 label=r'$\mathrm{sum}$')

        ax = plt.gca()
        ax.set_xlim([0, 250])
        if legend_on_error_plot:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('time (h)', fontsize=20)
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_fontsize(16)
        if normalise:
            ax.set_ylabel(r'$\mathregular{\left(Wkg^{-1}\right)}$', fontsize=20)
        else:
            ax.set_ylabel(r'$\mathregular{\left(W\right)}$', fontsize=20)
        if save:
            extra = (lgd,) if legend_on_error_plot else None
            fig.savefig(f'../results/ape_production_timeseries/error/{save_name}.pdf',
                        bbox_inches='tight', bbox_extra_artists=extra)
            fig.savefig(f'../results/ape_production_timeseries/error/{save_name}.png',
                        bbox_inches='tight', bbox_extra_artists=extra)
            plt.close()
        else:
            plt.show()


def ape_surface_fluxes(directory, hurr, run_id, ref_state, region_key='non-sponge', save=False):
    ape = np.load(f'{directory}/ape_{ref_state}_ref_at_timesteps_{run_id}.npz')

    exner = np.dstack([np.tile(hurr['pi_initial'], (hurr['m'], 1)).transpose()]*hurr['timesteps']) + hurr['pi']
    rv_coeff = hurr['Ls']/(hurr['cp']*exner)

    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.

    theta_e_flux = hurr['rho_initial_v'][0] * ape['G_theta_e'] * (
                hurr['vertical_flux_theta'][:-1, :, :] + rv_coeff * hurr['vertical_flux_rv'][:-1, :, :])
    theta_e_flux_sensible = hurr['rho_initial_v'][0] * ape['G_theta_e'] *hurr['vertical_flux_theta'][:-1, :, :]
    theta_e_flux_latent = hurr['rho_initial_v'][0] * ape['G_theta_e'] * (rv_coeff * hurr['vertical_flux_rv'][:-1, :, :])
    rt_flux = hurr['rho_initial_v'][0] * ape['G_rt'] * hurr['vertical_flux_rv'][:-1, :, :]

    areas = np.zeros_like(hurr['v'])
    areas[0, :, :] = np.tile(column_areas(hurr), (hurr['timesteps'], 1)).T
    areas = mask_array(hurr, areas, region_key=region_key)
    total_area = np.sum(areas)

    area_avg_theta_e = -safe_div(integrate_area_surface(hurr, theta_e_flux, region_key=region_key), total_area)
    area_avg_theta_e_sensible = -safe_div(integrate_area_surface(hurr, theta_e_flux_sensible, region_key=region_key), total_area)
    area_avg_theta_e_latent = -safe_div(integrate_area_surface(hurr, theta_e_flux_latent, region_key=region_key), total_area)
    area_avg_rt = -safe_div(integrate_area_surface(hurr, rt_flux, region_key=region_key), total_area)
    area_avg_moisture = -safe_div(integrate_area_surface(hurr, theta_e_flux_latent + rt_flux, region_key=region_key), total_area)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(time_list, area_avg_theta_e_sensible, color='#29A022', linewidth=2, label='$\mathregular{\\theta_{ei}}$ sensible')
    plt.plot(time_list, area_avg_theta_e_latent,  color='#FB8072', linewidth=2, label='$\mathregular{\\theta_{ei}}$ latent')
    plt.plot(time_list, area_avg_rt, color='#10659E', linewidth=2, label='$\mathregular{r_t}$')
    plt.plot(time_list, area_avg_moisture, linestyle=(0, (4, 2)), color='gray', linewidth=2, label='moisture total')
    ax = plt.gca()
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xlim([0, 250])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
    ax.tick_params(labelsize=16)
    ax.set_xlabel('time (h)', fontsize=20)
    ax.set_ylabel(r"$\mathregular{surface\; flux}$"
           "\n"  # Newline: the backslash is interpreted as usual
           r"$\mathregular{APE\; production\; \left(Wm^{-2}\right)}$", fontsize=18)
    plt.savefig(f'../results/ape_surface_flux_components_{clean_filename(region_key)}_{ref_state}_{run_id}.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    data_dir = '../data/J30pt3'
    run_id = 'J30pt3'
    hurr = read_fortran_output(data_dir)
    region_key = 'outer_inflow'
    ape_surface_fluxes(data_dir, hurr, run_id, 'initial', region_key='outer_inflow', save=True)
    plot_ape_production_total_budget(data_dir, hurr, run_id, 'initial', region_key=region_key, save=True)

