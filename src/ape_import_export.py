import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import flux_through_boundaries, column_to_time_grid
from model_reader import read_fortran_output
from volume_integral import vol_int


def get_ape_density(directory, run_id):
    ape_data = np.load(f'{directory}/ape_initial_ref_at_timesteps_{run_id}.npz')
    return ape_data['ape_density']


def get_ape_flux(directory, run_id):
    ape_budget = np.load(f'{directory}/ape_initial_ref_ice_precip_budget_{run_id}.npz')
    return ape_budget['ape_flux']


def get_boundary_contributions(directory, run_id, outer_radius_km):
    ape_density = get_ape_density(directory, run_id)
    hurr = read_fortran_output(directory)
    radial_idx = int(round(outer_radius_km * 1000. / hurr['dr']) - 1)
    total_integral, total_outer, total_top, total_inner, total_bottom, inner_r, outer_r, top_z, \
    bottom_z = flux_through_boundaries(hurr, ape_density, 0, radial_idx, 0, hurr['n'] - 2)
    return total_integral, total_outer, total_top, total_inner, total_bottom, inner_r, outer_r, top_z, bottom_z


def check_flux_integral(directory, run_id, outer_radius_km):
    ape_flux = get_ape_flux(directory, run_id)
    hurr = read_fortran_output(directory)
    time = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.
    total_integral = get_boundary_contributions(directory, run_id, outer_radius_km)[0]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.plot(time, vol_int(hurr, -ape_flux, region_key=f'r<{int(outer_radius_km)}'), 'k-', linewidth=2,
             label='integral over region')
    plt.plot(time, -total_integral, '--', color='gray', linewidth=2, label='sum over boundaries')
    ax.tick_params(labelsize=16)
    ax.set_xlabel('time (h)', fontsize=18)
    ax.set_ylabel('APE density flux convergence (W)', fontsize=16)
    plt.legend(fontsize=16)
    plt.title(f'r < {int(outer_radius_km)} km', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_boundary_contributions(directory, run_id, outer_radius_km):
    hurr = read_fortran_output(directory)
    total_integral, total_outer, total_top, total_inner, total_bottom = get_boundary_contributions(directory, run_id,
                                                                                                   outer_radius_km)[0:5]
    time = (np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.plot(time, -total_inner, linewidth=2, label='inner')
    plt.plot(time, -total_outer, linewidth=2, label='outer')
    plt.plot(time, -total_top, linewidth=2, label='top')
    plt.plot(time, -total_bottom, linewidth=2, label='bottom')
    plt.plot(time, -total_integral, '--', color='gray', linewidth=2, label='total')
    ax.tick_params(labelsize=16)
    ax.set_xlabel('time (h)', fontsize=18)
    ax.set_ylabel('APE density flux across boundary (W)', fontsize=16)
    plt.legend(fontsize=16)
    plt.title(f'r < {int(outer_radius_km)} km', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_ape_import(directory, run_id, hurr, timestep, outer_radius_km=300, title=False, save=False):
    ape_data = np.load(f'{directory}/ape_initial_ref_at_timesteps_{run_id}.npz')
    ape_density = ape_data['ape_density']
    z_list = np.arange(0, hurr['n']*hurr['dz'], hurr['dz']) + 0.5 * hurr['dz']
    z_km = z_list/1000.
    time = (np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.)[timestep]
    z_grid = column_to_time_grid(hurr, z_list)
    radial_idx = int(round(outer_radius_km * 1000. / hurr['dr']) - 1)
    outer_radial_flux = flux_through_boundaries(hurr, ape_density, 0, radial_idx, 0, 42)[6]
    total_import = -flux_through_boundaries(hurr, ape_density, 0, radial_idx, 0, 42)[1][timestep]
    total_import_raw_string = f'{total_import:0.2e}'
    mantissa, exponent = total_import_raw_string.split('e+')
    total_import_string = rf'${mantissa} \times 10^{{{exponent}}}$ W'

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))

    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    plt.plot(-outer_radial_flux[:, timestep], z_km, '-o', linewidth=2)
    plt.text(0.7, 0.5, f'net import:\n{total_import_string}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    ax.tick_params(labelsize=16)
    ax.set_ylim(bottom=0.)
    ax.set_ylabel('z (km)', fontsize=20)
    ax.set_xlabel('APE flux (W)', fontsize=20)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.offsetText.set_fontsize(16)
    if title:
        plt.title(f'r = {int(outer_radius_km)} km, t = {int(time)} h', fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(f'../results/radial_ape_flux_r_{int(outer_radius_km)}_t_{int(time)}.pdf')
    plt.show()


if __name__ == '__main__':
    directory = '../data/J30pt3'
    run_id = 'J30pt3'
    hurr = read_fortran_output(directory)
    plot_ape_import(directory, run_id, hurr, 199, save=True)
