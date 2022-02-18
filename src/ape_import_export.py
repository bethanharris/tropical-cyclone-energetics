import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import flux_through_boundaries
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


def plot_outer_boundary_flux(directory, run_id, outer_radius_km, timestep, title=False, save=True):
    hurr = read_fortran_output(directory)
    outer_radial_flux = get_boundary_contributions(directory, run_id, outer_radius_km)[6]
    time = (np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.)[timestep]
    z_km = (np.arange(0, hurr['dz']*hurr['n'], hurr['dz']) + 0.5 * hurr['dz'])/1000.

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))

    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    plt.plot(-outer_radial_flux[:, timestep], z_km, '-o', linewidth=2)
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
