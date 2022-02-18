from model_reader import read_fortran_output
from energy_budget_timeseries import energy_budget_linestyles
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from volume_integral import vol_int, integrate_area_surface, mask_array
from diabatic_budgets import theta_ei_budget, rt_budget
from matplotlib.patches import Rectangle
from utils import row_to_time_grid, column_to_time_grid, multiply_by_rho, clean_filename, column_areas, safe_div


# colormap scaling code authors: Paul H, Horea Christian, Leonor Carcia Gutierrez.
#  Modified to work properly when abs(min) > max

def auto_remap(data):
    start = 0
    midpoint = 0.5
    stop = 1.0
    if np.nanmin(data) >= 0:
        raise ValueError('You do not need to rescale your cmap to center zero.')
    if np.nanmax(data) > abs(np.nanmin(data)):
        start = (np.nanmax(data) - abs(np.nanmin(data))) / (2. * np.nanmax(data))
        midpoint = abs(np.nanmin(data)) / (np.nanmax(data) + abs(np.nanmin(data)))
        stop = 1.0
    if np.nanmax(data) == abs(np.nanmin(data)):
        start = 0
        midpoint = 0.5
        stop = 1.0
    if np.nanmax(data) < abs(np.nanmin(data)):
        start = 0
        midpoint = abs(np.nanmin(data)) / (np.nanmax(data) + abs(np.nanmin(data)))
        stop = (abs(np.nanmin(data)) + np.nanmax(data)) / (2. * abs(np.nanmin(data)))
    return start, midpoint, stop


def remappedColorMap(cmap, data, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range. Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.
    Input
    -----
    cmap : The matplotlib colormap to be altered
    data: You can provide your data as a numpy array, and the following
        operations will be computed automatically for you.
    start : Offset from lowest point in the colormap's range.
        Defaults to 0.0 (no lower ofset). Should be between
        0.0 and 0.5; if your dataset vmax <= abs(vmin) you should leave
        this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax)
    midpoint : The new center of the colormap. Defaults to
        0.5 (no shift). Should be between 0.0 and 1.0; usually the
        optimal value is abs(vmin)/(vmax+abs(vmin))
    stop : Offset from highets point in the colormap's range.
        Defaults to 1.0 (no upper ofset). Should be between
        0.5 and 1.0; if your dataset vmax >= abs(vmin) you should leave
        this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin))
    '''

    start, midpoint, stop = auto_remap(data)

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.hstack([
        np.linspace(start, 0.5, 128, endpoint=False),
        np.linspace(0.5, stop, 129)
    ])

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def azimuthal_flow(directory, run_id, timestep, min_v, max_v):
    hurr = read_fortran_output(directory)
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.
    time = time_list[timestep]

    dr_km = hurr['dr'] / 1000.
    dz_km = hurr['dz'] / 1000.

    r_km = np.arange(0, hurr['m'] * dr_km, dr_km)
    z_km = np.arange(0, hurr['n'] * dz_km, dz_km)

    v = hurr['v'][:, :, timestep]
    R, Z = np.meshgrid(r_km, z_km)
    levels = np.arange(min_v, max_v+1, 10)

    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    plt.contourf(R, Z, v, levels=levels, cmap=remappedColorMap(cm.PuOr, np.array([min_v, max_v])))
    ax.set_xlabel('r (km)', fontsize=22)
    ax.set_ylabel('z (km)', fontsize=22)
    ax.set_xlim([0, 2500])
    ax.set_ylim([0, 20])
    ax.tick_params(labelsize=16)
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathregular{v\; \left(ms^{-1}\right)}$', fontsize=24, labelpad=0)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(f'../results/v_{int(time)}h_{run_id}.png', dpi=400)
    plt.show()


def ape_snapshot(directory, run_id, timestep, ref_state='initial'):
    hurr = read_fortran_output(directory)
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.
    time = time_list[timestep]
    ape_data = np.load(f'{directory}/ape_{ref_state}_ref_at_timesteps_{run_id}.npz')
    ape_density = ape_data['ape_density'][:, :, timestep]

    dr_km = hurr['dr'] / 1000.
    dz_km = hurr['dz'] / 1000.
    r_km = np.arange(0, hurr['m'] * dr_km, dr_km)
    z_km = np.arange(0, hurr['n'] * dz_km, dz_km)
    R, Z = np.meshgrid(r_km, z_km)

    plt.figure(figsize=(6,4.5))
    ax = plt.gca()

    plt.pcolormesh(R, Z, ape_density, vmin=0., vmax=6000., cmap=cm.gist_heat_r)
    cbar = plt.colorbar(extend='both')
    ax.set_xlabel('r (km)', fontsize=22)
    ax.set_ylabel('z (km)', fontsize=22)
    ax.set_xlim([0, 2500])
    ax.set_ylim([0, 20])
    ax.tick_params(labelsize=16)
    cbar.set_label(r'$\mathregular{e_a\; \left(Jkg^{-1}\right)}$', fontsize=24, labelpad=5)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(f'../results/ea_{ref_state}_{int(time)}h_{run_id}.pdf')
    plt.show()


def zr_snapshot(directory, run_id, timestep, ref_state='initial'):
    hurr = read_fortran_output(directory)
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.
    time = time_list[timestep]
    ape_data = np.load(f'{directory}/ape_{ref_state}_ref_at_timesteps_{run_id}.npz')
    reference_height = ape_data['z_r'][0:hurr['no_sponge_z'], 0:hurr['no_sponge_r'], timestep]
    z_grid = column_to_time_grid(hurr, np.arange(0, hurr['n'] * hurr['dz'], hurr['dz']) + 0.5 * hurr['dz'])
    zr_diff = reference_height - z_grid[0:hurr['no_sponge_z'], 0:hurr['no_sponge_r'], timestep]

    dr_km = hurr['dr'] / 1000.
    dz_km = hurr['dz'] / 1000.
    r_km = np.arange(0, hurr['m'] * dr_km, dr_km)
    z_km = np.arange(0, hurr['n'] * dz_km, dz_km)
    R, Z = np.meshgrid(r_km, z_km)
    R = R[0:hurr['no_sponge_z'], 0:hurr['no_sponge_r']]
    Z = Z[0:hurr['no_sponge_z'], 0:hurr['no_sponge_r']]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='w', edgecolor='k')

    ax = axs[0]
    zr_mesh = ax.pcolormesh(R, Z, reference_height/1000., vmin=0., vmax=20., cmap=cm.viridis)
    cbar = fig.colorbar(zr_mesh, ax=ax)
    ax.set_xlabel('r (km)', fontsize=22)
    ax.set_ylabel('z (km)', fontsize=22)
    ax.set_xlim([0, 2500])
    ax.set_ylim([0, 20])
    ax.tick_params(labelsize=16)
    cbar.set_label(r'$\mathregular{z_r\; \left(km\right)}$', fontsize=22, labelpad=5)
    cbar.ax.tick_params(labelsize=16)

    ax = axs[1]
    zrdiff_mesh = ax.pcolormesh(R, Z, zr_diff/1000., cmap=remappedColorMap(cm.BrBG, zr_diff))
    cbar = fig.colorbar(zrdiff_mesh, ax=ax)
    ax.set_xlabel('r (km)', fontsize=22)
    ax.set_ylabel('z (km)', fontsize=22)
    ax.set_xlim([0, 2500])
    ax.set_ylim([0, 20])
    ax.tick_params(labelsize=16)
    cbar.set_label(r'$\mathregular{z_r-z\; \left(km\right)}$', fontsize=22, labelpad=5)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(f'../results/zr_{ref_state}_{int(time)}h_{run_id}.png', dpi=400)
    plt.show()


def efficiencies_snapshot(directory, run_id, timestep, ref_state='initial'):
    hurr = read_fortran_output(directory)
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.
    time = time_list[timestep]
    ape_data = np.load(f'{directory}/ape_{ref_state}_ref_at_timesteps_{run_id}.npz')
    G_theta_ei = ape_data['G_theta_e'][0:hurr['no_sponge_z'], 0:hurr['no_sponge_r'], timestep]
    G_rt = ape_data['G_rt'][0:hurr['no_sponge_z'], 0:hurr['no_sponge_r'], timestep]

    exner = hurr['pi'] + column_to_time_grid(hurr, hurr['pi_initial'])
    eff_theta_ei = G_theta_ei/(hurr['cp'] * exner[0:hurr['no_sponge_z'], 0:hurr['no_sponge_r'], timestep])
    eff_rt = G_rt/hurr['Ls']

    dr_km = hurr['dr'] / 1000.
    dz_km = hurr['dz'] / 1000.
    r_km = np.arange(0, hurr['m'] * dr_km, dr_km)
    z_km = np.arange(0, hurr['n'] * dz_km, dz_km)
    R, Z = np.meshgrid(r_km, z_km)
    R = R[0:hurr['no_sponge_z'], 0:hurr['no_sponge_r']]
    Z = Z[0:hurr['no_sponge_z'], 0:hurr['no_sponge_r']]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='w', edgecolor='k')

    ax = axs[0]
    G_thetae_mesh = ax.pcolormesh(R, Z, eff_theta_ei, cmap=remappedColorMap(cm.seismic, eff_theta_ei))
    cbar = fig.colorbar(G_thetae_mesh, ax=ax)
    ax.set_xlabel('r (km)', fontsize=22)
    ax.set_ylabel('z (km)', fontsize=22)
    ax.set_xlim([0, 2500])
    ax.set_ylim([0, 20])
    ax.tick_params(labelsize=16)
    cbar.set_label(r'$\mathregular{\varepsilon_{\theta_{ei}}}$', fontsize=26, labelpad=5)
    cbar.ax.tick_params(labelsize=16)

    ax = axs[1]
    Grt_mesh = ax.pcolormesh(R, Z, eff_rt, cmap=remappedColorMap(cm.seismic, eff_rt))
    cbar = fig.colorbar(Grt_mesh, ax=ax)
    ax.set_xlabel('r (km)', fontsize=22)
    ax.set_ylabel('z (km)', fontsize=22)
    ax.set_xlim([0, 2500])
    ax.set_ylim([0, 20])
    ax.tick_params(labelsize=16)
    cbar.set_label(r'$\mathregular{\varepsilon_{r_t}}$', fontsize=26, labelpad=5)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.offsetText.set_fontsize(16)

    plt.tight_layout()
    plt.savefig(f'../results/efficiencies_{ref_state}_{int(time)}h_{run_id}.png', dpi=400)
    plt.show()


def diabatic_ape_production_inflow_box(directory, run_id, hurr, ref_state, timestep, title=False):

    ape_data = np.load(f'{directory}/ape_{ref_state}_ref_at_timesteps_{run_id}.npz')

    lagr_rt, _, _= rt_budget(hurr)
    lagr_theta_e, _, _, _, _, _ = theta_ei_budget(hurr)
    lagr_prod_theta_e = multiply_by_rho(hurr, ape_data['G_theta_e'] * lagr_theta_e)
    lagr_prod_rt = multiply_by_rho(hurr, ape_data['G_rt'] * lagr_rt)

    r_grid = row_to_time_grid(hurr, hurr['r_list_v'])
    total_production = (2.*np.pi*r_grid*hurr['dr']*hurr['dz']*(lagr_prod_theta_e + lagr_prod_rt))[:, :, timestep]

    dr_km = hurr['dr'] / 1000.
    dz_km = hurr['dz'] / 1000.
    r_km = np.arange(0, hurr['m'] * dr_km, dr_km)
    z_km = np.arange(0, hurr['n'] * dz_km, dz_km)
    R, Z = np.meshgrid(r_km, z_km) 
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.
    time = time_list[timestep]

    upper_limit = np.percentile(total_production, 99.95)
    lower_limit = np.percentile(total_production, 0.1)
    cmap = remappedColorMap(cm.seismic, data=np.array([lower_limit, upper_limit]))

    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()

    cf = ax.pcolormesh(R, Z, total_production, cmap=cmap, vmin=lower_limit, vmax=upper_limit)
    ax.set_xlim([0., 2500.])
    ax.set_ylim([0., 15.])

    if title:
        ax.set_title(f't = {int(time)} h', fontsize=16)
    ax.tick_params(labelsize=16)

    ax.set_xlabel('r (km)', fontsize=22)
    ax.set_ylabel('z (km)', fontsize=22)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = plt.colorbar(cf, extend='both', format=fmt)
    cbar.set_label('total diabatic production (W)', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.get_offset_text().set(size=16)

    inflow_box = Rectangle((300, 0.), 1700, 1.875, linewidth=3, edgecolor='k', linestyle='--', alpha=0.7,
                            facecolor='none')
    ax.add_patch(inflow_box)
    plt.tight_layout()
    plt.savefig(f'../results/total_ape_production_{ref_state}_{int(time)}h_inflow_box_{run_id}.pdf', dpi=400)
    plt.show()
    return upper_limit, lower_limit


if __name__ == '__main__':
    directory = '../data/J30pt3'
    run_id = 'J30pt3'
    hurr = read_fortran_output(directory)
    azimuthal_flow(directory, run_id, 29, -50, 70)
    ape_snapshot(directory, run_id, 29)
    zr_snapshot(directory, run_id, 29)
    efficiencies_snapshot(directory, run_id, 29)
