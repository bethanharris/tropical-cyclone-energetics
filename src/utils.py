import numpy as np
import os
import string
from matplotlib.ticker import ScalarFormatter


def safe_log(x):
    if isinstance(x, np.float32) or isinstance(x, np.float64) or isinstance(x, float) or len(x.shape) == 0:
        x = np.array([x])
    logx = np.zeros_like(x, dtype=float)
    loggable = np.where(np.logical_and(np.isfinite(x), x > 0.))
    logx[loggable] = np.log(x[loggable])
    return logx


def safe_div(x, y):
    zeros = np.where(y == 0.)
    div = x/y
    div[zeros] = 0.
    return div


def running_mean(x, N):
    if N < 1:
        return x
    else:
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)


def file_cleanup(directory, patterns_to_delete):
    for filename in os.listdir(directory):
        if filename.startswith(tuple(patterns_to_delete)):
            try:
                os.remove(os.path.join(directory, filename))
            except:
                print('Could not delete file %s.' % filename)


def clean_filename(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned_filename = ''.join(c for c in filename if c in valid_chars)
    return cleaned_filename


def filename_to_label(filename):
    if 'production' in filename:
        filename = filename.split('production_')[1].strip('.npz')
    sounding, temperature = filename.split('_')
    temperature = temperature.replace('pt', '.')
    label = sounding[0].upper() + temperature
    return label


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self, vmin, vmax):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


def multiply_by_rho(hurr, data):
    rho_bar_v_grid = np.tile(hurr['rho_initial_v'], (hurr['m'], 1)).transpose()
    rho_bar_v_time_grid = np.dstack([rho_bar_v_grid] * hurr['timesteps'])
    return rho_bar_v_time_grid * data


def column_to_time_grid(hurr, column):
    grid = np.tile(column, (hurr['m'], 1)).transpose()
    time_grid = np.dstack([grid]*hurr['timesteps'])
    return time_grid


def row_to_time_grid(hurr, row):
    grid = np.tile(row, (hurr['n'], 1))
    time_grid = np.dstack([grid]*hurr['timesteps'])
    return time_grid


def time_profiles_to_grid(hurr, profiles):
    time_profile_grid = np.zeros((profiles.shape[0], hurr['m'], profiles.shape[1]))
    for t in range(hurr['timesteps']):
        time_profile = profiles[:, t]
        time_profile_grid[:, :, t] = np.tile(time_profile, (hurr['m'], 1)).transpose()
    return time_profile_grid


def column_areas(hurr):
    areas = np.array([np.pi * (hurr['r_list_u'][i+1] ** 2) - np.pi * (hurr['r_list_u'][i] ** 2) for i in
                      np.arange(hurr['m'])])
    return areas


def parcel_volumes(hurr):
    areas = column_areas(hurr)
    vols = areas * hurr['dz']
    volume_grid = row_to_time_grid(hurr, vols)
    return volume_grid


def flux_through_boundaries(hurr, data, inner_r_idx, outer_r_idx, bottom_z_idx, top_z_idx):
    #break down flux integral into boundary terms around region [inner_r_idx:outer_r_idx+1,bottom_z_idx:top_z_idx+1]
    #(i.e. edge indices are included in the volume integrated over)
    inner_r = np.zeros((hurr['n'], hurr['timesteps']))
    outer_r = np.zeros((hurr['n'],hurr['timesteps']))
    top_z = np.zeros((hurr['m'], hurr['timesteps']))
    bottom_z = np.zeros((hurr['m'],hurr['timesteps']))

    for t in range(hurr['timesteps']):
        for j in range(bottom_z_idx, top_z_idx+1):
            inner_r[j, t] = hurr['dz'] * np.pi * hurr['rho_initial_v'][j] * hurr['u'][j, inner_r_idx, t] * \
                            hurr['r_list_u'][inner_r_idx] * (
                                    data[j, inner_r_idx, t] + data[j, inner_r_idx-1, t])
            outer_r[j, t] = hurr['dz'] * np.pi * hurr['rho_initial_v'][j] * hurr['u'][
                j, outer_r_idx + 1, t] * \
                            hurr['r_list_u'][outer_r_idx + 1] * (
                                    data[j, outer_r_idx + 1, t] + data[j, outer_r_idx, t])
        for i in range(inner_r_idx, outer_r_idx+1):
            top_z[i, t] = hurr['dr'] * np.pi * hurr['r_list_v'][i] * hurr['rho_initial_w'][top_z_idx + 1] * \
                          hurr['w'][top_z_idx + 1, i, t] * (
                                      data[top_z_idx + 1, i, t] + data[top_z_idx, i, t])
            bottom_z[i, t] = hurr['dr'] * np.pi * hurr['r_list_v'][i] * hurr['rho_initial_w'][bottom_z_idx] * \
                          hurr['w'][bottom_z_idx, i, t] * (
                                      data[bottom_z_idx, i, t] + data[bottom_z_idx-1, i, t])

    total_integral_r = np.sum(outer_r - inner_r, axis=0)
    total_integral_z = np.sum(top_z - bottom_z, axis=0)
    total_integral = total_integral_r + total_integral_z

    total_outer = np.sum(outer_r, axis=0)
    total_inner = -np.sum(inner_r, axis=0)
    total_top = np.sum(top_z, axis=0)
    total_bottom = -np.sum(bottom_z, axis=0)
    return total_integral, total_outer, total_top, total_inner, total_bottom, inner_r, outer_r, top_z, bottom_z


def construct_full_profile(hurr, v_profile, surface, top):
    w_profile = 0.5*(v_profile[1:] + v_profile[:-1])
    full_profile = np.empty((2*hurr['n'] + 1, ))
    full_profile[0] = surface
    full_profile[1::2] = v_profile
    full_profile[2:-1:2] = w_profile
    full_profile[-1] = top
    return full_profile


def construct_full_grid(hurr, v_grid, surface):
    w_grid = 0.5*(v_grid[1:, :, :] + v_grid[:-1, :, :])
    full_grid = np.empty((2*hurr['n'] + 1, hurr['m'], hurr['timesteps']))
    full_grid[0, :, :] = surface
    full_grid[1::2, :, :] = v_grid
    full_grid[2:-1:2, :, :] = w_grid
    full_grid[-1, :, :] = v_grid[-1, :, :] + (v_grid[-1, :, ] - w_grid[-2, :, ])
    return full_grid

