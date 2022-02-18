import numpy as np
import thermo
import gma


def region_masks(hurr):
    z_list = np.arange(0, hurr['dz']*hurr['n'], hurr['dz']) + (hurr['dz']/2.)

    r_grid = np.tile(hurr['r_list_v'], (hurr['n'], 1))
    r_time_grid = np.dstack([r_grid]*hurr['timesteps'])
    z_grid = np.tile(z_list.transpose(), (hurr['m'], 1)).transpose()
    z_time_grid = np.dstack([z_grid]*hurr['timesteps'])

    sponge_bound_r = hurr['r_list_v'][hurr['m']-hurr['sponge_r']]
    sponge_bound_z = z_list[hurr['n'] - hurr['sponge_z']]

    core = np.where(np.logical_and(hurr['v'] > 33., r_time_grid < 1e6))
    cyclone = np.where(np.logical_and(hurr['v'] > 10., r_time_grid < 1.5e6))
    anticyclone = np.where(hurr['v'] < -10.)
    sponge = np.where(np.logical_or(r_time_grid >= sponge_bound_r, z_time_grid >= sponge_bound_z))

    mask_data = np.zeros_like(hurr['v'])
    mask_data[cyclone] = 1
    mask_data[core] = 2
    mask_data[anticyclone] = 3
    mask_data[sponge] = 4

    return mask_data


def rmw_mask(hurr):
    r_grid = np.tile(hurr['r_list_v'], (hurr['n'], 1))
    r_time_grid = np.dstack([r_grid] * hurr['timesteps'])

    rmw_grid = np.zeros_like(hurr['v'])

    for t in range(hurr['timesteps']):
        near_surface_wind = hurr['v'][0:5, :, t]
        rmw_idx = np.where(near_surface_wind == near_surface_wind.max())[1][0]
        rmw = hurr['r_list_v'][rmw_idx]
        rmw_grid[:, :, t] = np.ones((hurr['n'], hurr['m']))*rmw

    inside_rmw = r_time_grid < rmw_grid

    return inside_rmw


def mask_array(hurr, masked_variable, region_key):
    if region_key != 'all':
        region_mask = region_masks(hurr)
        if region_key.startswith('v>'):
            wind_threshold = float(region_key.split('>')[1])
            mask = np.where(np.logical_or(region_mask == 4, hurr['v'] < wind_threshold))
            masked_variable[mask] = 0.
        elif region_key.startswith('v<'):
            wind_threshold = float(region_key.split('<')[1])
            mask = np.where(np.logical_or(region_mask == 4, hurr['v'] > wind_threshold))
            masked_variable[mask] = 0.
        elif region_key.startswith('r<'):
            radius_threshold = float(region_key.split('<')[1])
            r_grid = np.dstack([np.tile(hurr['r_list_v']/1000., (hurr['n'], 1))]*hurr['timesteps'])
            mask = np.where(np.logical_or(region_mask == 4, r_grid > radius_threshold))
            masked_variable[mask] = 0.
        elif region_key.startswith('v_pc_'):
            wind_percentile = float(region_key.split('_')[-1])
            for t in range(hurr['timesteps']):
                winds_at_time = hurr['v'][:, :, t]
                positive_winds = winds_at_time[np.where(winds_at_time > 0.)]
                wind_threshold = np.percentile(positive_winds, wind_percentile)
                mask = np.where(winds_at_time < wind_threshold)
                (masked_variable[:, :, t])[mask] = 0.
        elif region_key.startswith('M<'):
            angmom_threshold = float(region_key.split('<')[1])
            r_grid = np.dstack([np.tile(hurr['r_list_v'], (hurr['n'], 1))]*hurr['timesteps'])
            ang_mom = r_grid*hurr['v'] + 0.5*hurr['f']*(r_grid**2)
            mask = np.where(np.logical_or(region_mask == 4, ang_mom > angmom_threshold))
            masked_variable[mask] = 0.
        elif region_key.startswith('j='):
            level = float(region_key.split('=')[1])
            masked_variable[np.arange(hurr['n']) != level, :, :] = 0.
        elif region_key.startswith('i='):
            column = float(region_key.split('=')[1])
            masked_variable[:, np.arange(hurr['m']) != column, :] = 0.
        elif region_key.startswith('parcel-'):
            j, i = [int(num) for num in region_key.split('-')[1:]]
            masked_variable[np.arange(hurr['n']) != j, :, :] = 0.
            masked_variable[:, np.arange(hurr['m']) != i, :] = 0.
        else:
            if region_key == 'core':
                mask = np.where(region_mask != 2)
                masked_variable[mask] = 0.
            elif region_key == 'cyclone':
                mask = np.where(np.logical_and(region_mask != 1, region_mask != 2))
                masked_variable[mask] = 0.
            elif region_key == 'cyclone-not-core':
                mask = np.where(region_mask != 1)
                masked_variable[mask] = 0.
            elif region_key == 'anticyclone':
                mask = np.where(region_mask != 3)
                masked_variable[mask] = 0.
            elif region_key == 'non-sponge':
                mask = np.where(region_mask == 4)
                masked_variable[mask] = 0.
            elif region_key == 'outer':
                mask = np.where(region_mask != 0)
                masked_variable[mask] = 0.
            elif region_key == 'sponge':
                mask = np.where(region_mask != 4)
                masked_variable[mask] = 0.
            elif region_key == 'surface':
                mask = np.where(region_mask == 4)
                masked_variable[mask] = 0.
                masked_variable[1:, :, :] = 0.
            elif region_key == 'inner':
                masked_variable[25:, :, :] = 0.
                masked_variable[:, 50:, :] = 0.
            elif region_key == 'inside_rmw':
                inside_rmw = rmw_mask(hurr)
                mask = np.where(inside_rmw==0)
                masked_variable[mask] = 0.
            elif region_key == 'outside_rmw':
                inside_rmw = rmw_mask(hurr)
                mask = np.where(inside_rmw)
                masked_variable[mask] = 0.
            elif region_key == 'cyclone_inside_rmw':
                inside_rmw = rmw_mask(hurr)
                mask = np.where(np.logical_or(inside_rmw == 0, np.logical_and(region_mask != 1, region_mask != 2)))
                masked_variable[mask] = 0.
            elif region_key == 'cyclone_outside_rmw':
                inside_rmw = rmw_mask(hurr)
                mask = np.where(np.logical_or(inside_rmw, np.logical_and(region_mask != 1, region_mask != 2)))
                masked_variable[mask] = 0.
            elif region_key == 'inflow':
                masked_variable[:, 800:, :] = 0.
                masked_variable[3:, :, :] = 0.
            elif region_key == 'outer_inflow':
                masked_variable[:, 0:120, :] = 0.
                masked_variable[:, 800:, :] = 0.
                masked_variable[3:, :, :] = 0.
            elif region_key == 'outer_inflow_j0':
                masked_variable[:, 0:40, :] = 0.
                masked_variable[:, 800:, :] = 0.
                masked_variable[1:, :, :] = 0.
            elif region_key == 'outflow':
                masked_variable[:, 600:, :] = 0.
                masked_variable[0:15, :, :] = 0.
                masked_variable[30:, :, :] = 0.
            else:
                raise KeyError('Invalid region specification %s' % region_key)
    return masked_variable


def vol_int(hurr, time_array, region_key='all'):
    masked_array = mask_array(hurr, np.copy(time_array), region_key)
    r_grid = np.dstack([np.tile(hurr['r_list_v'], (hurr['n'], 1))]*hurr['timesteps'])
    integrand = r_grid*masked_array
    volume_integral = 2.*np.pi*np.sum(np.sum(integrand * hurr['dz'], axis=1) * hurr['dr'], axis=0)
    return volume_integral


def integrate_area_surface(hurr, data, region_key='non-sponge'):
    masked_data = mask_array(hurr, np.copy(data), region_key)
    integral = [2.*np.pi*np.sum(hurr['dr']*hurr['r_list_v']*masked_data[0, :, t]) for t in range(hurr['timesteps'])]
    return np.array(integral)
