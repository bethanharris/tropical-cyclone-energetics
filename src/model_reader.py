import numpy as np
from collections import OrderedDict


def grid_shape_dictionary(m, n, n_wet, timesteps=None):
    """Create dictionary with c-grid types as keys, referencing the grid shape. If timesteps supplied,
    shape will have extra axis for time."""
    if timesteps is not None:
        grid_shapes = {'u': (n, m + 1, timesteps), 'v': (n, m, timesteps), 'w': (n + 1, m, timesteps),
                       'micro': (n_wet, m, timesteps)}
    else:
        grid_shapes = {'u': (n, m + 1), 'v': (n, m), 'w': (n + 1, m),
                       'micro': (n_wet, m)}
    return grid_shapes


def grid_size(grid_type, m, n, n_wet):
    """Return size of a c-grid type when given dimensions of v grid (m,n)"""
    grid_shapes = grid_shape_dictionary(m, n, n_wet)
    size_of_grid = np.product(grid_shapes[grid_type])
    return size_of_grid


def empty_array(grid_type, m, n, n_wet, timesteps):
    """Create empty numpy array for provided Arakawa c-grid type and number of timesteps.
    Accept arguments grid_type = 'u'/'v'/'w'/'micro', m = number of radial v gridpoints,
    n = number of vertical v gridpoints, n_wet = number of vertical micro gridpoints
    and timesteps = number of timesteps.
    Returns empty numpy array of shape (n, m+1, timesteps) for grid type 'u', shape (n, m, timesteps) for
    grid type 'v', (n+1, m, timesteps) for grid type 'w', and (n_wet, m, timesteps) for grid type 'micro'."""
    grid_shapes = grid_shape_dictionary(m, n, n_wet, timesteps)
    desired_shape = grid_shapes[grid_type]
    the_empty_array = np.empty(desired_shape)
    return the_empty_array


def read_data(datafile, grid_type, m, n, n_wet):
    """Reads data from f24 file for provided Arakawa c-grid type and shape of v grid.
    Accept arguments grid_type = 'u'/'v'/'w'/'micro', m = number of radial v gridpoints,
    n = number of vertical v gridpoints, n_wet = number of vertical micro gridpoints.
    Returns data in array of shape (n, m+1, timesteps) for grid type 'u', shape (n, m, timesteps) for
    grid type 'v', (n+1, m, timesteps) for grid type 'w', and (n_wet, m, timesteps) for grid type 'micro'."""
    data = np.fromfile(datafile, dtype=np.float32, count=grid_size(grid_type, m, n, n_wet))
    reshaped_data = np.reshape(data, grid_shape_dictionary(m, n, n_wet)[grid_type])
    return reshaped_data


def all_variable_grid_types():
    """Returns:
     An ordered dictionary with keys of variables written out by model at each timestep, giving their c-grid type"""
    variable_grid_dict = OrderedDict([
        ('u_before', 'u'),  # radial wind m/s
        ('v_before', 'v'),  # azimuthal wind m/s
        ('w_before', 'w'),  # vertical wind m/s
        ('pi_before', 'v'),  # exner perturbation
        ('theta_before', 'v'),  # potential temperature K
        ('rv_before', 'v'),  # water vapour mixing ratio kg/kg
        ('rl_before', 'v'),  # liquid water mixing ratio kg/kg
        ('rr_before', 'v'),  # ice mixing ratio kg/kg
        ('ri_before', 'v'),  # rain water mixing ratio kg/kg
        ('u', 'u'),  # radial wind m/s
        ('v', 'v'),  # azimuthal wind m/s
        ('w', 'w'),  # vertical wind m/s
        ('pi', 'v'),  # exner perturbation
        ('theta', 'v'),  # potential temperature K
        ('rv', 'v'),  # water vapour mixing ratio kg/kg
        ('rl', 'v'),  # liquid water mixing ratio kg/kg
        ('rr', 'v'),  # ice mixing ratio kg/kg
        ('ri', 'v'),  # rain water mixing ratio kg/kg
        ('subgrid_u', 'u'),  # diffusion U
        ('subgrid_v', 'v'),  # diffusion V
        ('subgrid_w', 'w'),  # diffusion W
        ('subgrid_theta', 'v'),  # diffusion T
        ('subgrid_rv', 'v'),  # diffusion QV
        ('subgrid_rl', 'v'),  # diffusion QL
        ('subgrid_rr', 'v'),  # diffusion QR
        ('subgrid_ri', 'v'),  # diffusion QI
        ('dthetav', 'v'),  # pressure correction
        ('vertical_flux_theta', 'w'),  # vertical theta flux
        ('vertical_flux_rv', 'w'),  # vertical rv flux
        ('radcool', 'v'),  # longwave radiative cooling
        ('micro_theta', 'micro'),  # microphysical scheme change in theta
        ('micro_rv', 'micro'),  # microphysical scheme change in rv
        ('precip_theta', 'v'),  # precip scheme change in theta
        ('precip_rv', 'v'),  # precip scheme change in rv
        ('precip_rl', 'v'),  # precip scheme change in rl
        ('precip_rr', 'v'),  # precip scheme change in rr
        ('precip_ri', 'v'),  # precip scheme change in ri
        ('u_after', 'u'),  # radial wind m/s
        ('v_after', 'v'),  # azimuthal wind m/s
        ('w_after', 'w'),  # vertical wind m/s
        ('pi_after', 'v'),  # exner perturbation
        ('theta_after', 'v'),  # potential temperature K
        ('rv_after', 'v'),  # water vapour mixing ratio kg/kg
        ('rl_after', 'v'),  # liquid water mixing ratio kg/kg
        ('rr_after', 'v'),  # ice mixing ratio kg/kg
        ('ri_after', 'v')  # rain water mixing ratio kg/kg
    ])
    return variable_grid_dict


def create_all_empty_arrays(m, n, n_wet, timesteps):
    """Creates a dictionary filled with keys of all read variables, referencing an empty array
    of the appropriate size."""
    model_data = {}
    all_variable_grid_dict = all_variable_grid_types()
    for variable, gridtype in all_variable_grid_dict.items():
        model_data[variable] = empty_array(gridtype, m, n, n_wet, timesteps)
    return model_data


def read_all_data(datafile, m, n, n_wet, timesteps):
    """Reads file data for all variables output by model at each timestep, and stores in data dictionary."""
    model_data = create_all_empty_arrays(m, n, n_wet, timesteps)
    read_variable_grid_dict = all_variable_grid_types()
    end_of_write_variables = ['ri_before', 'radcool', 'ri_after', 'precip_ri']  # read tmp after these
    _ = np.fromfile(datafile, dtype=np.float32, count=2)
    for i in range(timesteps):
        for variable, gridtype in read_variable_grid_dict.items():
            model_data[variable][:, :, i] = read_data(datafile, gridtype, m, n, n_wet)
            if variable in end_of_write_variables:
                _ = np.fromfile(datafile, dtype=np.float32, count=2)
    return model_data


def read_column_data(datafile, n, n_wet, grid_type):
    if grid_type == 'u' or grid_type == 'v':
        column_data = np.fromfile(datafile, dtype=np.float32, count=n)
    elif grid_type == 'w':
        column_data = np.fromfile(datafile, dtype=np.float32, count=n + 1)
    elif grid_type == 'micro':
        column_data = np.fromfile(datafile, dtype=np.float32, count=n_wet)
    else:
        raise KeyError('Unknown grid type for reading column')
    return column_data


def read_row_data(datafile, m, grid_type):
    if grid_type in ['v', 'w', 'micro']:
        column_data = np.fromfile(datafile, dtype=np.float32, count=m)
    elif grid_type == 'u':
        column_data = np.fromfile(datafile, dtype=np.float32, count=m + 1)
    else:
        raise KeyError('Unknown grid type for reading row')
    return column_data


def read_initial_data(datafile):
    """Create dictionary containing all model setup info and initial conditions."""
    initial_data = {}
    _ = np.fromfile(datafile, dtype=np.float32, count=1)

    params = np.fromfile(datafile, dtype=np.int32, count=8)
    initial_data['m'] = params[0]
    initial_data['n'] = params[1]
    initial_data['n_wet'] = params[2]
    initial_data['istart'] = params[3]
    initial_data['istop'] = params[4]
    initial_data['ibuff'] = params[5]
    initial_data['sponge_r'] = params[6]
    initial_data['sponge_z'] = params[7]
    initial_data['timesteps'] = (initial_data['istop'] - initial_data['istart'] + 1) // initial_data['ibuff']
    _ = np.fromfile(datafile, dtype=np.float32, count=2)
    constants = np.fromfile(datafile, dtype=np.float32, count=16)
    # DR,DZ,DT,F,G,RD,CP,C2,TBS,TBT,PDS,QVBT
    initial_data['dr'] = constants[0]
    initial_data['dz'] = constants[1]
    initial_data['dt'] = constants[2]
    initial_data['dts'] = constants[3]
    initial_data['f'] = constants[4]
    initial_data['g'] = constants[5]
    initial_data['Rd'] = constants[6]
    initial_data['cp'] = constants[7]
    initial_data['Lv'] = constants[8]
    initial_data['Lf'] = 3.34e5
    initial_data['Ls'] = initial_data['Lv'] + initial_data['Lf']
    initial_data['c2'] = constants[9]
    initial_data['theta_surface_initial'] = constants[10]
    initial_data['theta_top_initial'] = constants[11]
    initial_data['pi_surface_initial'] = constants[12]
    initial_data['pi_top_initial'] = constants[13]
    initial_data['rv_surface_initial'] = constants[14]
    initial_data['rv_top_initial'] = constants[15]

    initial_data['no_sponge_r'] = initial_data['m'] - initial_data['sponge_r']
    initial_data['no_sponge_z'] = initial_data['n'] - initial_data['sponge_z']

    _ = np.fromfile(datafile, dtype=np.float32, count=2)

    # RHOT,RHOW,R,RS,PN,TB,QVB
    initial_data['rho_initial_v'] = read_column_data(datafile, initial_data['n'], initial_data['n_wet'], 'v')
    initial_data['rho_initial_w'] = read_column_data(datafile, initial_data['n'], initial_data['n_wet'], 'w')
    initial_data['r_list_u'] = read_row_data(datafile, initial_data['m'], 'u')
    initial_data['r_list_v'] = read_row_data(datafile, initial_data['m'], 'v')
    initial_data['pi_initial'] = read_column_data(datafile, initial_data['n'], initial_data['n_wet'], 'v')
    initial_data['theta_initial'] = read_column_data(datafile, initial_data['n'], initial_data['n_wet'], 'v')
    initial_data['rv_initial'] = read_column_data(datafile, initial_data['n'], initial_data['n_wet'], 'v')
    initial_data['rho_thetav_initial_w'] = read_column_data(datafile, initial_data['n'], initial_data['n_wet'],
                                                            'w') * 100.
    return initial_data


def convert_micro_grids(model_data):
    # add zero rows to top of micro arrays to convert to v grid
    buffer_height = model_data['n'] - model_data['n_wet']
    zero_buffer = np.zeros((buffer_height, model_data['m'], model_data['timesteps']))
    for variable, gridtype in all_variable_grid_types().items():
        if gridtype == 'micro':  # convert micro arrays to v grid
            model_data[variable] = np.vstack((model_data[variable], zero_buffer))


def read_fortran_output(output_directory):
    """Creates dictionary filled with data generated by model, keyed by variable name."""
    output_file = output_directory + '/fort.24'
    model_data = {}
    with open(output_file, 'rb') as datafile:
        initial_data = read_initial_data(datafile)
        model_data.update(initial_data)
        timestep_data = read_all_data(datafile, model_data['m'], model_data['n'], model_data['n_wet'],
                                      model_data['timesteps'])
        model_data.update(timestep_data)
    convert_micro_grids(model_data)
    if np.max(model_data['rv'] > 0.05):
        print('Unrealistic values. Likely error reading fortran output file.')
    return model_data


def read_surface_data(data_directory, hurr):
    """Read surface potential temperature, mixing ratio and Exner pressure perturbation from fort.19"""
    m = hurr['m']
    timesteps = hurr['timesteps']
    surface_file = data_directory + '/fort.16'
    surface_data = {'theta': np.zeros((m, timesteps)),
                  'rv': np.zeros((m, timesteps)),
                  'pi': np.zeros((m, timesteps))}
    with open(surface_file, 'rb') as datafile:
        _ = np.fromfile(datafile, dtype=np.float32, count=1)
        for t in range(timesteps):
            surface_data['theta'][:, t] = np.fromfile(datafile, dtype=np.float32, count=m)
            surface_data['rv'][:, t] = np.fromfile(datafile, dtype=np.float32, count=m)
            surface_data['pi'][:, t] = np.fromfile(datafile, dtype=np.float32, count=m)
            _ = np.fromfile(datafile, dtype=np.float32, count=2)
    return surface_data


def read_initial_vortex(data_directory, hurr):
    """Read data for initialisation vortex from fort.49"""
    vortex_file = data_directory + '/fort.49'
    vortex_data = {'v': empty_array('v', hurr['m'], hurr['n'], hurr['n_wet'], 1),
                   'pi': empty_array('v', hurr['m'], hurr['n'], hurr['n_wet'], 1),
                   'theta': empty_array('v', hurr['m'], hurr['n'], hurr['n_wet'], 1),
                   'rv': empty_array('v', hurr['m'], hurr['n'], hurr['n_wet'], 1)}
    with open(vortex_file, 'rb') as datafile:
        _ = np.fromfile(datafile, dtype=np.float32, count=1)
        vortex_data['v'] = read_data(datafile, 'v', hurr['m'], hurr['n'], hurr['n_wet'])
        vortex_data['pi'] = read_data(datafile, 'v', hurr['m'], hurr['n'], hurr['n_wet'])
        vortex_data['theta'] = read_data(datafile, 'v', hurr['m'], hurr['n'], hurr['n_wet'])
        vortex_data['rv'] = read_data(datafile, 'v', hurr['m'], hurr['n'], hurr['n_wet'])
    return vortex_data


if __name__ == '__main__':
    hurr_data = read_fortran_output('../data')
