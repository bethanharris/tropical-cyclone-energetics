import numpy as np
import matplotlib.pyplot as plt
import thermo
from model_reader import read_fortran_output, read_surface_data
from utils import *


def intensity(directory, run_id, save=False, show=True):
    # plot time series of intensity in terms of both minimum surface pressure and maximum surface wind speed

    model_data = read_fortran_output(directory)
    surface_data = read_surface_data(directory, model_data)
    surface_exner = surface_data['pi'] + model_data['pi_surface_initial']
    surface_pressure_mb = thermo.pressure_from_PI(surface_exner)

    blue = '#093b9e'
    orange = '#f19928'
    minimum_surface_pressure = np.zeros(model_data['timesteps'], )
    maximum_wind_speed = np.zeros(model_data['timesteps'], )

    for t in np.arange(model_data['timesteps']):
        azimuthal_wind = model_data['v'][0:model_data['no_sponge_z'], 0:model_data['no_sponge_r'], t]
        surface_pressure = surface_pressure_mb[0:model_data['no_sponge_r'], t]
        minimum_surface_pressure[t] = np.min(surface_pressure)
        maximum_wind_speed[t] = np.max(np.max(azimuthal_wind))

    time = np.linspace(1, model_data['timesteps'],
                       model_data['timesteps']) * (model_data['dt'] * model_data['ibuff']) / 3600.
    if show or save:
        plt.figure(figsize=(7.5, 4.5))
        ax = plt.gca()
        ax.plot(time, maximum_wind_speed, color=blue, linewidth=3)
        ax2 = ax.twinx()
        ax2.plot(time, minimum_surface_pressure, color=orange, linewidth=3, linestyle='--')
        ax.set_ylim([0, 80])
        ax.set_xlim([0, time[-1]])
        ax2.set_yticks([900, 925, 950, 975, 1000])
        ax2.set_ylim([885, 1020])
        ax.set_xlabel('time (h)', fontsize=26, labelpad=5)
        ax2.set_ylabel(r'$\mathregular{p_{min}\; \left(hPa\right)}$', fontsize=26, color=orange)
        ax.set_ylabel(r'$\mathregular{v_{max}\; \left(ms^{-1}\right)}$', fontsize=26, color=blue, labelpad=10)
        ax2.tick_params(axis='y', colors=orange, labelsize=20, pad=4)
        ax.tick_params(axis='y', colors=blue, labelsize=20, pad=4)
        ax.tick_params(axis='x', colors='k', labelsize=20, pad=5)
        plt.tight_layout()
        if save:
            plt.savefig(f'../results/intensity_{run_id}.pdf')
        elif show:
            plt.show()

    return time, minimum_surface_pressure, maximum_wind_speed


if __name__ == '__main__':
    t, p, v = intensity('../data/J30pt3', 'J30pt3', save=True)
