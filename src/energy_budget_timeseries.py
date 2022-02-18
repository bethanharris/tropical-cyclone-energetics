import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from volume_integral import vol_int
from utils import clean_filename, safe_div, running_mean
from model_reader import read_fortran_output


def energy_budget_linestyles():
    colours = {'flux': '#377eb8',
               'zr_var': '#a65628',
               'src': '#858989',
               'subgrid': '#4daf4a',
               'coriolis': '#f4cb42',
               'pgrad': '#984ea3',
               'buoy': '#ff7f00',
               'theta_e_prod': '#e41a1c',
               'rt_prod': '#f781bf',
               'total_prod': '#f781bf',
               'dthetav': '#b3de69'}
    shapes = {'flux': 'o',
              'zr_var': 'x',
              'src': '2',
              'subgrid': '^',
              'coriolis': 'p',
              'pgrad': 'D',
              'buoy': '>',
              'theta_e_prod': '*',
              'rt_prod': 'v',
              'total_prod': 'v',
              'dthetav': 's'}
    return colours, shapes


def time_jump_masker(ape_budget):
    time_discontinuous = np.abs(ape_budget['zr_change_time']) > 0.
    masked_budget = {}
    for key in ape_budget.keys():
        component = ape_budget[key]
        component[time_discontinuous] = 0.
        masked_budget[key] = component
    return masked_budget


def plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key='non-sponge', title=False, end_time=None, save=False, normalise=False,
                     smooth=False, legend_on_error_plot=False):
    time_list = np.arange(1, hurr['timesteps'] + 1) * hurr['ibuff'] * hurr['dt'] / 3600.
    if end_time and end_time > time_list.size:
        end_time = None
    time_list = time_list[0:end_time]
    if isinstance(region_key, str):
        region_key = [region_key]

    terms = budget_terms.keys()

    for region in region_key:
        integrated_budget_terms = {term: vol_int(hurr, budget_terms[term], region_key=region) for term in terms}

        if normalise:
            rho_grid = np.dstack([np.tile(hurr['rho_initial_v'], (hurr['m'], 1)).transpose()] * hurr['timesteps'])
            region_mass = vol_int(hurr, rho_grid, region_key=region)
            integrated_budget_terms = {term: safe_div(integrated_budget_terms[term], region_mass) for term in terms}

        if smooth:
            window = 3
            time_list = running_mean(time_list, window)
            integrated_budget_terms = {term: running_mean(integrated_budget_terms[term], window) for term in terms}

        fig = plt.figure(figsize=(8, 5))
        if title:
            plt.title(f'{title}, {run_id} {region} region', fontsize=16)
        for term in terms:
            if term == 'estimated_deriv':
                pass
            elif term == 'time_deriv':
                plt.plot(time_list, integrated_budget_terms[term][0:end_time], 'k--',
                        linewidth=2, label=budget_labels[term])
            else:
                plt.plot(time_list, integrated_budget_terms[term][0:end_time], color=budget_colors[term],
                         linewidth=2, marker=budget_markers[term], label=budget_labels[term])

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
            y_label = r'$\mathregular{\left(Wkg^{-1}\right)}$'
        else:
            y_label = r'$\mathregular{\left(W\right)}$'
        ax.set_ylabel(y_label, fontsize=20)
        if save:
            save_name = f'{clean_filename(region)}_{energy_type}_{run_id}'
            if normalise:
                save_name += '_normalised'
            if smooth:
                save_name += '_smoothed'
            fig.savefig(f'../results/energy_budget_timeseries/{save_name}.pdf',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


        estimated_derivative = integrated_budget_terms['estimated_deriv']
        fig = plt.figure(figsize=(6, 4.5))
        if title:
            plt.title(title, fontsize=16)
        plt.plot(time_list, integrated_budget_terms['time_deriv'][0:end_time], 'k',
                 linewidth=2, label=budget_labels['time_deriv'])
        plt.plot(time_list, estimated_derivative[0:end_time], '--', color='orange', dashes=(5, 5),
                 linewidth=2, label=r'$\mathrm{sum}$')
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
        ax.set_ylabel(y_label, fontsize=20)
        if save:
            extra = (lgd,) if legend_on_error_plot else None
            fig.savefig(f'../results/energy_budget_timeseries/error/{save_name}.pdf',
                        bbox_inches='tight', bbox_extra_artists=extra)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


def primary_ke(data_dir, hurr, run_id, region_key='non-sponge', title=False, end_time=None, save=False,
               normalise=False, smooth=False, legend_on_error_plot=False):
    energy_type = 'primary_ke'
    if title:
        title = 'Primary KE budget'

    budget = np.load(f'{data_dir}/kinetic_elastic_energy_budgets_{run_id}.npz')
    colours, shapes = energy_budget_linestyles()

    budget_terms = OrderedDict({'flux': -budget['ek_flux_v'],
                                'src': budget['mass_source_v'],
                                'subgrid': budget['subgrid_v'],
                                'coriolis': -budget['coriolis_v']
                                })

    budget_colors = {term: colours[term] for term in budget_terms.keys()}
    budget_markers = {term: shapes[term] for term in budget_terms.keys()}

    budget_labels = {'flux': r'$-\nabla\cdot\left(\overline{\rho}e_k^v\mathbf{v}\right)$',
                     'src': r'$e_k^v\nabla\cdot\left(\overline{\rho}\mathbf{v}\right)$',
                     'subgrid': r'$\overline{\rho}vD_v$',
                     'coriolis': r'$-\overline{\rho}\left(fv+\frac{v^2}{r}\right)u$',
                     'time_deriv': r'$\frac{\partial\left(\overline{\rho}e_k^v\right)}{\partial t}$'
                     }

    budget_terms['estimated_deriv'] = sum([term for term in budget_terms.values()])
    budget_terms['time_deriv'] = budget['time_change_ek_v']

    budget.close()

    plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key=region_key, title=title, end_time=end_time, save=save, normalise=normalise,
                     smooth=smooth, legend_on_error_plot=legend_on_error_plot)


def radial_ke(data_dir, hurr, run_id, region_key='non-sponge', title=False, end_time=None, save=False,
              normalise=False, smooth=False, legend_on_error_plot=False):
    energy_type = 'radial_ke'
    if title:
        title = 'Radial KE budget'

    budget = np.load(f'{data_dir}/kinetic_elastic_energy_budgets_{run_id}.npz')
    colours, shapes = energy_budget_linestyles()

    budget_terms = OrderedDict({'flux': -budget['ek_flux_u'],
                                'src': budget['mass_source_u'],
                                'subgrid': budget['subgrid_u'],
                                'coriolis': budget['coriolis_u'],
                                'pgrad': -budget['pressure_grad_u']
                                })

    budget_colors = {term: colours[term] for term in budget_terms.keys()}
    budget_markers = {term: shapes[term] for term in budget_terms.keys()}

    budget_labels = {'flux': r'$-\nabla\cdot\left(\overline{\rho}e_k^u\mathbf{v}\right)$',
                     'src': r'$e_k^u\nabla\cdot\left(\overline{\rho}\mathbf{v}\right)$',
                     'subgrid': r'$\overline{\rho}uD_u$',
                     'coriolis': r'$\overline{\rho}\left(fv+\frac{v^2}{r}\right)u$',
                     'pgrad': r'$-c_p\overline{\rho}\overline{\theta_v}u\frac{\partial\pi}{\partial r}$',
                     'time_deriv': r'$\frac{\partial\left(\overline{\rho}e_k^u\right)}{\partial t}$'
                     }

    budget_terms['estimated_deriv'] = sum([term for term in budget_terms.values()])
    budget_terms['time_deriv'] = budget['time_change_ek_u']

    budget.close()

    plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key=region_key, title=title, end_time=end_time, save=save, normalise=normalise,
                     smooth=smooth, legend_on_error_plot=legend_on_error_plot)


def horizontal_ke(data_dir, hurr, run_id, region_key='non-sponge', title=False, end_time=None, save=False,
                  normalise=False, smooth=False, legend_on_error_plot=False):

    energy_type = 'horizontal_ke'
    if title:
        title = 'Horizontal KE budget'

    budget = np.load(f'{data_dir}/kinetic_elastic_energy_budgets_{run_id}.npz')
    colours, shapes = energy_budget_linestyles()

    budget_terms = OrderedDict({'flux': -(budget['ek_flux_u'] + budget['ek_flux_v']),
                                'src': budget['mass_source_u'] + budget['mass_source_v'],
                                'subgrid': budget['subgrid_u'] + budget['subgrid_v'],
                                'pgrad': -budget['pressure_grad_u']
                                })

    budget_colors = {term: colours[term] for term in budget_terms.keys()}
    budget_markers = {term: shapes[term] for term in budget_terms.keys()}

    budget_labels = {'flux': r'$-\nabla\cdot\left(\overline{\rho}e_k^h\mathbf{v}\right)$',
                     'src': r'$e_k^h\nabla\cdot\left(\overline{\rho}\mathbf{v}\right)$',
                     'subgrid': r'$\overline{\rho}\left(uD_u + vD_v\right)$',
                     'pgrad': r'$-c_p\overline{\rho}\overline{\theta_v}u\frac{\partial\pi}{\partial r}$',
                     'time_deriv': r'$\frac{\partial\left(\overline{\rho}e_k^h\right)}{\partial t}$'
                     }

    budget_terms['estimated_deriv'] = sum([term for term in budget_terms.values()])
    budget_terms['time_deriv'] = budget['time_change_ek_u'] + budget['time_change_ek_v']

    budget.close()

    plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key=region_key, title=title, end_time=end_time, save=save, normalise=normalise,
                     smooth=smooth, legend_on_error_plot=legend_on_error_plot)


def vertical_ke(data_dir, hurr, run_id, ref_state, region_key='non-sponge', title=False, end_time=None, save=False,
                smooth=False, normalise=False, legend_on_error_plot=False):

    energy_type = f'vertical_ke_{ref_state}'
    if title:
        title = 'Vertical KE budget'

    budget = np.load(f'{data_dir}/kinetic_elastic_energy_budgets_{run_id}.npz')
    colours, shapes = energy_budget_linestyles()

    if ref_state == 'initial':
        pressure_grad_w = budget['pressure_grad_w']
        buoyancy_flux = budget['buoyancy']
    else:
        raise KeyError('Invalid reference state type. Only initial reference state currently supported.')

    budget_terms = OrderedDict({'flux': -budget['ek_flux_w'],
                                'src': budget['mass_source_w'],
                                'subgrid': budget['subgrid_w'],
                                'buoy': buoyancy_flux,
                                'pgrad': -pressure_grad_w
                                })

    budget_colors = {term: colours[term] for term in budget_terms.keys()}
    budget_markers = {term: shapes[term] for term in budget_terms.keys()}

    budget_labels = {'flux': r'$-\nabla\cdot\left(\overline{\rho}e_k^w\mathbf{v}\right)$',
                     'src': r'$e_k^w\nabla\cdot\left(\overline{\rho}\mathbf{v}\right)$',
                     'subgrid': r'$\overline{\rho}wD_w$',
                     'buoy': r'$\overline{\rho}bw$',
                     'pgrad': r'$-c_p\overline{\rho}\overline{\theta_v}w\frac{\partial\pi}{\partial z}$',
                     'time_deriv': r'$\frac{\partial\left(\overline{\rho}e_k^w\right)}{\partial t}$'
                     }

    budget_terms['estimated_deriv'] = sum([term for term in budget_terms.values()])
    budget_terms['time_deriv'] = budget['time_change_ek_w']

    budget.close()

    plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key=region_key, title=title, end_time=end_time, save=save, normalise=normalise,
                     smooth=smooth, legend_on_error_plot=legend_on_error_plot)


def secondary_ke(data_dir, hurr, run_id, ref_state, region_key='non-sponge', title=False, end_time=None, save=False,
                 smooth=False, normalise=False, legend_on_error_plot=False):

    energy_type = f'secondary_ke_{ref_state}'
    if title:
        title = 'Secondary KE budget'

    budget = np.load(f'{data_dir}/kinetic_elastic_energy_budgets_{run_id}.npz')
    colours, shapes = energy_budget_linestyles()

    if ref_state == 'initial':
        pressure_grad_w = budget['pressure_grad_w']
        buoyancy_flux = budget['buoyancy']
    else:
        raise KeyError('Invalid reference state type. Only initial reference state currently supported.')

    budget_terms = OrderedDict({'flux': -(budget['ek_flux_u'] + budget['ek_flux_w']),
                                'src': budget['mass_source_u'] + budget['mass_source_w'],
                                'subgrid': budget['subgrid_u'] + budget['subgrid_w'],
                                'buoy': buoyancy_flux,
                                'pgrad': -(budget['pressure_grad_u'] + pressure_grad_w),
                                'coriolis': budget['coriolis_u']
                                })

    budget_colors = {term: colours[term] for term in budget_terms.keys()}
    budget_markers = {term: shapes[term] for term in budget_terms.keys()}

    budget_labels = {'flux': r'$-\nabla\cdot\left [ \overline{\rho}\left (e_k^u + e_k^w  \right )\mathbf{v}\right ]$',
                     'src': r'$\left(e_k^u+e_k^w\right)\nabla\cdot\left(\overline{\rho}\mathbf{v}\right)$',
                     'subgrid': r'$\overline{\rho}\left(uD_u+wD_w\right)$',
                     'buoy': r'$\overline{\rho}bw$',
                     'pgrad': r'$-c_p\overline{\rho}\overline{\theta_v}\mathbf{v}\cdot\nabla\pi$',
                     'coriolis':r'$\overline{\rho}\left(fv+\frac{v^2}{r}\right)u$',
                     'time_deriv': r'$\frac{\partial}{\partial t}\left [ \overline{\rho} \left ( e_k^u + e_k^w \right ) \right ]$'
                     }

    budget_terms['estimated_deriv'] = sum([term for term in budget_terms.values()])
    budget_terms['time_deriv'] = budget['time_change_ek_u'] + budget['time_change_ek_w']

    budget.close()

    plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key=region_key, title=title, end_time=end_time, save=save, normalise=normalise,
                     smooth=smooth, legend_on_error_plot=legend_on_error_plot)


def total_ke(data_dir, hurr, run_id, ref_state, region_key='non-sponge', title=False, end_time=None, save=False,
             smooth=False, normalise=False, legend_on_error_plot=False):

    energy_type = f'total_ke_{ref_state}'
    if title:
        title = 'Total KE budget'

    budget = np.load(f'{data_dir}/kinetic_elastic_energy_budgets_{run_id}.npz')
    colours, shapes = energy_budget_linestyles()

    if ref_state == 'initial':
        pressure_grad_w = budget['pressure_grad_w']
        buoyancy_flux = budget['buoyancy']
    else:
        raise KeyError('Invalid reference state type. Only initial reference state currently supported.')

    budget_terms = OrderedDict({'flux': -(budget['ek_flux_u'] + budget['ek_flux_v'] + budget['ek_flux_w']),
                                'src': budget['mass_source_u'] + budget['mass_source_v'] + budget['mass_source_w'],
                                'subgrid': budget['subgrid_u'] + budget['subgrid_v'] + budget['subgrid_w'],
                                'buoy': buoyancy_flux,
                                'pgrad': -(budget['pressure_grad_u'] + pressure_grad_w)
                                })

    budget_colors = {term: colours[term] for term in budget_terms.keys()}
    budget_markers = {term: shapes[term] for term in budget_terms.keys()}

    budget_labels = {
        'flux': r'$-\nabla\cdot\left ( \overline{\rho}e_k\mathbf{v}\right )$',
        'src': r'$e_k\nabla\cdot\left(\overline{\rho}\mathbf{v}\right)$',
        'subgrid': r'$\overline{\rho}\mathbf{v}\cdot\mathbf{D}$',
        'buoy': r'$\overline{\rho}bw$',
        'pgrad': r'$-c_p\overline{\rho}\overline{\theta_v}\mathbf{v}\cdot\nabla\pi$',
        'time_deriv': r'$\frac{\partial}{\partial t}\left ( \overline{\rho} e_k  \right )$'
        }

    budget_terms['estimated_deriv'] = sum([term for term in budget_terms.values()])
    budget_terms['time_deriv'] = budget['time_change_ek_u'] + budget['time_change_ek_v'] + budget['time_change_ek_w']

    budget.close()

    plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key=region_key, title=title, end_time=end_time, save=save, normalise=normalise,
                     smooth=smooth, legend_on_error_plot=legend_on_error_plot)


def aee(data_dir, hurr, run_id, ref_state, region_key='non-sponge', title=False, end_time=None, save=False,
        smooth=False, normalise=False, legend_on_error_plot=False):

    energy_type = f'aee_{ref_state}'
    if title:
        title = 'AEE budget'

    colours, shapes = energy_budget_linestyles()

    if ref_state == 'initial':
        budget = np.load(f'{data_dir}/kinetic_elastic_energy_budgets_{run_id}.npz')
        time_change_ee = budget['time_change_ee']
        pressure_div = budget['pressure_div']
        pressure_grad_u = budget['pressure_grad_u']
        pressure_grad_w = budget['pressure_grad_w']
        pressure_dthetav = budget['pressure_dthetav']
    else:
        raise KeyError('Invalid reference state type. Only initial reference state currently supported.')

    budget_terms = OrderedDict({'flux': -pressure_div,
                                'pgrad': pressure_grad_u + pressure_grad_w,
                                'dthetav': pressure_dthetav
                                })

    budget_colors = {term: colours[term] for term in budget_terms.keys()}
    budget_markers = {term: shapes[term] for term in budget_terms.keys()}

    budget_labels = {
        'flux': r'$-\nabla\cdot\left(\overline{\rho}c_p\overline{\theta_v}\pi\mathbf{v}\right)$',
        'pgrad': r'$c_p\overline{\rho}\overline{\theta_v}\mathbf{v}\cdot\nabla\pi$',
        'dthetav': r'$\overline{\rho}c_p\pi\frac{D\theta_v}{D t}$',
        'time_deriv': r'$\frac{\partial\left(\overline{\rho}e_e\right)}{\partial t}$'
    }

    budget_terms['estimated_deriv'] = sum([term for term in budget_terms.values()])
    budget_terms['time_deriv'] = time_change_ee

    budget.close()

    plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key=region_key, title=title, end_time=end_time, save=save, normalise=normalise,
                     smooth=smooth, legend_on_error_plot=legend_on_error_plot)


def aee_plus_ke(data_dir, hurr, run_id, ref_state, region_key='non-sponge', title=False, end_time=None, save=False,
                normalise=False, smooth=False, legend_on_error_plot=False):

    energy_type = f'aee_plus_ke_{ref_state}'
    if title:
        title = 'Total KE + AEE budget'

    budget = np.load(f'{data_dir}/kinetic_elastic_energy_budgets_{run_id}.npz')
    colours, shapes = energy_budget_linestyles()

    if ref_state == 'initial':
        time_change_ee = budget['time_change_ee']
        pressure_div = budget['pressure_div']
        pressure_dthetav = budget['pressure_dthetav']
        buoyancy_flux = budget['buoyancy']
    else:
        raise KeyError('Invalid reference state type. Only initial reference state currently supported.')

    budget_terms = OrderedDict({'flux': -(budget['ek_flux_u'] + budget['ek_flux_v'] + budget['ek_flux_w']
                                          + pressure_div),
                                'dthetav': pressure_dthetav,
                                'src': budget['mass_source_u'] + budget['mass_source_v'] + budget['mass_source_w'],
                                'subgrid': budget['subgrid_u'] + budget['subgrid_v'] + budget['subgrid_w'],
                                'buoy': buoyancy_flux
                                })

    budget_colors = {term: colours[term] for term in budget_terms.keys()}
    budget_markers = {term: shapes[term] for term in budget_terms.keys()}

    budget_labels = {
        'flux': r'$-\nabla\cdot\left [ \overline{\rho}\left (e_k + c_p\overline{\theta_v}\pi\right )\mathbf{v}\right ]$',
        'dthetav': r'$\overline{\rho}c_p\pi\frac{D\theta_v}{D t}$',
        'src': r'$e_k\nabla\cdot\left(\overline{\rho}\mathbf{v}\right)$',
        'subgrid': r'$\overline{\rho}\mathbf{v}\cdot\mathbf{D}$',
        'buoy': r'$\overline{\rho}bw$',
        'time_deriv': r'$\frac{\partial}{\partial t}\left [ \overline{\rho} \left ( e_k + e_e\right ) \right ]$'
    }

    budget_terms['estimated_deriv'] = sum([term for term in budget_terms.values()])
    budget_terms['time_deriv'] = (budget['time_change_ek_u'] + budget['time_change_ek_v']
                                  + budget['time_change_ek_w'] + time_change_ee)

    budget.close()

    plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key=region_key, title=title, end_time=end_time, save=save, normalise=normalise,
                     smooth=smooth, legend_on_error_plot=legend_on_error_plot)


def ape(directory, hurr, run_id, ref_state, region_key='non-sponge', end_time=None, save=False,
        title=False, mask_time_jumps=False, normalise=False, show_total_prod=False, smooth=False,
        legend_on_error_plot=False):

    energy_type = f'ape_{ref_state}_ref'
    if show_total_prod:
        energy_type += '_total_prod'
    if mask_time_jumps:
        energy_type += '_mask_time_jumps'
    if title:
        title = 'APE budget'

    budget = np.load(f'{directory}/ape_{ref_state}_ref_budget_{run_id}.npz')
    if mask_time_jumps:
        budget = time_jump_masker(budget)
    colours, shapes = energy_budget_linestyles()

    zr_var = budget['zr_change_space'] if mask_time_jumps else (budget['zr_change_time'] + budget['zr_change_space'])

    if show_total_prod:
        budget_terms = OrderedDict({
            'total_prod': budget['theta_e_prod'] + budget['rt_prod'],
            'buoy': -budget['buoyancy_flux'],
            'flux': -budget['ape_flux'],
            'src': budget['ape_src'],
            'zr_var': zr_var
        })
    else:
        budget_terms = OrderedDict({
            'theta_e_prod': budget['theta_e_prod'],
            'rt_prod': budget['rt_prod'],
            'buoy': -budget['buoyancy_flux'],
            'flux': -budget['ape_flux'],
            'src': budget['ape_src'],
            'zr_var': zr_var
        })

    budget_labels = {
        'total_prod': r'$\overline{\rho}\left(\varepsilon_{\theta_{ei}}c_p\Pi\frac{D\theta_{ei}}{Dt} + \varepsilon_{r_t}L_s\frac{Dr_t}{Dt}\right)$',
        'theta_e_prod': r'$\overline{\rho}\varepsilon_{\theta_{ei}}c_p\Pi\frac{D\theta_{ei}}{Dt}$',
        'rt_prod': r'$\overline{\rho}\varepsilon_{r_t}L_s\frac{Dr_t}{Dt}$',
        'buoy': r'$-\overline{\rho}bw$',
        'flux': r'$-\nabla\cdot\left(\overline{\rho}e_a\mathbf{v}\right)$',
        'src': r'$e_a\nabla\cdot\left(\overline{\rho}\mathbf{v}\right)$',
        'zr_var': r'$\mathregular{discontinuity}$',
        'time_deriv': r'$\frac{\partial\left(\overline{\rho}e_a\right)}{\partial t}$'
    }

    budget_colors = {term: colours[term] for term in budget_terms.keys()}
    budget_markers = {term: shapes[term] for term in budget_terms.keys()}

    budget_terms['estimated_deriv'] = sum([term for term in budget_terms.values()])
    budget_terms['time_deriv'] = budget['ape_deriv']

    plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key=region_key, title=title, end_time=end_time, save=save, normalise=normalise,
                     smooth=smooth, legend_on_error_plot=legend_on_error_plot)


def total_energy(directory, hurr, run_id, ref_state, region_key='non-sponge', end_time=None, save=False,
                 title=False, mask_time_jumps=False, normalise=False, show_total_prod=False, smooth=False,
                 legend_on_error_plot=False):

    energy_type = f'total_energy_{ref_state}_ref'
    if show_total_prod:
        energy_type += '_total_prod'
    if mask_time_jumps:
        energy_type += '_mask_time_jumps'
    if title:
        title = 'Total KE + APE + AEE budget'

    ke_budget = np.load(f'{data_dir}/kinetic_elastic_energy_budgets_{run_id}.npz')
    ape_budget = np.load(f'{directory}/ape_{ref_state}_ref_budget_{run_id}.npz')
    if mask_time_jumps:
        ape_budget = time_jump_masker(ape_budget)
    colours, shapes = energy_budget_linestyles()

    zr_var = ape_budget['zr_change_space'] if mask_time_jumps else (ape_budget['zr_change_time']
                                                                    + ape_budget['zr_change_space'])

    if ref_state == 'initial':
        time_change_ee = ke_budget['time_change_ee']
        pressure_div = ke_budget['pressure_div']
        pressure_dthetav = ke_budget['pressure_dthetav']
    else:
        raise KeyError('Invalid reference state type. Only initial reference state currently supported.')

    if show_total_prod:
        budget_terms = OrderedDict({
            'total_prod': ape_budget['theta_e_prod'] + ape_budget['rt_prod'],
            'zr_var': zr_var,
            'flux': -(ke_budget['ek_flux_u'] + ke_budget['ek_flux_v'] + ke_budget['ek_flux_w']
                      + pressure_div + ape_budget['ape_flux']),
            'src': (ke_budget['mass_source_u'] + ke_budget['mass_source_v'] + ke_budget['mass_source_w']
                    + ape_budget['ape_src']),
            'subgrid': ke_budget['subgrid_u'] + ke_budget['subgrid_v'] + ke_budget['subgrid_w'],
            'dthetav': pressure_dthetav
        })
    else:
        budget_terms = OrderedDict({
            'theta_e_prod': ape_budget['theta_e_prod'],
            'rt_prod': ape_budget['rt_prod'],
            'zr_var': zr_var,
            'flux': -(ke_budget['ek_flux_u'] + ke_budget['ek_flux_v'] + ke_budget['ek_flux_w']
                      + pressure_div + ape_budget['ape_flux']),
            'src': (ke_budget['mass_source_u'] + ke_budget['mass_source_v'] + ke_budget['mass_source_w']
                    + ape_budget['ape_src']),
            'subgrid': ke_budget['subgrid_u'] + ke_budget['subgrid_v'] + ke_budget['subgrid_w'],
            'dthetav': pressure_dthetav

        })

    budget_labels = {
        'total_prod': r'$\overline{\rho}\left(\varepsilon_{\theta_{ei}}c_p\Pi\frac{D\theta_{ei}}{Dt} + \varepsilon_{r_t}L_s\frac{Dr_t}{Dt}\right)$',
        'theta_e_prod': r'$\overline{\rho}\varepsilon_{\theta_{ei}}c_p\Pi\frac{D\theta_{ei}}{Dt}$',
        'rt_prod': r'$\overline{\rho}\varepsilon_{r_t}L_s\frac{Dr_t}{Dt}$',
        'flux': r'$-\nabla\cdot\left [ \overline{\rho}\left (e_k + e_a + c_p\overline{\theta_v}\pi\right )\mathbf{v}\right ]$',
        'src': r'$\left(e_k+e_a\right)\nabla\cdot\left(\overline{\rho}\mathbf{v}\right)$',
        'subgrid': r'$\overline{\rho}\mathbf{v}\cdot\mathbf{D}$',
        'dthetav': r'$\overline{\rho}c_p\pi\frac{\partial\theta}{\partial t}$',
        'zr_var': r'$\mathregular{discontinuity}$',
        'time_deriv': r'$\frac{\partial}{\partial t}\left [ \overline{\rho} \left ( e_k + e_a + e_e\right ) \right ]$'
    }

    budget_colors = {term: colours[term] for term in budget_terms.keys()}
    budget_markers = {term: shapes[term] for term in budget_terms.keys()}

    budget_terms['estimated_deriv'] = sum([term for term in budget_terms.values()])
    budget_terms['time_deriv'] = (ke_budget['time_change_ek_u'] + ke_budget['time_change_ek_v'] +
                                  ke_budget['time_change_ek_w'] + time_change_ee + ape_budget['ape_deriv'])

    ke_budget.close()

    plot_time_series(hurr, run_id, energy_type, budget_terms, budget_labels, budget_colors, budget_markers,
                     region_key=region_key, title=title, end_time=end_time, save=save, normalise=normalise,
                     smooth=smooth, legend_on_error_plot=legend_on_error_plot)


if __name__ == '__main__':
    data_dir = '../data/J30pt3'
    hurr = read_fortran_output(data_dir)
    run_id = 'J30pt3'
    region_key = 'non-sponge'
    primary_ke(data_dir, hurr, run_id, region_key=region_key, save=True)
    radial_ke(data_dir, hurr, run_id, region_key=region_key, save=True)
    horizontal_ke(data_dir, hurr, run_id, region_key=region_key, save=True)
    vertical_ke(data_dir, hurr, run_id, 'initial', region_key=region_key, save=True)
    secondary_ke(data_dir, hurr, run_id, 'initial', region_key=region_key, save=True)
    aee(data_dir, hurr, run_id, 'initial', region_key=region_key, save=True)
    aee_plus_ke(data_dir, hurr, run_id, 'initial', region_key=region_key, save=True)
    total_ke(data_dir, hurr, run_id, 'initial', region_key=region_key, save=True)
    ape(data_dir, hurr, run_id, 'initial', region_key=region_key, save=True, mask_time_jumps=True, show_total_prod=True)
    ape(data_dir, hurr, run_id, 'initial', region_key='r<300', save=True, mask_time_jumps=True, show_total_prod=True)
    total_energy(data_dir, hurr, run_id, 'initial', region_key=region_key, save=True, mask_time_jumps=True, show_total_prod=True)
    total_energy(data_dir, hurr, run_id, 'initial', region_key='r<300', save=True, mask_time_jumps=True, show_total_prod=True)
