import numpy as np
import matplotlib.pyplot as plt
from continuous_ape import buoyancy_interp_parcel, initial_state_interp, ape_density_parcel
from model_reader import read_fortran_output


def plot_initial_state(hurr):
    specvol_ref, pmb_ref = initial_state_interp(hurr)
    all_z = np.arange(hurr['n']*hurr['dz'])

    blue = '#093b9e'
    orange = '#f19928'

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 8
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size

    plt.figure()
    ax = plt.gca()
    ax2 = ax.twiny()
    ax.plot(pmb_ref(all_z), all_z/1000., linewidth=2, color=blue)
    ax2.plot(specvol_ref(all_z), all_z/1000., linewidth=2, color=orange)
    ax.set_ylabel(r'$\mathregular{z\, (km)}$', fontsize=24, color='k')
    ax.set_xlabel(r'$\mathregular{\overline{p}\, (mb)}$', fontsize=24, color=blue)
    ax2.set_xlabel(r'$\mathregular{\overline{\alpha}\, \left(m^3kg^{-1}\right)}$', fontsize=24, color=orange, labelpad=7)
    ax2.tick_params(axis='x', colors=orange, labelsize=20, pad=2)
    ax.tick_params(axis='x', colors=blue, labelsize=20, pad=4)
    ax.tick_params(axis='y', colors='k', labelsize=20, pad=10)
    plt.tight_layout()
    plt.show()


def plot_thermo_lifted(hurr):
    ref_interps = initial_state_interp(hurr)
    buoyancy_interp, t_K_interp, pd_mb_interp, rel_hum_interp, pressure_mb_initial_interp, rv_interp, rl_interp, spec_vol_fine = buoyancy_interp_parcel(hurr, 342.5, 0.015, ref_interps)

    all_z = np.arange(14000.)

    blue = '#093b9e'
    orange = '#f19928'

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 8
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size

    plt.figure()
    ax = plt.gca()
    ax2 = ax.twiny()
    ax.plot(t_K_interp(all_z), all_z/1000., linewidth=2, color=blue)
    ax2.plot(rl_interp(all_z), all_z/1000., linewidth=2, color=orange)
    ax2.set_xticks([0, 0.005, 0.01, 0.015])
    ax.set_ylabel(r'$\mathregular{z\, (km)}$', fontsize=24, color='k')
    ax.set_xlabel(r'$\mathregular{T\, (K)}$', fontsize=24, color=blue)
    ax2.set_xlabel(r'$\mathregular{r_l\, \left(kgkg^{-1}\right)}$', fontsize=24, color=orange, labelpad=7)
    ax2.tick_params(axis='x', colors=orange, labelsize=20, pad=2)
    ax.tick_params(axis='x', colors=blue, labelsize=20, pad=4)
    ax.tick_params(axis='y', colors='k', labelsize=20, pad=10)
    plt.suptitle(r'$\mathregular{\theta_e=342.5\,K,\; r_t=0.015\,kgkg^{-1}}$', fontsize=18, y=0.83, x=0.55)
    plt.tight_layout()
    plt.show()


def plot_buoyancy(hurr):
    ref_interps = initial_state_interp(hurr)
    buoyancy_interp, t_K_interp, pd_mb_interp, rel_hum_interp, pressure_mb_initial_interp, rv_interp, rl_interp, spec_vol_fine = buoyancy_interp_parcel(hurr, 342.5, 0.015, ref_interps)
    buoyancy_perturb, _, _, _, _, _, _, _ = buoyancy_interp_parcel(hurr, 343.0, 0.015, ref_interps)

    lnbs = buoyancy_interp.roots()
    perturb_lnbs = buoyancy_perturb.roots()

    all_z = np.arange(100., 14000.)

    blue = '#093b9e'
    orange = '#f19928'
    green = '#0df704'

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 8
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size

    plt.figure()
    ax = plt.gca()
    ax.axvline(0, color=orange, linestyle='--', linewidth=2, zorder=2)
    ax.plot(buoyancy_interp(all_z), all_z/1000., linewidth=2, color=blue, zorder=1)
    ax.plot(buoyancy_perturb(all_z), all_z/1000., '-.', color='gray', linewidth=2, zorder=1)
    ax.text(0.095, 6, r'$\mathregular{\Delta\theta_e = 0.5K}$', color='gray', fontsize=18)
    ax.scatter([buoyancy_interp(lnbs[1]), buoyancy_perturb(perturb_lnbs[0])], [lnbs[1]/1000., perturb_lnbs[0]/1000.], color=green, marker='*', s=145, edgecolors='k', zorder=3)
    print(lnbs[1]/1000., perturb_lnbs[0]/1000.)
    ax.set_xticks([-0.05, 0, 0.05, 0.1])
    ax.set_ylabel(r'$\mathregular{z\, (km)}$', fontsize=24)
    ax.set_xlabel(r'$\mathregular{b\, (ms^{-2})}$', fontsize=24)
    ax.tick_params(colors='k', labelsize=20, pad=4)
    plt.title(r'$\mathregular{\theta_e=342.5\,K,\; r_t=0.015\,kgkg^{-1}}$', fontsize=18)
    plt.tight_layout()
    plt.show()


def subplots_buoyancy(hurr, theta_e, rt):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.4), facecolor='w', edgecolor='k')
    ax = axs[0]
    ref_interps = initial_state_interp(hurr)
    buoyancy_interp, t_K_interp, pd_mb_interp, rel_hum_interp, pressure_mb_initial_interp, rv_interp, rl_interp, spec_vol_fine = buoyancy_interp_parcel(hurr, theta_e, rt, ref_interps)
    buoyancy_perturb, _, _, _, _, _, _, _ = buoyancy_interp_parcel(hurr, theta_e+1., rt, ref_interps)
    lnbs = buoyancy_interp.roots()
    perturb_lnbs = buoyancy_perturb.roots()
    lnbs = lnbs[lnbs>200.]
    perturb_lnbs = perturb_lnbs[perturb_lnbs>200.]

    blue = '#093b9e'
    orange = '#f19928'
    green = '#0df704'

    all_z = np.arange(200., 13500.)
    ax2 = ax.twiny()
    ax.plot(t_K_interp(all_z), all_z/1000., '--', linewidth=2, color=blue, dashes=(5,1))
    ax2.plot(rl_interp(all_z), all_z/1000., linewidth=2, color=orange)
    ax.set_xticks([220, 240, 260, 280, 300])
    ax2.set_xticks([0, 0.005, 0.01, 0.015])
    ax.set_ylabel(r'$\mathregular{z\, (km)}$', fontsize=24, color='k')
    ax.set_yticks(np.arange(0, 15, 2))
    ax.set_ylim([0, 14])
    ax.set_xlabel(r'$\mathregular{T\, (K)}$', fontsize=24, color=blue)
    ax2.set_xlabel(r'$\mathregular{r_l\, \left(kgkg^{-1}\right)}$', fontsize=24, color=orange, labelpad=7)
    ax2.tick_params(axis='x', colors=orange, labelsize=20, pad=2)
    ax.tick_params(axis='x', colors=blue, labelsize=20, pad=4)
    ax.tick_params(axis='y', colors='k', labelsize=20, pad=10)
    ax.set_title('(a)', fontsize=20)

    ax = axs[1]
    ax.axvline(0, color='k', linestyle='--', linewidth=2, zorder=2)
    ax.plot(buoyancy_interp(all_z), all_z / 1000., linewidth=2, color=blue, zorder=1)
    ax.plot(buoyancy_perturb(all_z), all_z / 1000., '-.', color='gray', linewidth=2, zorder=1)
    ax.text(0.05, 6.5, r'$\mathregular{\Delta\theta_e = 1K}$', color='gray', fontsize=18)
    ax.scatter([buoyancy_interp(lnbs[0]), buoyancy_perturb(perturb_lnbs[-1])],
               [lnbs[0] / 1000., perturb_lnbs[-1] / 1000.], color=green, marker='*', s=185, edgecolors='k', zorder=3)
    print(lnbs[0] / 1000., perturb_lnbs[-1] / 1000.)
    ax.set_xticks([-0.05, 0, 0.05, 0.1])
    ax.set_xlabel(r'$\mathregular{b\, (ms^{-2})}$', fontsize=24)
    ax.set_title('(b)', fontsize=20, pad=62)
    ax.tick_params(colors='k', labelsize=20, pad=4)
    ax.set_yticks(np.arange(0, 15, 2))
    ax.set_ylim([0, 14])
    ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig('../results/lift_discontinuity.pdf')
    plt.savefig('../results/lift_discontinuity.png')
    plt.show()


def heating_parcel(hurr, save=False):
    rt = 0.014

    theta_e_before = np.linspace(340., 340.69348, 20, endpoint=True)
    theta_e_after = np.linspace(340.6935, 341., 20)
    theta_e_at = np.linspace(340.69348, 340.6935, 100, endpoint=True)
    theta_e_list = np.hstack((theta_e_before, theta_e_at, theta_e_after))
    ref_interps = initial_state_interp(hurr)

    e_a = np.zeros_like(theta_e_list)
    z_r = np.zeros_like(theta_e_list)
    G_thetae = np.zeros_like(theta_e_list)
    G_rt = np.zeros_like(theta_e_list)

    count = 0
    for theta_e in theta_e_list:
        e_a[count], z_r[count], G_thetae[count], G_rt[count] = ape_density_parcel(hurr, theta_e, rt, -0.18, ref_interps)
        count += 1

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), facecolor='w', edgecolor='k')
    ax = axs[0]
    # ax.set_yticks(np.arange(0, 801, 100))
    ax.set_ylim([0, 800])
    ax.set_xticks(np.arange(340., 341.1, 0.5))
    ax.plot(theta_e_list, e_a, color='k', linewidth=2)
    ax.set_xlabel(r'$\mathregular{\theta_e\;\left(K\right)}$', fontsize=20)
    ax.set_ylabel(r'$\mathregular{e_a\;\left(Jkg^{-1}\right)}$', fontsize=20)
    ax.tick_params(labelsize=16, pad=5)

    ax = axs[1]
    ax.plot(theta_e_list, G_thetae/hurr['cp'], color='k', linewidth=2)
    ax.set_xticks(np.arange(340., 341.1, 0.5))
    ax.set_ylim([0, 0.3])
    ax.set_xlabel(r'$\mathregular{\theta_e\;\left(K\right)}$', fontsize=20)
    ax.set_ylabel(r'$\mathregular{\varepsilon_{\theta_e}}$', fontsize=24)
    ax.tick_params(labelsize=16, pad=5)
    # ax.tick_params(axis='y', pad=5)

    plt.tight_layout()
    if save:
        plt.savefig('../results/discontinuity_heating_efficiency.png', dpi=400)
    plt.show()


if __name__ == '__main__':
    directory = '../data/J30pt3'
    hurr = read_fortran_output(directory)
    subplots_buoyancy(hurr, 340., 0.014)
