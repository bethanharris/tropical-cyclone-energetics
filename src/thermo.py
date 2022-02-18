import numpy as np
import gma

g = 9.81
cpd = 1004.5
cvd = 717.5
cl = 4179.9
ci = 2106
Lv = 2.5e6
Lf = 3.34e5
Ls = Lv + Lf
es0_mb_ice = 6.107
p0_mb = 1000.
p0_pa = p0_mb * 100.
kappa = gma.Rd / cpd
epsilon = gma.Rd / gma.Rv


def spec_vol(PI, theta_rho):
    num = gma.Rd * theta_rho
    denom = p0_pa * (PI ** (cvd / gma.Rd))
    return num / denom


def theta_v(theta, qv):
    theta_v = theta * (1 + 0.61 * qv)
    return theta_v


def theta_rho_exact(theta, rv, rl, ri):
    theta_rho = theta * (1. + rv / epsilon) / (1. + rv + rl + ri)
    return theta_rho


def pressure_from_PI(PI):
    pressure = p0_mb * (PI) ** (1. / kappa)
    return pressure


def PI_from_pressure(pressure_mb):
    PI = (pressure_mb / p0_mb) ** kappa
    return PI


def theta_from_T(t_K, p_mb):
    theta = t_K * (p0_mb / p_mb) ** (gma.Rd / cpd)
    return theta


def T_from_theta(theta_K, p_mb):
    t_K = theta_K * (p_mb / p0_mb) ** (gma.Rd / cpd)
    return t_K


def buoyancy_spec_vol(spec_vol, spec_vol_ref):
    b = g * (spec_vol - spec_vol_ref) / spec_vol_ref
    return b


def theta_e_exact(t_c, p_mb, rt_kgkg):
    qt_kgkg = specific_humidity_from_mixing_ratio(rt_kgkg)
    rl_kgkg = gma.liquid_mixing_ratio(t_c, p_mb, qt_kgkg)
    rv_kgkg = rt_kgkg - rl_kgkg
    cp_eff = cpd + rt_kgkg * cl
    e_mb = gma.partial_vapour_pressure(t_c, p_mb, qt_kgkg)
    pd_mb = p_mb - e_mb
    es_mb = gma.saturation_vapour_pressure(t_c)
    rel_hum = e_mb / es_mb
    theta_term = (t_c + gma.t0) * (1000. / pd_mb) ** ((1. - rt_kgkg) * gma.Rd / cp_eff)
    humidity_term = rel_hum ** (-rv_kgkg * gma.Rv / cp_eff)
    exp_term = np.exp(Lv * rv_kgkg / (cp_eff * (t_c + gma.t0)))
    theta_e = theta_term * humidity_term * exp_term
    return theta_e


def theta_ei_exact(t_c, p_mb, rt_kgkg):
    rv_kgkg, rl_kgkg, ri_kgkg = gma.all_mixing_ratios(t_c, p_mb, rt_kgkg)
    frozen = ri_kgkg > 1.e-7
    cp_eff = cpd + rt_kgkg * ci
    qt_kgkg = specific_humidity_from_mixing_ratio(rt_kgkg)
    e_mb = gma.partial_vapour_pressure(t_c, p_mb, qt_kgkg)
    pd_mb = p_mb - e_mb
    esl_mb = gma.saturation_vapour_pressure(t_c)
    esi_mb = gma.saturation_vapour_pressure_ice(t_c)
    es_mb = frozen * esi_mb + ~frozen * esl_mb
    rel_hum = e_mb / es_mb
    theta_term = (t_c + gma.t0) * (1000. / pd_mb) ** (gma.Rd / cp_eff)
    humidity_term = rel_hum ** (-rv_kgkg * gma.Rv / cp_eff)
    exp_term_rv = np.exp(Ls * rv_kgkg / (cp_eff * (t_c + gma.t0)))
    exp_term_rl = np.exp(Lf * rl_kgkg / (cp_eff * (t_c + gma.t0)))
    theta_ei = theta_term * humidity_term * exp_term_rv * exp_term_rl
    return theta_ei


def spec_vol_from_thetae_rt(theta_e, rt, p_mb):
    if isinstance(theta_e, np.float32) or isinstance(theta_e, np.float64) or isinstance(theta_e, float):
        theta_e = np.array([theta_e])
        p_mb = np.array([p_mb])
        rt = np.array([rt])
    qt = specific_humidity_from_mixing_ratio(rt)

    dt_bound = 1.e-8  # desired precision of final lifted temperature

    lower_bound = -274.
    upper_bound = 500.  # Upper and lower ends of initial range to bisect

    low = np.full(theta_e.shape, lower_bound)
    high = np.full(theta_e.shape, upper_bound)

    # Find number of bisection steps required to reach desired accuracy
    steps = np.ceil(np.log2((upper_bound - lower_bound) / dt_bound))

    # Bisect
    for _ in np.arange(0, steps):
        mid = (low + high) / 2.
        theta_e_mid = theta_e_exact(mid, p_mb, rt)
        delta_theta_e_mid = theta_e_mid - theta_e
        pos = np.where(delta_theta_e_mid > 0.)
        neg = np.where(delta_theta_e_mid < 0.)
        low[neg] = mid[neg]
        high[pos] = mid[pos]
        mid_new = (low + high) / 2.

    t2_c = mid_new

    t_K = t2_c + gma.t0

    spec_vol = gma.spec_vol(t2_c, p_mb, qt)

    return spec_vol, t_K


def specific_humidity_from_mixing_ratio(mixing_ratio):
    specific_humidity = mixing_ratio / (1 + mixing_ratio)
    return specific_humidity


def saturation_vapour_pressure_ice(t_c):
    t_K = t_c + gma.t0
    es_mb = es0_mb_ice * np.exp(Ls * (t_K - gma.t0) / (gma.Rv * t_K ** 2))
    return es_mb
