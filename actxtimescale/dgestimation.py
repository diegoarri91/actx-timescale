import itertools
import numpy as np
from scipy.optimize import least_squares, minimize

from actxtimescale.dichotomized_gaussian import DichotomizedGaussian
from actxtimescale.utils import raw_autocorrelation
from spiketrain.sptr import SpikeTrains


def exp_fixed_offset_residuals(theta, dt, h, offset, y):
    return offset + theta[0] *  np.exp(-h * dt / theta[1])  - y


def exp_fixed_offset_lsq_regularized(theta, dt, h, offset, y, w0, winf):
    return np.sum((offset + theta[0] *  np.exp(-h * dt / theta[1])  - y)**2) + w0 / theta[1]**2 + winf * theta[1]**2


def jac_exp_fixed_offset_lsq_regularized(theta, dt, h, offset, y, w0, winf):
    res = (offset + theta[0] *  np.exp(-h * dt / theta[1])  - y)
    dx0 = 2 * np.sum(res * np.exp(-h * dt / theta[1]))
    dx1 = 2 * np.sum(res * theta[0] *  np.exp(-h * dt / theta[1]) * h * dt / theta[1]**2) - 2 * w0 / theta[1]**3 + 2 * winf * theta[1]
    return np.array([dx0, dx1]).T


def exp_fixed_lsq_regularized(theta, dt, h, y, w0, winf):
    return np.sum((theta[2] + theta[0] *  np.exp(-h * dt / theta[1])  - y)**2) + w0 / theta[1]**2 + winf * theta[1]**2


def jac_exp_lsq_regularized(theta, dt, h, y, w0, winf):
    res = (theta[2] + theta[0] *  np.exp(-h * dt / theta[1])  - y)
    dx0 = 2 * np.sum(res * np.exp(-h * dt / theta[1]))
    dx1 = 2 * np.sum(res * theta[0] *  np.exp(-h * dt / theta[1]) * h * dt / theta[1]**2) - 2 * w0 / theta[1]**3 + 2 * winf * theta[1]
    dx2 = 2 * np.sum(res)
    return np.array([dx0, dx1, dx2]).T


def fit_exponential(y, dt, args, tau0s=None):

    tau0s = tau0s if tau0s is not None else [10, 20, 40, 80, 160, 320, 640]
    best_mse = np.inf
    for tau0 in tau0s:
        theta0 = np.array([y[0], tau0, y[-100].mean()])
        _res_lsq = minimize(exp_fixed_lsq_regularized, theta0.copy(), method='L-BFGS-B',
                            options=dict(ftol=1e-30, gtol=1e-20, maxiter=100),
                            tol=1e-30, jac=jac_exp_lsq_regularized,
                            args=(dt, args, y, 0., 0.))

        if _res_lsq['fun'] < best_mse and _res_lsq['x'][0] > 0 and _res_lsq['x'][1] > 0:
            res_lsq = _res_lsq.copy()
            A = _res_lsq['x'][0]
            tau = _res_lsq['x'][1]
            offset = _res_lsq['x'][2]
            best_mse = _res_lsq['fun']

    dic_results = {
        'params': {
            'tau': tau,
            'A': A,
            'offset': offset
        },
        'mse': best_mse
    }
    return dic_results

def fit_autocorr_minimize(dtbin, mean_autocorr, offset, tau0s, A0s, fit_offset=False, offset0s=None, w0=0, winf=0,
                          arg0_autocorr=1, tf_autocorr=1000, bounds=(-np.inf, np.inf)):
    
    h = np.arange(0, len(mean_autocorr), 1)
#     arg0_autocorr = 1
    argf_autocorr = np.searchsorted(h * dtbin, tf_autocorr)
    
    best_mse = np.inf

    for tau0, A0 in itertools.product(tau0s, A0s):

        if not fit_offset:
            theta0 = np.array([A0, tau0])# * (1 + np.random.randn(2) * 0.2)
            _res_lsq = minimize(exp_fixed_offset_lsq_regularized,
                                theta0.copy(),
                                method='L-BFGS-B',
                                options=dict(ftol=1e-30, gtol=1e-20, maxiter=40),
                                tol=1e-30,
                                jac=jac_exp_fixed_offset_lsq_regularized,
                                args=(dtbin,
                                      h[arg0_autocorr:argf_autocorr].copy(),
                                      offset,
                                      mean_autocorr[arg0_autocorr:argf_autocorr].copy(),
                                      w0,
                                      winf),
                                bounds=bounds)
        else:
            _bounds = bounds + ([0, np.inf], )
            for _offset in offset0s:
                theta0 = np.array([A0, tau0, _offset])# * (1 + np.random.randn(2) * 0.2)
                _res_lsq = minimize(exp_fixed_lsq_regularized,
                                    theta0.copy(),
                                    method='L-BFGS-B',
                                    options=dict(ftol=1e-30, gtol=1e-20, maxiter=40),
                                    tol=1e-30,
                                    jac=jac_exp_lsq_regularized,
                                    args=(dtbin,
                                          h[arg0_autocorr:argf_autocorr].copy(),
                                          mean_autocorr[arg0_autocorr:argf_autocorr].copy(),
                                          w0,
                                          winf),
                                    bounds=_bounds)
                
        if _res_lsq['fun'] < best_mse and _res_lsq['x'][0] > 0 and _res_lsq['x'][1] > 0:
            res_lsq = _res_lsq.copy()
            best_mse = _res_lsq['fun']

    if np.isinf(best_mse):
        return np.nan, np.nan, np.nan, np.nan
    
    A = res_lsq['x'][0]
    tau = res_lsq['x'][1]
    offset_est = np.nan if not fit_offset else res_lsq['x'][2]
    
#     mse_lsq = res_lsq['fun'] / (argf_autocorr - arg0_autocorr)
    _offset = np.nan if not fit_offset else res_lsq['x'][2]
    mse_lsq = np.mean(exp_fixed_offset_residuals(res_lsq['x'],
                                                 dtbin,
                                                 h[arg0_autocorr:argf_autocorr],
                                                 offset,
                                                 mean_autocorr[arg0_autocorr:argf_autocorr])**2)
    
    return A, tau, offset_est, mse_lsq


def fit_autocorr(dtbin, mean_autocorr, offset, tau0s, A0s, tf_autocorr=1000, bounds=(-np.inf, np.inf)):
    
    h = np.arange(0, len(mean_autocorr), 1)
    arg0_autocorr = 1
    argf_autocorr = np.searchsorted(h * dtbin, tf_autocorr)
    
    best_mse = np.inf
    for tau0, A0 in itertools.product(tau0s, A0s):
        theta0 = np.array([A0, tau0])
        _res_lsq = least_squares(exp_fixed_offset_residuals, theta0.copy(), 
                                args=(dtbin,
                                      h[arg0_autocorr:argf_autocorr].copy(),
                                      offset,
                                      mean_autocorr[arg0_autocorr:argf_autocorr].copy()),
                                bounds=bounds)
        if _res_lsq['cost'] < best_mse:
            res_lsq = _res_lsq.copy()
            best_mse = _res_lsq['cost']
            
    A = res_lsq['x'][0]
    tau = res_lsq['x'][1]
    mse_lsq = res_lsq['cost'] / (argf_autocorr - arg0_autocorr) * 2
    
    return A, tau, mse_lsq


def estimate_parameters(dt, dtbin, mask_spikes, tau0s, A0s, tf_dg_autocorr=1000, bounds=(-np.inf, np.inf),
                        n_bootstrap=1, uniform=False):
    lent = mask_spikes.shape[0]
    n_trials = mask_spikes.shape[1]
    
    n_spikes = np.sum(mask_spikes, 0)
    mean_n_spikes = np.mean(n_spikes)
    sd_n_spikes = np.std(n_spikes, ddof=1)
    n_spikes = np.sum(n_spikes)
    lam = n_spikes / n_trials / (lent * dt)
    offset = (lam * dtbin)**2
    
    t = np.arange(0, mask_spikes.shape[0], 1) * dt
    st = SpikeTrains(t, mask_spikes)
    tbins = np.arange(0, lent * dt + dtbin, dtbin)
    binned_spk_count = st.spike_count(tbins, normed=False)
    autocorr = raw_autocorrelation(binned_spk_count, biased=False)

    mean_autocorr = np.nanmean(autocorr, axis=1)
    sd_autocorr = np.nanstd(autocorr, axis=1)

    isi = st.isi_distribution()
    mean_isi = np.mean(isi)
    sd_isi = np.std(isi, ddof=1)

    mse_mean = np.mean((mean_autocorr - np.mean(mean_autocorr))**2)
    mse_offset = np.mean((mean_autocorr - offset)**2)
    
    if uniform:
        logtau0_min, logtau0_max = np.log(np.min(tau0s)) * 0.9, np.log(np.max(tau0s)) * 1.4
        tau0s = np.exp(np.random.rand(n_init) * (logtau0_max - logtau0_min) + logtau0_min)
    
    A, tau, mse_lsq = fit_autocorr(dtbin, mean_autocorr, offset, tau0s, A0s, tf_autocorr=tf_dg_autocorr, bounds=bounds)
    
    return [n_spikes, n_trials, mean_n_spikes, sd_n_spikes, mean_isi, sd_isi, mean_autocorr, sd_autocorr, mse_mean,
            mse_offset, lam, A, tau, mse_lsq]

def exp_autocorr_fixed_offset_res(theta, dt, h, offset, y):
    return offset + theta[0] *  np.exp(-h * dt / theta[1])  - y


def fit_autocorr(n_bootstrap):
    Abin_est, tau_exp, mse = [], [], []
    for ii in range(n_bootstrap):
        best_mse = np.inf
        for tau0, A0 in itertools.product(tau0s, A0s):
#         for jj in range(20):
#             A0 = np.random.rand() * min(2 * A, bounds[1][0])
#             tau0 = np.random.rand() * 2 * tau
#             mu0 = np.random.rand() * 2 * np.log(tau)
            try:
#                 A0 = A0 if A0 > 0 else 1e-4
#                 tau0 = tau if tau > 0 else 1
                theta0 = np.array([A0, tau0])
                _res_lsq = least_squares(exp_autocorr_fixed_offset_res, theta0.copy(), method='lm', 
                                        args=(dtbin, h[1:].copy(), offset_bin, autocorr_samples[1:, ii].copy()), 
                                        bounds=bounds)
            except:
                print('A0', A0) 
                print('tau0', tau0)
                
            if _res_lsq['cost'] < best_mse:
                res_lsq = _res_lsq.copy()
                best_mse = _res_lsq['cost']

        Abin_est.append(res_lsq['x'][0])
#         tau_exp.append(res_lsq['x'][1])

        tau_exp.append(np.exp(res_lsq['x'][1]))
    
        mse.append(res_lsq['cost'] / len(h[1:]) * 2)


def estimate_surrogate_autocorr(dt, dtbin, tf, lam, Abin, tau, tau0s, A0s, fit_offset=False, offset0s=None,
                                arg0_autocorr=1, tf_fit_autocorr=None, tf_dg_autocorr=None, biased=False, w0=0, winf=0,
                                bounds=(-np.inf, np.inf), n_trials=200, n_bootstrap=1, seed=None, uniform=False,
                                n_init=5):

    offset_bin = (lam * dtbin)**2
    offset = (lam * dt)**2
    tf_fit_autocorr = tf_fit_autocorr if tf_fit_autocorr is not None else tau * 5

    if dtbin > dt:
        n_bin = int(dtbin / dt)
        dd = np.exp(-dt / tau)
        A = Abin * np.exp(-dtbin / tau) * (1 - 2 * dd + dd**2) / (1 - 2 * dd**n_bin + dd**(2 * n_bin))
            
    np.random.seed(seed)
    
    tf_dg_autocorr = tf_dg_autocorr if tf_dg_autocorr is not None else min(10 * tau, 3200)
    t_autocorr = np.arange(0, tf_dg_autocorr, dt)
    exp_autocorr = offset + A * np.exp(-t_autocorr / tau)
    exp_autocorr[0] = lam * dt * (1 - lam * dt)
    t = np.arange(0, tf, dt)
    T = len(t)
    
    dg = DichotomizedGaussian(lam=lam, raw_autocorrelation=exp_autocorr)
    dg.set_t(t)
    max_error = dg.max_error
    
    if dg.gp.ch_lower is not None:
        cholesky = 1
        mask_spikes = dg.sample(t, shape=(n_trials, n_bootstrap), cholesky=True)
    else:
        cholesky = 0
        return [[np.nan] * n_bootstrap] * 8 + [np.nan, np.nan]
    
    fr = np.sum(mask_spikes, axis=(0, 1)) / n_trials / tf * 1000
    
    tbins =np.arange(0, tf + dtbin, dtbin)
    x = SpikeTrains(t, mask_spikes).spike_count(tbins)
        
    mu_est = np.mean(x, axis=(0, 1))
    var_est = np.mean((x - mu_est) **2, axis=(0, 1))
    
    autocorr_samples = raw_autocorrelation(x, biased=biased)
    autocorr_samples = np.nanmean(autocorr_samples, 1)
    
    h = np.arange(0, len(autocorr_samples), 1)
    
    if uniform:
        logtau0_min, logtau0_max = np.log(np.min(tau0s)) * 0.9, np.log(np.max(tau0s)) * 1.4
        tau0s = np.exp(np.random.rand(len(tau0s)) * (logtau0_max - logtau0_min) + logtau0_min)

    Abin_est, tau_exp, offset_bin_est, mse = [], [], [], []
    for ii in range(n_bootstrap):
        A_best, tau_best, offset_best, mse_lsq = fit_autocorr_minimize(dtbin,
                                                                       autocorr_samples[:, ii],
                                                                       offset_bin,
                                                                       tau0s,
                                                                       A0s,
                                                                       fit_offset=fit_offset,
                                                                       offset0s=offset0s,
                                                                       w0=w0, winf=winf,
                                                                       arg0_autocorr=arg0_autocorr,
                                                                       tf_autocorr=tf_fit_autocorr,
                                                                       bounds=bounds)
        
        Abin_est.append(A_best)
        tau_exp.append(tau_best)
        offset_bin_est.append(offset_best)
        mse.append(mse_lsq)

    return [fr, mu_est, var_est, autocorr_samples.T, Abin_est, tau_exp, offset_bin_est, mse, max_error, cholesky]


def bias_variance_logtau(data, logtau_true, autocor_true=None):
    
    if autocor_true is not None:
        autocor_est = np.stack(data['raw_autocorr'].values, 1)
        mean_autocor_est = np.nanmean(autocor_est, 1)
        bias_autocor = mean_autocor_est - autocor_true[:len(mean_autocor_est)]
        var_autocor = np.nanvar(autocor_est, 1, ddof=1)
    
    bias_logtau = data['logtau_dg'].mean() - logtau_true
    var_logtau = data['logtau_dg'].var()
    
    mean_A_est = data['Abin_dg'].mean()
    
    return bias_autocor, var_autocor, bias_logtau, var_logtau, mean_A_est


def A_nobin_transform(Abin, dtbin, dt_dg, tau):
    n_bin = int(dtbin / dt_dg)
    dd = np.exp(-dt_dg / tau)
    factor = np.exp(-dtbin / tau) * (1 - 2 * dd + dd**2) / (1 - 2 * dd**n_bin + dd**(2 * n_bin))
    A = Abin * factor
    return A


