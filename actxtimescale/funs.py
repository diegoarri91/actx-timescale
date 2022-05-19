import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spiketrain.sptr import SpikeTrains

from actxtimescale.data.celldata import CellData
from actxtimescale.dgestimation import estimate_surrogate_autocorr, fit_autocorr_minimize, bias_variance_logtau
from actxtimescale.dichotomized_gaussian import DichotomizedGaussian
from actxtimescale.plot import set_labels
from actxtimescale.utils import raw_autocorrelation

import warnings

warnings.filterwarnings('ignore')

def get_autocorrelation(cell, t0bin, tfbin, dtbin, biased, root_folder='', bootstrap=False):
    """
    Computes autocorrelation
    """
    try:
        cd = CellData(folder=root_folder + cell)
        dt = cd.dt

        if not(cd.df is None):
            cd.df['end_stimulus'] = cd.df['trigger'] + cd.df['duration']
            cd.df['next_trigger'] = np.append(cd.df.iloc[1:]['trigger'].values, float(dt * cd.n_samples))
            cd.df['after_stimulus_period'] = cd.df['next_trigger'] - cd.df['end_stimulus']

            t_last_spk = cd.arg_spikes[-1, 0] * dt

            idx = list(cd.df[(cd.df['after_stimulus_period'] > tfbin) & (cd.df.trigger < t_last_spk)].index)

            if len(idx) > 0:
                mask_spikes = cd.get_mask_spikes_after_stimulus(idx, t0=t0bin, tf=tfbin)
            else:
                return [np.nan] * 11
                
        else:
            tf_data = float(dt * cd.n_samples)
            triggers = np.arange(0, tf_data, tfbin - t0bin)
            mask_spikes = cd.get_mask_spikes_from_triggers(triggers, t0=0, tf=tfbin - t0bin)
            
        t = np.arange(0, mask_spikes.shape[0], 1) * dt
        if bootstrap:
            idx_bs = np.random.choice(mask_spikes.shape[1], size=mask_spikes.shape[1], replace=True)
            mask_spikes = mask_spikes[:, idx_bs]
        st = SpikeTrains(t, mask_spikes)

        tbins = np.arange(0, t[-1] + dtbin, dtbin)
        n_trials = mask_spikes.shape[1]
        binned_spk_count = st.spike_count(tbins)
        autocor = raw_autocorrelation(binned_spk_count, biased=biased)

        mean_autocor = np.nanmean(autocor, axis=1)
        sd_autocor = np.nanstd(autocor, axis=1)

        n_spikes = np.sum(mask_spikes, 0)
        mean_n_spikes = np.mean(n_spikes)
        sd_n_spikes = np.std(n_spikes, ddof=1)
        n_spikes = np.sum(n_spikes)

        isi = st.interspike_intervals   ()
        mean_isi = np.mean(isi)
        sd_isi = np.std(isi, ddof=1)
        
        lam = n_spikes / n_trials / (tfbin - t0bin)
        offset = (lam * dtbin)**2

        mse_mean = np.mean((mean_autocor - np.mean(mean_autocor))**2)
        mse_offset = np.mean((mean_autocor - offset)**2)
            
        return [dt, n_spikes, n_trials, mean_n_spikes, sd_n_spikes, mean_isi, sd_isi, mean_autocor, sd_autocor, mse_mean, mse_offset]
    
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print(cell)
        return [np.nan] * 11
    
    
def get_autocorr_after_stimulus(cd, t0bin, tfbin):
    
    if not(cd.df is None):
        cd.df['end_stimulus'] = cd.df['trigger'] + cd.df['duration']
        cd.df['next_trigger'] = np.append(cd.df.iloc[1:]['trigger'].values, float(cd.dt * cd.n_samples))
        cd.df['after_stimulus_period'] = cd.df['next_trigger'] - cd.df['end_stimulus']

        t_last_spk = cd.arg_spikes[-1, 0] * cd.dt

        idx = list(cd.df[(cd.df['after_stimulus_period'] > tfbin) & (cd.df.trigger < t_last_spk)].index)

        if len(idx) > 0:
            mask_spikes = cd.get_mask_spikes_after_stimulus(idx, t0=t0bin, tf=tfbin)
        else:
            return None
                
    else:
        tf_data = float(cd.dt * cd.n_samples)
        triggers = np.arange(0, tf_data, tfbin - t0bin)
        mask_spikes = cd.get_mask_spikes_from_triggers(triggers, t0=0, tf=tfbin - t0bin)
        
    return mask_spikes


def fit_exponential_fixed_offset(t0bin, tfbin, dtbin, n_spikes, n_trials, autocorr, tf_autocorr, tau0s, A0s, offset0s=None, 
                                 n_init=40, bounds=(-np.inf, np.inf), fit_offset=False, uniform=True, seed=None):
    """
    Fits exponential function to autocorrelation
    """
    try:
        autocorr = np.array(autocorr)
        lam = n_spikes / n_trials / (tfbin - t0bin)
        offset_bin = (lam * dtbin)**2
        
        h = np.arange(0, len(autocorr), 1)

        arg_aux = 1 + int(50 / dtbin)
        arg0_autocorr = 1 + np.argmax(autocorr[1:arg_aux])
        argf_autocorr = np.searchsorted(h * dtbin, tf_autocorr)
        t0_autocorr = h[arg0_autocorr] * dtbin
        
        if uniform:
            np.random.seed(seed)
            logtau0_min, logtau0_max = np.log(np.min(tau0s)) * 0.9, np.log(np.max(tau0s)) * 1.4
            tau0ss = np.exp(np.random.rand(n_init) * (logtau0_max - logtau0_min) + logtau0_min)
            
        Abin_est, tau_est, offset_bin_est, mse_lsq = fit_autocorr_minimize(dtbin, autocorr, offset_bin, tau0ss, A0s, w0=0, winf=0, 
                                                           arg0_autocorr=arg0_autocorr, tf_autocorr=tf_autocorr, 
                                                           fit_offset=fit_offset, offset0s=offset0s, bounds=bounds)

        mse_offset = np.mean((autocorr - offset_bin)**2)
        
        return [t0_autocorr, Abin_est, tau_est, offset_bin_est, mse_lsq]
        
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return [np.nan] * 5
    
    
def estimate_surrogate(t0bin, tfbin, dtbin, biased, n_spikes, n_trials, dtdg, t0_autocorr, tf_autocorr, tau_true, Abin,
                       offset_bin, tau0s, A0s, n_init=40, bounds=(-np.inf, np.inf), shared_hem_tau=False,
                       log_tau_hemisphere=None, offset0s=None, fit_offset=False, uniform=True, seed=None):
    """
    Given the exponential fit of the autocorrelation, generates DG model with the exponential autocorrelations,
    generate samples and refits an exponential to obtain a new tau estimation
    """
    tf = tfbin - t0bin
    
    arg0_autocorr = int(t0_autocorr / dtbin)

    if shared_hem_tau:
        tau_true = log_tau_hemisphere
    
    if not(fit_offset):
        lam = n_spikes / n_trials / tf
    else:
        lam = np.sqrt(offset_bin) / dtbin
        
    if uniform:
        np.random.seed(seed)
        logtau0_min, logtau0_max = np.log(np.min(tau0s)) * 0.9, np.log(np.max(tau0s)) * 1.4
        tau0ss = np.exp(np.random.rand(n_init) * (logtau0_max - logtau0_min) + logtau0_min)
    
    fr, mu_est, var_est, autocorr_samples, Abin_dg, tau_exp, offset_bin_dg, mse, max_error, cholesky = \
        estimate_surrogate_autocorr(dtdg, dtbin, tf, lam, Abin, tau_true, tau0ss, A0s, fit_offset=fit_offset,
                                    offset0s=offset0s, arg0_autocorr=arg0_autocorr, tf_fit_autocorr=tf_autocorr,
                                    biased=bool(biased), n_trials=n_trials, n_bootstrap=1, seed=seed, uniform=uniform,
                                    n_init=n_init, bounds=bounds)

    return [fr, mu_est, var_est, autocorr_samples, Abin_dg, tau_exp, offset_bin_dg, mse, max_error, cholesky]


def bias_variance_tau(data, df_autocorr):
    
    idx = data['index'].iloc[0]
    n_samples = len(data)
    autocorr_true = df_autocorr.loc[idx, 'autocorr']
    
#     data = data[(data.tau > 0) & (data.tau < 5e3)]
    tau5, tau95 = data.tau_est.quantile(0.025), data.tau_est.quantile(0.975)
    data = data[(data.tau_est >= tau5) & (data.tau_est <= tau95)]
    
#     if data.tau.isna().all():
#         return pd.Series([np.nan] * 11, index=['n_samples', 'meanA', 'bias_autocorr', 'var_autocorr', 'bias_tau', 'var_tau', 'cholesky', 'max_error'])
    try:
        autocorr_est = np.stack(data['raw_autocorr'].values, 1)
    except Exception as e:
        print(data.tau_est.describe())
    
    mean_autocorr_est = np.nanmean(autocorr_est, 1)
    bias_autocorr = mean_autocorr_est - autocorr_true[:len(mean_autocorr_est)]
    var_autocorr = np.nanmean(autocorr_est**2, 1) - mean_autocorr_est**2
    
    tau_true = data['tau_true'].iloc[0]
    mean_tau_est = data['tau_est'].mean()
    bias_tau = mean_tau_est - tau_true
    var_tau = (data['tau_est']**2).mean() - mean_tau_est**2
    
    mean_A_est = data['A'].mean()
    max_error = data['max_error'].mean() 
    cholesky = data['cholesky'].mean()
    
    vals = [n_samples, mean_A_est, bias_autocorr, var_autocorr, bias_tau, var_tau, cholesky, max_error]
    
    return pd.Series(vals, index=['n_samples', 'meanA', 'bias_autocorr', 'var_autocorr', 'bias_tau', 
                                  'var_tau', 'cholesky', 'max_error'])

def bias_variance(data, df_autocorr):
    
    idx = data['index'].iloc[0]
    n_samples = len(data)
    autocorr_true = df_autocorr.loc[idx, 'autocorr']
    max_error = data['max_error'].iloc[0]
    shared_hem_tau = data['shared_hem_tau'].iloc[0]
    
    if not(shared_hem_tau):
        logtau_true = data['logtau_est'].iloc[0]
    else:
#         hemisphere = data['hemisphere'].iloc[0]
#         logtau_true = logtau_hem[hemisphere]
        logtau_true = data['logtau_hemisphere'].iloc[0]
    
    cholesky = data['cholesky'].mean()
    data = data[data.cholesky.astype(bool)].copy()
    
    bias_autocor, var_autocor, bias_logtau, var_logtau, mean_A_est = bias_variance_logtau(data, logtau_true, autocor_true=autocorr_true)
    
    vals = [n_samples, bias_autocor, var_autocor, bias_logtau, var_logtau, mean_A_est, cholesky, max_error]
    
    return pd.Series(vals, index=['n_samples', 'bias_autocor', 'var_autocor', 'bias_logtau', 
                                  'var_logtau', 'meanA', 'cholesky', 'max_error'])

def neuron_summary_layout():
    
    fig, axs = plt.subplots(figsize=(16, 12), nrows=3, ncols=4)
    axs = np.concatenate((axs))
    
    axs[0].set_xlabel('time (ms)')
    
    axs[1].set_xlabel('time (ms)')
    
    axs[2].set_xlabel('time (ms)')
    
#     axs[1].set_title('psth')
#     axs[1].set_ylabel('')
    
#     myplt.set_labels(axs[2], 'speed', 'firing rate (Hz)', 'tuning (only stimulus duration)')
    
    axs[3].set_xlim(0, 500)
    set_labels(axs[3], 'interval (ms)', 'pdf', 'isi distribution')
    
    return fig, axs

def neuron_autocor_layout():
    
    fig, axs = plt.subplots(figsize=(16, 3), nrows=1, ncols=4)
    
    axs[0].set_xlabel('time (ms)')
    
    axs[1].set_xlabel('time (ms)')
    
    axs[2].set_xlabel('time (ms)')
    
    axs[3].set_xlim(0, 600)
    set_labels(axs[3], 'interval (ms)', 'pdf', 'isi distribution')
    
    return fig, axs


def exp_autocorr(theta, dt, h, offset):
    return offset + theta[0] *  np.exp(-h * dt / theta[1])


def plot_autocorrelation_summary(axs, st, row, sample_dg=True, bins_isi=None, plot_isi_dg=False):
    ms = 4
    arg0_plot, arg0_autocorr = 1, 1
    autocorr, bias_autocorr, var_autocorr = np.array(row['autocorr']), np.array(row['bias_autocor']), np.array(row['var_autocor'])
    t0bin, tfbin, dtbin, tf_autocorr, fit_offset  = row[['t0bin', 'tfbin', 'dtbin', 'tf_autocorr', 'fit_offset']]
    n_spikes, n_trials, A, tau, bias_tau, var_tau, r2 = row[['n_spikes', 'n_trials', 'Abin_est', 'tau_est', 'bias_tau', 'var_tau', 'r2']]
    n_trials = int(n_trials)
    
    dt_dg = 1
    
    if not(fit_offset):
        lam = n_spikes / n_trials / (tfbin - t0bin)
    else:
        print(row['offset_bin'])
        offset = row['offset_bin']
        lam = np.sqrt(offset) / dtbin

    fr = lam * 1000
    
    h = np.arange(0, len(autocorr), 1)
    t_autocorr = h * dtbin
    offset = (lam * dtbin)**2
#     argf_autocorr = np.searchsorted(t_autocorr, min([tau * 10, tf_autocorr]))
    argf_autocorr = np.searchsorted(t_autocorr, tf_autocorr)
#     argf_autocorr = max([argf_autocorr, 5])
    argf_plot = argf_autocorr

    if sample_dg and bias_autocorr is not None:
        autocorr_est = bias_autocorr[:argf_autocorr] + autocorr[:argf_autocorr]
        sd_autocorr = np.sqrt(var_autocorr)
    
    n_bin = int(dtbin / dt_dg)
    dd = np.exp(-dt_dg / tau)
    A_no_bin = A * np.exp(-dtbin / tau) * (1 - 2 * dd + dd**2) / (1 - 2 * dd**n_bin + dd**(2 * n_bin))
    offset_no_bin = (lam * dt_dg)**2
    t_no_bin = np.arange(0, tfbin - t0bin, dt_dg)
    exp_autocorr_no_bin = offset_no_bin + A_no_bin * np.exp(-t_no_bin / tau)
    exp_autocorr_no_bin[0] = lam * dt_dg * (1 - lam * dt_dg)
    
    isi = st.isi_distribution()
    
    if sample_dg and bias_autocorr is not None:
        dg = DichotomizedGaussian(lam=lam, raw_autocorrelation=exp_autocorr_no_bin)
        dg.set_t(t_no_bin)
        mask_spikes_samples = dg.sample(t_no_bin, shape=(n_trials,), cholesky=True)
        st_samples = SpikeTrains(t_no_bin, mask_spikes_samples)
        isi_samples = st_samples.isi_distribution()
        n_spikes_samples = np.sum(mask_spikes_samples)
        fr_samples = n_spikes_samples / n_trials / (tfbin - t0bin) * 1000
    
    st.plot(ax=axs[0], ms=ms, mew=0, marker='.', color='C0')
    axs[0].set_title('n_spikes=' + str(int(n_spikes)) + '   fr=' + str(np.round(fr, 1)) + 'Hz' +\
                    '\n n_trials=' + str(n_trials))
    
    axs[1].plot(t_autocorr[arg0_plot:argf_plot], autocorr[arg0_plot:argf_plot], color='C0', label='data')
    axs[1].plot(t_autocorr[arg0_autocorr:argf_autocorr], exp_autocorr([A, tau], dtbin, h, offset)[arg0_autocorr:argf_autocorr], color='C1', label='exp_fit')
    axs[1].text(0.65, 0.95, 'tau=' + str(np.round(tau, 1)) + 'ms', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[1].text(0.65, 0.81, 'r2=' + str(np.round(r2, 3)), horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[1].plot(t_autocorr[arg0_autocorr:argf_autocorr], np.zeros(argf_autocorr - arg0_autocorr) + offset, 'k--', lw=0.5)
    
    if bins_isi is None and sample_dg:
        _, bins_isi = np.histogram(np.concatenate([isi, isi_samples]), bins=20)
        
    axs[1].set_title('raw autocorrelation_' + str(dtbin) + 'ms')    
    axs[3].hist(isi, bins=bins_isi, density=True, alpha=0.5, label='data')

    if sample_dg:
        axs[1].plot(t_autocorr[arg0_autocorr:argf_autocorr], autocorr_est[arg0_autocorr:argf_autocorr], color='C3', label='sampled')
        axs[1].fill_between(t_autocorr[arg0_autocorr:argf_autocorr], autocorr_est[arg0_autocorr:argf_autocorr] - sd_autocorr[arg0_autocorr:argf_autocorr], 
                                                                     autocorr_est[arg0_autocorr:argf_autocorr] + sd_autocorr[arg0_autocorr:argf_autocorr], alpha=0.5, color='C3')
#         axs[1].text(0.65, 0.9, 'tau bias=' + str(np.round(bias_tau, 1)) + 'ms', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].text(0.65, 0.88, 'tau sd=' + str(np.round(np.sqrt(var_tau), 1)) + 'ms', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        st_samples.plot(ax=axs[2], ms=ms, mew=0, marker='.', color='C3')
        axs[2].set_title('n_spikes=' + str(int(n_spikes_samples)) + '   fr=' + str(np.round(fr_samples, 1)) + 'Hz')
        
    if sample_dg and plot_isi_dg:
        axs[3].hist(isi_samples, bins=bins_isi, density=True, alpha=0.5, label='dg', color='C3')
