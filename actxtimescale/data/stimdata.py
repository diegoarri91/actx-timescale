import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

from spiketrain.sptr import SpikeTrains

from actxtimescale.dgestimation import fit_exponential
from actxtimescale.utils import raw_autocorrelation

class StimData:
    
    def __init__(self, st, df, t0=None, tf=None, neuron=None):
        self.st = st
        self.df = df
        self.t0 = t0
        self.tf = tf
        self.neuron = neuron
        
    def average_psth_aligned_to_peak(self, kernel, psth_time_window, peak_time_window, min_peak_value, peak80_time_window_after_peak):

        def peak_larger_than_min(psth, min_value):
            peak = np.max(psth, axis=0)
            mask_peak_larger_than_min = peak > min_value
            return mask_peak_larger_than_min

        def psth_aligned_to_peak(psth):
            arg_peak = np.argmax(psth, axis=0)
            time_peak = arg_peak * self.st.dt + self.t0
            mask_peak_in_window = (time_peak >= 0) & (time_peak < peak_time_window)
            psth, arg_peak = psth[:, mask_peak_in_window], arg_peak[mask_peak_in_window]
            psth_window = np.take_along_axis(psth, arg_peak[None, :] + np.arange(-arg0, int((psth_time_window + peak80_time_window_after_peak) / self.st.dt), 1)[:, None], axis=0)
            return psth_window, mask_peak_in_window

        def psth_aligned_to_80peak(psth_window):
            arg_80peak, train = np.where((psth_window[arg0:-1] >= psth_window[arg0] * 0.8) & (psth_window[arg0 + 1:] < psth_window[arg0] * 0.8))
            mask_first_80peak = ~pd.Series(train).duplicated().values
            arg_80peak, train = arg_80peak[mask_first_80peak], train[mask_first_80peak]

            time_80peak = arg_80peak * self.st.dt
            mask_80peak_in_window = time_80peak < peak80_time_window_after_peak
            arg_80peak, train = arg_80peak[mask_80peak_in_window], train[mask_80peak_in_window]

            arg_80peak = arg0 + arg_80peak
            psth_window = psth_window[:, train]
            psth_window = np.take_along_axis(psth_window, arg_80peak[None, :] + np.arange(-arg0, int(psth_time_window / self.st.dt), 1)[:, None], axis=0)

            return psth_window, train
        
        psth = self.psth(kernel) * 1000
        ntrains = self.df.n_trials.values

        arg0 = int(20 / self.st.dt)
        # arg0 = 1
        larger_than_min = peak_larger_than_min(psth, min_peak_value)
        psth = psth[:, larger_than_min]
        ntrains = ntrains[larger_than_min]

        psth_window, mask_peak_in_window = psth_aligned_to_peak(psth)
        ntrains = ntrains[mask_peak_in_window]

        psth_window, train = psth_aligned_to_80peak(psth_window)
        ntrains = ntrains[train]

        avg_psth = np.sum(ntrains[None, :] * psth_window, axis=1) / np.sum(ntrains)
        time = np.arange(-arg0, avg_psth.shape[0] - arg0, 1) * self.st.dt
        # print(arg0)
        
        return time, avg_psth

    def fit_average_psth_aligned_to_peak(self, kernel, psth_time_window, peak_time_window, min_peak_value,
                                         peak80_time_window_after_peak):

        arg0 = int(20 / self.st.dt)
        time, avg_psth = self.average_psth_aligned_to_peak(kernel, psth_time_window, peak_time_window,
                                                                min_peak_value, peak80_time_window_after_peak)
        avg_is_nan = np.all(np.isnan(avg_psth))
        if not avg_is_nan:
            args = np.arange(0, avg_psth.shape[0] - arg0, 1)
            dic_results = fit_exponential(avg_psth[arg0:], self.st.dt, args)
        else:
            dic_results = {'params': {'tau': np.nan, 'A': np.nan, 'offset': np.nan}, 'mse': np.nan}

        return time, avg_psth, dic_results

    def plot_average_psth_aligned_to_peak(self, ax, kernel, psth_time_window, peak_time_window, min_peak_value,
                                         peak80_time_window_after_peak):

        time, avg_psth, dic_results = self.fit_average_psth_aligned_to_peak(
            kernel, psth_time_window, peak_time_window, min_peak_value, peak80_time_window_after_peak
        )

        tau = dic_results['params']['tau']
        A = dic_results['params']['A']
        offset = dic_results['params']['offset']

        arg0 = int(20 / self.st.dt)
        time_fit = np.arange(0, avg_psth.shape[0] - arg0, 1) * self.st.dt

        ax.plot(time, avg_psth, lw=5)
        ax.plot(time_fit, A * np.exp(-time_fit / tau) + offset)

        ax.text(0.7, 0.7, 'tau={:.2f} ms'.format(tau), horizontalalignment='center',  verticalalignment='center',
                transform=ax.transAxes)

        ax.set_xlabel('time (ms)')
        ax.set_ylabel('psth (Hz)')

        return ax, tau

    def plot_stimulus_duration(self, ax, groupby, lw=0.5, label_every=1, draw_lines_using_trials=True):
        ax.plot([self.st.t[0], self.st.t[-1]], [-0.5, -0.5], 'k--', lw=lw, alpha=0.5)
        line = - 0.5
        lw = 0.5
        next_val = val = self.df.loc[0, groupby]
        # thick = False
        yticks = []
        yticklabels = []
        lines = [line]
        for ii, idx in enumerate(self.df.index):
            last_line = line
            if draw_lines_using_trials:
                line += self.df.n_trials[idx]
            else:
                line += 1
            if ii < len(self.df.index) - 1:
                val = self.df.iloc[ii][groupby]
                next_val = self.df.iloc[ii + 1][groupby]
            else:
                val = self.df.iloc[ii][groupby]
                next_val = val + 1
            # if use_thick:
            #     thick = (val != next_val)
            # lw = lw_fun(thick)
            ax.plot([self.st.t[0], self.st.t[-1]], [line, line], 'k--', lw=lw, alpha=0.5)
            ax.fill_between([0, self.df.duration[idx]], last_line, line, color='k', alpha=0.1)
            if ii % label_every == 0:
                yticks.append((line + last_line) / 2)
                yticklabels.append(val)
            lines.append(line)
            # if self.df.loc[idx, 'type'] == 'clicktrain':
            #     isi = self.df.loc[idx, 'isi']
            #     for ii in range(9):
            #         l0 = ii * isi + 5
            #         lf = (ii + 1) * isi
            #         stim0 = ii * isi
            #         ax.plot([stim0, stim0], [last_line, line], color='k', linestyle='--', lw=lw, alpha=0.5)
            #         ax.plot([stim0 + 5, stim0 + 5], [last_line, line], color='k', linestyle='--', lw=lw, alpha=0.5)

        ax.set_ylabel(groupby)

        ax.set_xlabel('time (ms)')
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        extra = line * 0.05
        ax.set_ylim(-extra, line + extra)
        
        return lines

    def plot_raster(self, figsize=(5, 4), groupby=None, ax=None, ms=None, mew=None, marker=None):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        mask = self.st.mask.reshape(len(self.st.t), -1, order='F')
        na_trials = np.any(np.isnan(mask), axis=0)
        mask = mask[:, ~na_trials]
        st_plot = SpikeTrains(self.st.t, mask)
        st_plot.plot(ax=ax, ms=ms, mew=mew, marker=marker)

        #### for testing
#         offset = 0
#         for ii, idx in enumerate(self.df.index):
#             mask = self.st.mask[:, :, ii]
#             mask = mask[:, ~np.any(np.isnan(mask), 0)]
#             mask[500, :] = True
#             st2 = SpikeTrains(self.st.t, mask)
#             st2.plot(ax=ax, offset=offset, ms=ms, mew=mew, marker=marker, color='C'+str(ii%10))
#             offset += self.df.n_trials[idx]
    
        if groupby is not None:
            self.plot_stimulus_duration(ax, groupby)
        
        return ax
    
    def plot_psth(self, kernel, groupby, figsize=(5, 4), ax=None, use_thick=False, label_every=1):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        lines = self.plot_stimulus_duration(ax, groupby, draw_lines_using_trials=False)
        lines = np.array(lines)
        
        psth = self.psth(kernel) * 1000
        # scale = self.df.n_trials.max()
        psth_max = np.max(psth)
        # scale = 0.75
        # scale = psth_max * 0.75
        psth = psth / psth_max # * scale * 0.95
        
        ax.plot(self.st.t, psth + lines[None, :-1], color='C0')
        
        t_scale = (self.st.t[-1] - self.st.t[0]) * 0.85 + self.st.t[0]
        
        ax.plot([t_scale] * 2, [lines[2], lines[3]], 'k-')
        # ax.plot([self.st.t[0], self.st.t[-1]], [lines[3], lines[3]], 'k--', alpha=0.5)
        ax.text(t_scale * 1.12, (lines[2] + lines[3]) / 2, '{:.1f}Hz'.format(psth_max), horizontalalignment='center', 
                verticalalignment='center')
    
#         if groupby is not None:
#             lw_fun = lambda b: 1 if b else 0.5
#             ax.plot([self.st.t[0], self.st.t[-1]], [-0.5, -0.5], 'k--', lw=lw_fun(use_thick), alpha=0.5)
#             line = - 0.5
#             lw = 0.5
#             next_val= val = self.df.loc[0, groupby]
#             thick = False
#             pos = []
#             labels = []
#             for ii, idx in enumerate(self.df.index):
#                 last_line = line
#                 line += self.df.n_trials[idx]
#                 if ii < len(self.df.index) - 1:
#                     val = self.df.iloc[ii][groupby]
#                     next_val= self.df.iloc[ii + 1][groupby]
#                 else:
#                     val = self.df.iloc[ii][groupby]
#                     next_val = val+ 1
#                 if use_thick:
#                     thick = (val != next_val)
#                 lw = lw_fun(thick)
#                 ax.plot([self.st.t[0], self.st.t[-1]], [line, line], 'k--', lw=lw, alpha=0.5)
#                 ax.fill_between([0, self.df.duration[idx]], last_line, line, color='k', alpha=0.1)
#                 ax.plot(self.st.t, psth[:, ii] + last_line, color='C0')
#                 if ii % label_every == 0:
#                     pos.append((line + last_line) / 2)               
#                     labels.append(val)
            
#             ax.set_ylabel(groupby)
                    
#             ax.set_xlabel('time (ms)')
#             ax.set_yticks(pos)
#             ax.set_yticklabels(labels)
#             extra = line *0.05
#             ax.set_ylim(-extra, line + extra)
        
#         return ax
    
    def plot_isi(self, t0=None, tf=None, ax=None, bins=None, **kwargs):
        isis = []
        for ii in range(self.st.mask.shape[2]):
            mask = self.st.mask[..., ii]
            mask = mask[:, ~np.any(np.isnan(mask), 0)]
            mask = np.array(mask, dtype=bool)
            st = SpikeTrains(self.st.t, mask)
            isis.append(st.isi_distribution(t0=t0, tf=tf))
        isis = np.concatenate(isis)
        print('isi total counts', isis.shape)
        ax.hist(isis, bins=bins, **kwargs)
        return ax
    
    def psth(self, kernel):
        average_spike_train = SpikeTrains(self.st.t, np.nansum(self.st.mask, 1) / self.df.n_trials.values)
        psth = average_spike_train.convolve(kernel)
        return psth
    
    def sort_values(self, columns, ascending=True):
        self.df = self.df.sort_values(columns, ascending=ascending)
        idx = self.df.index
        self.df = self.df.reset_index(drop=True)
        self.st.mask = self.st.mask[..., idx]
        return self
        
    def fr_stimulus(self, t_post=0):
        arg0_stim = int(-self.t0 / self.st.dt)
        durations = self.df.duration.values
        argfs_stim = np.array((-self.t0 + durations + t_post) / self.st.dt, dtype=int) 
        fr_stim = np.array([np.nansum(self.st.mask[arg0_stim:argf_stim, :, ii], 0) / (duration + t_post) for ii, duration, argf_stim in zip(range(len(self.df)), durations, argfs_stim)]).T
        return fr_stim
    
    def get_autocorrelation(self, tbins, autocorr_type='raw', biased=False):
        
        n_trials = self.st.mask.shape[1] - np.sum(np.isnan(self.st.mask[0]), 0)
        n_trials = n_trials[None, :]
#         print(n_trials.shape)
        binned_spk_count = self.st.spike_count(tbins, normed=False)
        
#         raw_autocorr = np.zeros((len(tbins) - 1, len(fr))) * np.nan
        
        if autocorr_type == 'raw':
            autocorr = raw_autocorrelation(binned_spk_count, biased=biased)
    
        if autocorr_type == 'pearson':
            mean_over_trials = np.nansum(binned_spk_count, axis=1) / n_trials
            variance_over_trials = np.nansum((binned_spk_count - mean_over_trials[:, None, :])**2, axis=1) / (n_trials - 1)
            mask = (variance_over_trials < 1e-10)
            mask = np.stack([mask] * binned_spk_count.shape[1], 1)
#             z = np.zeros(binned_spk_count.shape) * np.nan
            z = (binned_spk_count - mean_over_trials[:, None, :]) / np.sqrt(variance_over_trials[:, None, :])
            z[mask] = 0
            autocorr = raw_autocorrelation(z, biased=biased)
            
        return autocorr

    def get_psth_time_scale_from_peak_decay():
    
        psth = stim_data.psth(kernel) * 1000
        arg_peak = np.argmax(psth, axis=0)
        peak = np.take_along_axis(psth, arg_peak[None, :], axis=0)

        arg_window = arg_peak[None, :] + np.arange(0, 4000, 1)[:, None]
        psth_window = np.take_along_axis(psth, arg_window, axis=0)
        mask = (psth_window[:-1] >= peak * 0.8) & (psth_window[1:] < peak * 0.8)
        # psth = psth[:, np.any(mask, axis=0) & (peak[0] >= 2)]

        ntrains = stim_data.df.n_trials.values
        args, train = np.where(mask)
        mask_train = ~pd.Series(train).duplicated().values
        args = args[mask_train]
        train = train[mask_train]

        ntrains = ntrains[train]
        psth_window = psth_window[:, train]
        psth_window = np.take_along_axis(psth_window, args[None, :] + np.arange(0, 2000, 1)[:, None], axis=0)
        avg_psth = np.sum(ntrains[None, :] * psth_window, axis=1) / np.sum(ntrains)

        from scipy.optimize import least_squares, minimize
        from actxanalysis.timescales.dgestimation import exp_fixed_lsq_regularized, jac_exp_lsq_regularized

        for tau0 in [20, 40, 80, 160, 320]:
            theta0 = np.array([avg_psth[0], tau0, avg_psth[-100].mean()])
            _res_lsq = minimize(exp_fixed_lsq_regularized, theta0.copy(), method='L-BFGS-B', options=dict(ftol=1e-30, gtol=1e-20, maxiter=40), 
                                tol=1e-30, jac=jac_exp_lsq_regularized, 
                                args=(stim_data.st.dt, np.arange(0, 2000, 1), avg_psth, 0., 0.))
            time = np.arange(0, 2000, 1) * stim_data.st.dt
            A = _res_lsq['x'][0]
            tau = _res_lsq['x'][1]
            offset = _res_lsq['x'][2]

            if _res_lsq['fun'] < best_mse and _res_lsq['x'][0] > 0 and _res_lsq['x'][1] > 0:
                res_lsq = _res_lsq.copy()
                best_mse = _res_lsq['fun']

        return avg_psth, {'A': A, 'tau': tau, 'offset': offset, 'best_mse': best_mse}