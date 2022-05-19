import matplotlib as mpl
import numpy as np
from scipy.special import erf

from actxtimescale.bayes_integration import normal_posterior


def plot_posteriors(axs, df, logtau, gaussian_mean=False, log=True, xlim=None, quantile_err=[0.025, 0.975], observations=True):
    tau = np.exp(logtau)
    for ii, hemisphere in enumerate(['left ACx', 'right ACx']):
        _df = df[df.hemisphere == hemisphere]
        _df = _df[(~_df.logtau_est.isna()) & (~_df.bias_logtau.isna()) & (~_df.var_logtau.isna())]
        
        sd2_population = df['logtau_est'].var()
        posterior = normal_posterior(_df, logtau, obs='logtau_est', bias='bias_logtau', sd2='var_logtau', gaussian_mean=gaussian_mean, 
                                                   sd2_population=sd2_population, prior='uniform')

        logtau_corrected = _df.logtau_corrected.values[None, :]
        cum_ln = normal_cum(logtau[:, None], logtau_corrected, _df.var_logtau.values[None, :]**0.5)
        logtau_q = np.array([logtau[np.searchsorted(cum_ln[:, ii], quantile_err)] for ii in range(cum_ln.shape[1])])
        logtau_error = np.abs(logtau_corrected.T - logtau_q)
            
        tau_corrected = _df.tau_corrected.values[None, :]
        tau_q = np.exp(logtau_q)
        tau_error = np.abs(tau_corrected.T - tau_q)
        
        if not(log):
            posterior = posterior / tau # corresponding change of variables to go from pdf(x) to pdf(log(variable))
            argf_plot = np.searchsorted(logtau, xlim[1])

            y = axs[ii].get_ylim()[1] / 2
            max_post = np.max(posterior)
            y = np.arange(1, len(_df) + 1, 1) / len(_df) * max_post
            axs[ii].plot(tau[:argf_plot], posterior[:argf_plot], linestyle='-', color='C' + str(ii), lw=1.5, alpha=1, label=hemisphere)
            axs[ii].fill_between(tau[:argf_plot], posterior[:argf_plot] * 0, posterior[:argf_plot], color='C' + str(ii), alpha=0.4)
            if observations:
                markers, caps, bars = axs[ii].errorbar(tau_corrected[0], y, xerr=tau_error.T, fmt='o', color='C' + str(ii), capsize=4, alpha=0.5, label='observations')
            
            mean_posterior = np.sum(posterior[:-1] * tau[:-1] * np.diff(tau))
            sd_posterior = np.sum(posterior[:-1] * (tau[:-1] - mean_posterior)**2 * np.diff(tau))**0.5
            cum_posterior = np.cumsum(posterior[:-1] * np.diff(tau))
#             print(np.searchsorted(cum_posterior, [0.15865, 0.84135]))
#             print('quantiles 0.15865, 0.84135', tau[np.searchsorted(cum_posterior, [0.15865, 0.84135])])
            print('quantiles', quantile_err, tau[np.searchsorted(cum_posterior, quantile_err)])
            print('mean, sd', mean_posterior, sd_posterior)
#             print('mean, sd', mean_posterior, sd_posterior)
            
        else:
            y = axs[ii].get_ylim()[1] / 2
            max_post = np.max(posterior)
            y = np.arange(1, len(_df) + 1, 1) / len(_df) * max_post
            axs[ii].plot(logtau, posterior, color='C' + str(ii), lw=1.5, alpha=1, label=hemisphere)
            axs[ii].fill_between(logtau, posterior * 0, posterior, color='C' + str(ii), alpha=0.4)
            if observations:
                markers, caps, bars = axs[ii].errorbar(logtau_corrected[0], y, xerr=logtau_error.T, fmt='o', color='C' + str(ii), capsize=4, alpha=0.5, label='observations')


    axs[0].tick_params(axis='both', which='both', bottom=True, labelbottom=False)

#     for ax in axs:
#         y0, yf = ax.get_ylim()
#         ax.set_ylim(y0, yf * 1.3)

    if xlim is None and not(log):
        xlim = (0, 350)
    elif xlim is None:
        xlim = (0, 10)
        
    axs[1].set_xlim(xlim)
    
    axs[0].legend()
    axs[1].legend()


def normal_cum(x, mu, sd):
    return (1 + erf((x - mu) / sd / 2**.5 ) ) / 2


def myerrorbar(x=None, y=None, hue=None, xerr=None, yerr=None, palette=None, data=None, ax=None, shift_hue=0, alpha=1., legend=True, **kwargs):

    hue_vals = data[hue].unique()
    palette = palette if palette is not None else {hue_val: 'C' + str(i) for i, hue_val in enumerate(hue_vals)}

    for i, hue_val in enumerate(hue_vals):
        data_hue = data[data[hue] == hue_val]
        x_data = data_hue[x]
        y_data = data_hue[y]
        
        xerr_data = None if xerr is None else data_hue[xerr]
            
        if yerr is not None:
            yerr_data = data_hue[yerr]
            
        label = hue_val if legend else None
        if isinstance(alpha, float):
            markers, caps, bars = ax.errorbar(x_data + i * shift_hue, y_data, xerr=xerr_data, yerr=yerr_data, color=palette[hue_val], label=label, 
                                              alpha=alpha, **kwargs)
        else:
            alpha_data = data_hue[alpha]
            ax.errorbar(x_data[0] + i * shift_hue, y_data[0], xerr=xerr_data[0], yerr=yerr_data[0], label=label, color=palette[hue_val], alpha=alpha_data[0], **kwargs)
            for _x, _y, _xerr, _yerr, _alpha in zip(x_data[1:], y_data[1:], xerr_data[1:], yerr_data[1:], alpha_data[1:]):
                ax.errorbar(_x + i * shift_hue, _y, xerr=_xerr, yerr=_yerr, color=palette[hue_val], alpha=_alpha, **kwargs)
        
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if legend:
        ax.legend()
    return ax


def set_labels(ax, xlabel=None, ylabel=None, title=None, fontsize=None):
    fontsize = fontsize if fontsize is not None else mpl.rcParams['axes.labelsize']
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    