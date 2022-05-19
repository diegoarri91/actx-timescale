import numpy as np


def normal_loglikelihoods_shared_mean(df, mu, obs=None, bias=None, sd2=None):

    observations = df[obs].values[None, :]
    bias = df[bias].values[None, :]
    sd2 = df[sd2].values[None, :]

    log_likelihood = -(observations - bias - mu[:, None])**2 / (2 * sd2) - np.log(np.sqrt(2 * np.pi * sd2))
    log_likelihood = np.sum(log_likelihood, 1)

    return log_likelihood


def normal_posterior(df, mu, obs=None, bias=None, sd2=None, gaussian_mean=False, sd2_population=None, prior='uniform', prior_pars=None):

    dmu = mu[1] - mu[0]
    if not(gaussian_mean):
        log_likelihood = normal_loglikelihoods_shared_mean(df, mu, obs=obs, bias=bias, sd2=sd2)
    else:
        log_likelihood = normal_loglikelihoods_gaussian_mean(df, mu, obs=obs, bias=bias, sd2=sd2, sd2_population=sd2_population)
        
    if prior == 'uniform':
        log_posterior = log_likelihood - np.max(log_likelihood)
    
    posterior = np.exp(log_posterior)
    posterior = posterior / np.sum(posterior * dmu)
    
    return posterior


def lognormal_cum(x, mu, sd2):
    return (1 + erf((np.log(x) - mu) / sd2**0.5 / 2**.5 )) / 2


def normal_loglikelihoods_gaussian_mean(df, mu, obs=None, bias=None, sd2=None, sd2_population=None):

    observations = df[obs].values[None, :]
    bias = df[bias].values[None, :]
    sd2 = df[sd2].values[None, :]

    log_likelihood = -(observations - bias - mu[:, None])**2 / (2 * (sd2 + sd2_population)) - np.log(np.sqrt(2 * np.pi * sd2 + sd2_population))
    log_likelihood = np.sum(log_likelihood, 1)

    return log_likelihood


def log_marginal_likelihood(df, x, gaussian_mean=False):
    l = []
    dx = x[1] - x[0]
    for ii, hemisphere in enumerate(['left ACx', 'right ACx']):
        _df = df[df.hemisphere == hemisphere]
        _df = _df[(~_df.logtau_est.isna()) & (~_df.bias_logtau.isna()) & (~_df.var_logtau.isna())]

        sd2_population = df['logtau_est'].var()
        posterior = normal_posterior(_df, x, obs='logtau_est', bias='bias_logtau', sd2='var_logtau', gaussian_mean=gaussian_mean, 
                                                   sd2_population=sd2_population, prior='uniform')

        if not(gaussian_mean):
            likelihoods = np.exp(normal_loglikelihoods_shared_mean(_df, x, obs='logtau_est', bias='bias_logtau', sd2='var_logtau'))
        else:
            likelihoods = np.exp(normal_loglikelihoods_gaussian_mean(_df, x, obs='logtau_est', bias='bias_logtau', sd2='var_logtau', 
                                                                     sd2_population=sd2_population))
        
        l.append(np.sum(likelihoods * posterior * dx))
        
    lml = np.sum(np.log10(np.array(l)))
    
    return lml


def log_marginal_likelihood_single_model(df, x, gaussian_mean=False):
    l = []
    dx = x[1] - x[0]

    df = df[(~df.logtau_est.isna()) & (~df.bias_logtau.isna()) & (~df.var_logtau.isna())]

    sd2_population = df['logtau_est'].var()
    posterior = normal_posterior(df, x, obs='logtau_est', bias='bias_logtau', sd2='var_logtau', gaussian_mean=gaussian_mean, 
                                                   sd2_population=sd2_population, prior='uniform')

    if not(gaussian_mean):
        likelihoods = np.exp(normal_loglikelihoods_shared_mean(df, x, obs='logtau_est', bias='bias_logtau', sd2='var_logtau'))
    else:
        likelihoods = np.exp(normal_loglikelihoods_gaussian_mean(df, x, obs='logtau_est', bias='bias_logtau', sd2='var_logtau', 
                                                                     sd2_population=sd2_population))
        
    ml = np.sum(likelihoods * posterior * dx)
    lml = np.log10(ml)
    
    return lml
