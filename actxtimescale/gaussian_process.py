import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded

from actxtimescale.utils import band_matrix, diag_indices, get_dt, unband_matrix


class GaussianProcess:

    def __init__(self, mu=None, autocov=None):
        self.mu = mu
        self.autocov = autocov
        
        self.cov = None
        self.inv_cov = None
        self.inv_cov_banded = None
        self.ch_lower = None
        
    def set_cov(self, t):
        
        cov = np.zeros((len(t), len(t)))
        for v in range(len(t)):
            argf = min(len(self.autocov) + v, len(t))
            cov[v, v:argf] = self.autocov[:min(len(self.autocov), len(t) - v)]

        cov[np.tril_indices(cov.shape[0], k=-1)] = cov.T[np.tril_indices(cov.shape[0], k=-1)]
        
        self.cov = cov
        
        return self
    
    def set_cholesky(self, t):
        
        dt = get_dt(t)
        max_band = min(len(self.autocov), len(t))
        
        cov = np.zeros((max_band, len(t)))
        for v in range(max_band):
            cov[v, :len(t) - v] = self.autocov[v]

        ch = cholesky_banded(cov, lower=True)
        self.ch_lower = unband_matrix(ch, symmetric=False, lower=True)
        
        return self

    def set_t(self, t, inv_cov=True, cholesky=True):

        dt = get_dt(t)
        max_band = min(len(self.autocov), len(t))
        
        cov = np.zeros((max_band, len(t)))
        for v in range(max_band):
            cov[v, :len(t) - v] = self.autocov[v]

#         if inv_cov:
#             self.inv_cov = solveh_banded(cov, np.eye(len(t)), lower=True)
#             self.inv_cov_banded = band_matrix(self.inv_cov, max_band=max_band)
#             max_band_inv_cov = np.where(np.all(np.abs(self.inv_cov_banded) < eps_max_band_inv_cov, axis=1))[0][0]
#             self.inv_cov_banded = self.inv_cov_banded[:max_band_inv_cov, :]

        if cholesky:
            ch = cholesky_banded(cov, lower=True)
            self.ch_lower = unband_matrix(ch, symmetric=False, lower=True)

        return self

    def sample(self, t=None, shape=(1,), seed=None, cholesky=True):
        
        np.random.seed(seed)
        
        if cholesky:
            rand = np.random.randn(self.ch_lower.shape[0], *shape)
            xi = np.einsum('st,t...->s...', self.ch_lower, rand) + self.mu
        else:
            xi = np.random.multivariate_normal(self.mu + np.zeros(self.cov.shape[0]), self.cov, size=shape)
            xi = np.moveaxis(xi, -1, 0)
            
        return xi
