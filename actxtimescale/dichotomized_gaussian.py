import numpy as np
from scipy.special import erfinv
from scipy.stats import multivariate_normal

from actxtimescale.gaussian_process import GaussianProcess
from actxtimescale.utils import get_dt


class DichotomizedGaussian:
    """Sample stationary spike trains using the Dichotomized Gaussian (Macke et al 2009).

    Attributes:
        lam (str): Baseline firing rate
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, lam=0, raw_autocorrelation=1):
        self.lam = lam
        self.raw_autocorrelation = raw_autocorrelation
        
        self.drho = None
        self.max_error = None

    def set_t(self, t, drho=1e-3):
#         self.t = t
        dt = get_dt(t)

        p = self.lam * dt
        cov0 = 1
        mu = np.sqrt(2 * cov0) * erfinv(2 * p - 1)
        
        rho_gauss = np.arange(-1 + drho, 1, drho)
        rho_dg = []
        for _rho_gauss in rho_gauss:
            cov_gauss = np.array([[cov0, cov0 * _rho_gauss], [cov0 * _rho_gauss, cov0]])
            rho_dg.append(1 + multivariate_normal.cdf([0, 0], mean=np.ones(2) * mu, cov=cov_gauss) - \
                          2 * multivariate_normal.cdf([0], mean=np.ones(1) * mu, cov=np.array([cov0])))
        rho_dg = np.array(rho_dg)
        
        self.drho = drho
        idx = np.argmin((self.raw_autocorrelation[:, None] - rho_dg[None, :])**2, 1) 
        self.max_error = np.max((self.raw_autocorrelation - rho_dg[idx])**2)
        autocov = cov0 * rho_gauss[idx]
        autocov[0] = cov0
        
        self.gp = GaussianProcess(mu=mu, autocov=autocov)
        
        try:
            self.gp.set_cholesky(t)
        except(np.linalg.LinAlgError):
            self.gp.set_cov(t)
        
        return self
    
    def sample(self, t=None, shape=(1,), seed=None, cholesky=True):
        gp_samples = self.gp.sample(shape=shape, seed=seed, cholesky=cholesky)
        mask_spikes = gp_samples > 0
        return mask_spikes
    
    def sample2(self, t, shape=(1,), seed=None):
        """Sample spike trains.

        Args:
            t: 1d-array of time points
            shape: Output is a mask x with x.shape = (len(t),) + shape
            seed: sets numpy seed

        Returns:
            Boolean mask of spikes

        """
        np.random.seed(seed)
        
        dt = get_dt(t)
        
        p = self.lam * dt
        mu = np.sqrt(2) * erfinv(2 * p - 1)
        print(p, mu)
#         var = p * (1 - p)
        var = 1
        
        gaussian_samples = np.random.multivariate_normal(np.ones(len(t)) * mu, np.eye(len(t)) * var, size=shape).T
        print(gaussian_samples[:, 0])
        
        mask_spikes = gaussian_samples > 0
            
        return mask_spikes
