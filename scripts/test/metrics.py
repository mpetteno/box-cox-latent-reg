import logging

import numpy as np
import scipy
import sklearn


def compute_gaussian_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Fréchet Inception Distance (FID) between two Gaussian distributions."""

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        logging.info(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    assert np.isfinite(covmean).all() and not np.iscomplexobj(covmean)

    tr_covmean = np.trace(covmean)

    frechet_dist = diff.dot(diff)
    frechet_dist += np.trace(sigma1) + np.trace(sigma2)
    frechet_dist -= 2 * tr_covmean
    return frechet_dist


def compute_mmd_rbf(x, y, gamma=None):
    """Compute Maximum Mean Discrepancy (MMD) between two sets of samples with an RBF kernel."""
    k_xx = sklearn.metrics.pairwise.rbf_kernel(x, x, gamma=gamma)
    k_yy = sklearn.metrics.pairwise.rbf_kernel(y, y, gamma=gamma)
    k_xy = sklearn.metrics.pairwise.rbf_kernel(x, y, gamma=gamma)
    mmd = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    return mmd


def compute_mmd_polynomial(x, y, gamma=None, degree=3, coef0=1):
    """Compute Maximum Mean Discrepancy (MMD) between two sets of samples with a polynomial kernel."""
    k_xx = sklearn.metrics.pairwise.polynomial_kernel(x, x, degree, gamma, coef0)
    k_yy = sklearn.metrics.pairwise.polynomial_kernel(y, y, degree, gamma, coef0)
    k_xy = sklearn.metrics.pairwise.polynomial_kernel(x, y, degree, gamma, coef0)
    mmd = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    return mmd
