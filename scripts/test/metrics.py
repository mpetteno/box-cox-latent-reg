import numpy as np
from scipy import stats
from scipy.integrate import simpson
from scipy.spatial.distance import pdist, squareform


def compute_mmd(x, y, kernel='rbf', **kwargs):
    """
    Compute Maximum Mean Discrepancy (MMD) between two samples.

    Args:
        x (array-like): Samples from first distribution
        y (array-like): Samples from second distribution
        kernel (str): Kernel type ('rbf', 'linear', 'polynomial')
        **kwargs: Additional kernel parameters:
            - gamma (float): RBF kernel bandwidth parameter (for 'rbf')
            - degree (int): Polynomial degree (for 'polynomial')

    Returns:
        float: MMD value
    """
    x = np.atleast_2d(x).T  # Ensure column vector
    y = np.atleast_2d(y).T  # Ensure column vector

    # Default parameters
    gamma = kwargs.get('gamma', 1.0)
    degree = kwargs.get('degree', 3)

    # Compute kernel matrices
    if kernel == 'rbf':
        # RBF kernel: k(x,y) = exp(-gamma * ||x-y||^2)
        def rbf_kernel(X):
            pairwise_dists = squareform(pdist(X, 'sqeuclidean'))
            return np.exp(-gamma * pairwise_dists)

        K_xx = rbf_kernel(x)
        K_yy = rbf_kernel(y)
        K_xy = np.exp(-gamma * np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, axis=2))

    elif kernel == 'linear':
        # Linear kernel: k(x,y) = x^T y
        K_xx = np.dot(x, x.T)
        K_yy = np.dot(y, y.T)
        K_xy = np.dot(x, y.T)

    elif kernel == 'polynomial':
        # Polynomial kernel: k(x,y) = (gamma*x^T y + 1)^degree
        K_xx = (gamma * np.dot(x, x.T) + 1) ** degree
        K_yy = (gamma * np.dot(y, y.T) + 1) ** degree
        K_xy = (gamma * np.dot(x, y.T) + 1) ** degree

    else:
        raise ValueError(f"Kernel '{kernel}' not supported")

    # Compute MMD
    n_x = x.shape[0]
    n_y = y.shape[0]

    # Biased MMD estimator
    mmd = (np.sum(K_xx) / (n_x * n_x) +
           np.sum(K_yy) / (n_y * n_y) -
           2 * np.sum(K_xy) / (n_x * n_y))

    return mmd


def compute_distribution_metrics(samples, density_estimation_method='kde', **kwargs):
    """
    Compute KL divergence, Hellinger distance, distribution overlap, and MMD
    between an empirical distribution and a standard Gaussian.

    Args:
        samples (array-like): Samples from the unknown distribution
        density_estimation_method (str): Estimation method ('kde' or 'histogram')
        **kwargs: Additional parameters for the method

    Returns:
        dict: Dictionary with all metrics
    """
    # Determine range and evaluation points
    x_min = min(np.min(samples) - 3, -6)
    x_max = max(np.max(samples) + 3, 6)
    num_points = kwargs.get('num_points', 1000)
    x = np.linspace(x_min, x_max, num_points)

    # Standard Gaussian PDF
    q_x = stats.norm.pdf(x, 0, 1)

    # Estimate density
    if density_estimation_method == 'kde':
        bandwidth = kwargs.get('bandwidth', None)
        kde = stats.gaussian_kde(samples, bw_method=bandwidth)
        p_x = kde(x)
    elif density_estimation_method == 'histogram':
        bins = kwargs.get('bins', 'auto')
        hist, bin_edges = np.histogram(samples, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        p_x = np.interp(x, bin_centers, hist, left=0, right=0)
    else:
        raise ValueError(f"Method '{density_estimation_method}' not supported")

    # Ensure numerical stability
    p_x = np.maximum(p_x, 1e-10)

    # KL Divergence: ∫p(x)log(p(x)/q(x))dx
    kl_integrand = p_x * np.log(p_x / q_x)
    kl_divergence = simpson(kl_integrand, x)

    # Hellinger distance: √(1-∫√(p(x)q(x))dx)
    bc_integrand = np.sqrt(p_x * q_x)
    bc_coefficient = simpson(bc_integrand, x)
    hellinger_distance = np.sqrt(max(0, 1 - bc_coefficient))

    # Overlap coefficient: ∫min(p(x),q(x))dx
    overlap_integrand = np.minimum(p_x, q_x)
    overlap = simpson(overlap_integrand, x)

    # Total variation distance: 0.5*∫|p(x)-q(x)|dx
    tvd_integrand = 0.5 * np.abs(p_x - q_x)
    total_variation = simpson(tvd_integrand, x)

    # Jensen-Shannon divergence: 0.5*[KL(P||M) + KL(Q||M)] where M=(P+Q)/2
    m_x = (p_x + q_x) / 2
    js_div = 0.5 * (
            simpson(p_x * np.log(p_x / m_x), x) +
            simpson(q_x * np.log(q_x / m_x), x)
    )

    # Wasserstein distance
    # For 1D distributions with CDFs F and G: ∫|F(x)-G(x)|dx
    # Approximate using numerical integration of the absolute difference of CDFs
    p_cdf = np.cumsum(p_x) * (x[1] - x[0])
    p_cdf /= p_cdf[-1]  # Normalize
    q_cdf = stats.norm.cdf(x, 0, 1)
    wasserstein = simpson(np.abs(p_cdf - q_cdf), x)

    # Compute MMD
    # Generate standard Gaussian samples for comparison
    np.random.seed(42)  # For reproducibility
    n_samples = len(samples)

    # Sample from standard Gaussian
    std_gaussian_samples = np.random.normal(0, 1, size=n_samples)

    # Compute MMD with different kernels
    mmd_rbf = compute_mmd(samples, std_gaussian_samples, kernel='rbf', gamma=0.5)
    mmd_linear = compute_mmd(samples, std_gaussian_samples, kernel='linear')
    mmd_poly = compute_mmd(samples, std_gaussian_samples, kernel='polynomial', degree=3)

    return {
        'kl_divergence': kl_divergence,
        'hellinger_distance': hellinger_distance,
        'overlap_coefficient': overlap,
        'total_variation': total_variation,
        'jensen_shannon': js_div,
        'wasserstein': wasserstein,
        'mmd_rbf': mmd_rbf,
        'mmd_linear': mmd_linear,
        'mmd_polynomial': mmd_poly,
        'x': x,
        'p_x': p_x,
        'q_x': q_x
    }
