"""
evt_utils.py
------------
Extreme Value Theory utilities for tail-risk estimation.

Implements the Peaks-Over-Threshold (POT) method with Generalized Pareto
Distribution (GPD) fitting, and CVaR estimation following the approach
used in Davar, Godin & Garrido (2024) — POTPG.

Extended here for use in a multi-agent setting where both individual
and systemic CVaR terms are estimated from real historical shock data.
"""

import numpy as np
from scipy import stats


def fit_gpd(losses: np.ndarray, quantile: float = 0.95):
    """
    Fit a Generalized Pareto Distribution to exceedances above a threshold
    using the Peaks-Over-Threshold method.

    Parameters
    ----------
    losses : array-like
        Daily loss values (positive = loss, i.e. negative returns).
    quantile : float
        Quantile used to set the threshold u (default: 0.95).

    Returns
    -------
    dict with keys:
        xi    : tail index (>0 = fat-tailed, 0 = exponential, <0 = bounded)
        sigma : scale parameter
        u     : threshold
        Fbar_u: fraction of observations exceeding the threshold
    """
    losses = np.asarray(losses)
    u = np.quantile(losses, quantile)
    exceedances = losses[losses > u] - u
    xi, loc, sigma = stats.genpareto.fit(exceedances, floc=0)
    Fbar_u = (losses > u).mean()
    return {"xi": xi, "sigma": sigma, "u": u, "Fbar_u": Fbar_u}


def cvar_from_gpd(gpd_params: dict, alpha: float = 0.95) -> float:
    """
    Compute CVaR (Conditional Value-at-Risk / Expected Shortfall) at level
    alpha using closed-form GPD tail extrapolation.

    This avoids the instability of empirical CVaR at very high confidence
    levels where data is scarce — the key motivation from POTPG.

    Parameters
    ----------
    gpd_params : dict
        Output of fit_gpd().
    alpha : float
        Confidence level (default: 0.95).

    Returns
    -------
    float : CVaR estimate
    """
    xi = gpd_params["xi"]
    sigma = gpd_params["sigma"]
    u = gpd_params["u"]
    Fbar_u = gpd_params["Fbar_u"]

    # VaR via GPD quantile function
    var = u + (sigma / xi) * (((1 - alpha) / Fbar_u) ** (-xi) - 1)

    # CVaR closed form for GPD tail
    cvar = (var + sigma - xi * u) / (1 - xi)
    return float(cvar)


def compute_fragility_index(
    loss_fn,
    intensities: list,
    n_trials: int = 25,
    rng: np.random.Generator = None
) -> tuple:
    """
    Estimate the fragility index as the mean second derivative of loss
    with respect to shock intensity, with bootstrap 95% CI.

    Positive = Fragile (loss accelerates with shock intensity)
    Near zero = Robust
    Negative = Antifragile (loss decelerates / system benefits from shocks)

    Parameters
    ----------
    loss_fn : callable
        Function(intensity) -> mean loss for a given shock multiplier.
    intensities : list of float
        Shock intensity multipliers to sweep over.
    n_trials : int
        Number of bootstrap resamples for CI.
    rng : np.random.Generator
        Random number generator (optional).

    Returns
    -------
    (mean_fragility, ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    trial_losses = {m: [] for m in intensities}
    for m in intensities:
        for _ in range(n_trials):
            trial_losses[m].append(loss_fn(m))

    mean_losses = [np.mean(trial_losses[m]) for m in intensities]
    d2 = np.gradient(np.gradient(mean_losses, intensities), intensities)
    mean_frag = float(np.mean(d2))

    # Bootstrap CI
    boot_frags = []
    for _ in range(500):
        resampled = [np.mean(rng.choice(trial_losses[m], size=n_trials, replace=True))
                     for m in intensities]
        d2_boot = np.gradient(np.gradient(resampled, intensities), intensities)
        boot_frags.append(np.mean(d2_boot))

    ci_low, ci_high = np.percentile(boot_frags, [2.5, 97.5])
    return mean_frag, float(ci_low), float(ci_high)


def classify_fragility(fragility_index: float, threshold: float = 0.05) -> str:
    """
    Classify a fragility index value.

    Parameters
    ----------
    fragility_index : float
    threshold : float
        Boundary for Robust classification (default: 0.05).

    Returns
    -------
    str : 'Fragile', 'Robust', or 'Antifragile'
    """
    if fragility_index > threshold:
        return "Fragile"
    elif fragility_index < -threshold:
        return "Antifragile"
    return "Robust"
