import argparse
import torch
from lib import load_config, load_json
import os
from scipy import optimize
from scipy.stats import norm
from math import sqrt
import numpy as np
import lib


# Dual between mu-GDP and (epsilon,delta)-DP
def delta_eps_mu(eps, mu):
    return norm.cdf(-eps / mu +
                    mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


# inverse Dual
def eps_from_mu(mu, delta):

    def f(x):
        return delta_eps_mu(x, mu) - delta

    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


def gdp_mech(
        sample_rate,
        niter,
        noise_multiplier,
        delta
):
    mu = sample_rate * sqrt(niter * (np.exp(1 / (noise_multiplier ** 2)) - 1))
    epsilon = eps_from_mu(mu, delta)
    return epsilon


def eps_from_config(config, info):
    batch_size = config['train_params']['batch_size']
    sample_size = info['train_size']
    epochs = config['train_params']['epochs']
    prob = batch_size / sample_size

    steps_per_epoch = max(1, sample_size // batch_size)
    total_steps = steps_per_epoch * epochs

    epsilon = gdp_mech(
        sample_rate=prob,
        niter=total_steps,
        noise_multiplier=config['dp']['noise_multiplier'],
        delta=config['dp']['delta'],
    )

    return epsilon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar='FILE', default='exp/wilt/tvae/config.toml')
    args, _ = parser.parse_known_args()
    raw_config = load_config(args.config)
    raw_info = load_json(os.path.join(raw_config['real_data_path'], 'info.json'))

    delta = raw_config['dp']['delta']
    eps = eps_from_config(raw_config, raw_info)
    print(f"(epsilon, delta) = ({eps}, {delta})")
