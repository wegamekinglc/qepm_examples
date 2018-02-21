# -*- coding: utf-8 -*-
"""
Created on 2018-2-21

@author: cheng.li
"""

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

f = np.array([0.1, 0.0, -0.1]).reshape((-1, 1))
vols = np.array([0.3, 0.3, 0.3])
C = np.array([1., 0.5, 0.5, 0.5, 1., 0.5, 0.5, 0.5, 1.]).reshape((-1, 3))

sigma = np.diag(vols) @ C @ np.diag(vols)


def full_invest(sigma, f, lamb):
    i = np.ones_like(f)
    sigma_inv = np.linalg.inv(sigma)
    s1 = i.T @ sigma_inv @ i
    s2 = i.T @ sigma_inv @ f
    w = sigma_inv @ i / s1 + (s1 * sigma_inv @ f - s2 * sigma_inv @ i) / lamb / s1
    mu = w.T @ f
    vol = math.sqrt(w.T @ sigma @ w)
    return mu[0, 0], vol


def long_short(sigma, f, lamb):
    i = np.ones_like(f)
    sigma_inv = np.linalg.inv(sigma)
    s1 = i.T @ sigma_inv @ i
    s2 = i.T @ sigma_inv @ f
    w = (s1 * sigma_inv @ f - s2 * sigma_inv @ i) / lamb / s1
    mu = w.T @ f
    vol = math.sqrt(w.T @ sigma @ w)
    return mu[0, 0], vol


lambs = np.exp(np.linspace(math.log(100), math.log(1.), 100))
df_values = np.zeros((len(lambs), 4))

for i, lamb in enumerate(lambs):
    mu, vol = full_invest(sigma, f, lamb)
    df_values[i, 0] = mu
    df_values[i, 1] = vol

    mu, vol = long_short(sigma, f, lamb)
    df_values[i, 2] = mu
    df_values[i, 3] = vol

df = pd.DataFrame(df_values, columns=['mu', 'full_invest', 'mu2', 'long_short'])
fig = plt.figure(figsize=(12, 6))
plt.plot(df['full_invest'], df['mu'])
plt.plot(df['long_short'], df['mu2'])
plt.legend(labels=['full_invest', 'long_short'])
plt.xlabel('$\sigma$')
plt.ylabel('$\mu$')
plt.show()
