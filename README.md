[![Build Status](https://github.com/Estep-Bingham-Lab/numpyro_msc/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Estep-Bingham-Lab/numpyro_msc/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Estep-Bingham-Lab/numpyro_msc/branch/main/graph/badge.svg)](https://codecov.io/gh/Estep-Bingham-Lab/numpyro_msc)

# `numpyro_msc`

Many short chains sampling with NumPyro à la [Margossian et al. (2024)](https://doi.org/10.1214/24-BA1453).

## Installation

Using pip
```bash
pip install numpyro_msc @ git+https://github.com/Estep-Bingham-Lab/numpyro_msc.git
```

## Example

```python
from jax import random
from jax import numpy as jnp

import numpyro
from numpyro import distributions as dist 

from numpyro_msc import msc, diagnostics

# A banana-shaped target where NUTS works
def rosenbrock():
    x = numpyro.sample('x', dist.Normal())
    y = numpyro.sample('y', dist.Normal(0.03*(jnp.square(x)-100)))
    return x,y

rng_key = random.key(1)
n_super = 64
n_within = 64
mcmc = msc.many_short_chains(rosenbrock, rng_key, n_super, n_within)
```
```
sample: 100%|█████████████████████████████| 2048/2048 [00:10<00:00, 203.85it/s]
```
```python
print(diagnostics.nested_rhats(mcmc=mcmc, n_super=n_super))
```
```
{'x': Array(1.0045928, dtype=float32), 'y': Array(1.0074906, dtype=float32)}
```

## Tips for working with dynamic samplers

In order to avoid performance degradation when `vmap`ing samplers with wildly 
varying step duration---as is the case of NUTS---a simple fix is to reduce the 
space for adaptation. For example, we use a lower `max_tree_depth` of 8 in the 
default settings. This greatly reduces the range of possible step durations,
thereby improving performance. As a rule-of-thumb, set parameters such as
the `max_tree_depth` to the lowest possible value that still produces accurate
samples.

For an in-depth discussion of this issue and an interesting (but considerably 
more complex) solution, 
[see this article](https://openreview.net/forum?id=Mlmpf4Izrj).

## References

Margossian, C. C., Hoffman, M. D., Sountsov, P., Riou-Durand, L., 
Vehtari, A., & Gelman, A. (2024). [Nested ̂R: Assessing the convergence 
of Markov chain Monte Carlo when running many short 
chains.](https://doi.org/10.1214/24-BA1453) *Bayesian Analysis*, 1(1), 1-28
