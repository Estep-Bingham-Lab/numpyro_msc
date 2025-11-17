# `numpyro_msc`

Many short chains sampling with NumPyro à la Margossian et al. (2024).

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

from numpyro_msc import msc, nested_rhat

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
print(nested_rhat.nested_rhats(mcmc=mcmc, n_super=n_super))
```
```
{'x': Array(1.0045928, dtype=float32), 'y': Array(1.0074906, dtype=float32)}
```

## References

Margossian, C. C., Hoffman, M. D., Sountsov, P., Riou-Durand, L., 
Vehtari, A., & Gelman, A. (2024). Nested ̂R: Assessing the convergence 
of Markov chain Monte Carlo when running many short chains. *Bayesian 
Analysis*, 1(1), 1-28
