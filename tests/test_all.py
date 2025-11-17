import unittest
from functools import partial
import math

import jax
from jax import lax
from jax import numpy as jnp
from jax import random

import numpyro
from numpyro import distributions as dist 
from numpyro.infer import MCMC, NUTS

from numpyro_msc import msc, nested_rhat

class TestAll(unittest.TestCase):

    def test_all(self):
        key_1, key_2 = random.split(random.key(1))
        n_super = 64
        n_within = 64
        
        # A banana-shaped target where NUTS works
        def rosenbrock():
            x = numpyro.sample('x', dist.Normal())
            y = numpyro.sample('y', dist.Normal(0.03*(jnp.square(x)-100)))
            return x,y
        mcmc = msc.many_short_chains(rosenbrock, key_1, n_super, n_within)
        assert nested_rhat.max_nested_rhat(mcmc=mcmc, n_super=n_super) < 1.01

        # well separated modes => parallel indep chains should fail => diagnostic
        # should pick this up
        def mixture():
            return numpyro.sample(
                'x', 
                dist.Mixture(
                    dist.Categorical(jnp.array([0.3,0.7])),
                    dist.Normal(jnp.array([-2.,2.]), jnp.array([0.1,0.1]))
                )
            )
        mcmc = msc.many_short_chains(
            mixture, key_2, n_super, n_within, keep_last_step_only=False
        )
        self.assertGreater(nested_rhat.max_nested_rhat(mcmc=mcmc, n_super=n_super), 2)
        samples = mcmc.get_samples()
        self.assertNotAlmostEqual(0.7, (samples['x']>0).mean(), delta=0.1)

if __name__ == '__main__':
    unittest.main()
