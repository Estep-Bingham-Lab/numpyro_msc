import unittest

from jax import numpy as jnp
from jax import random

import numpyro
from numpyro import distributions as dist 

from numpyro_msc import diagnostics, msc

class TestAll(unittest.TestCase):

    def test_all(self):
        key_1, key_2, key_3 = random.split(random.key(1), 3)
        n_super = 32
        n_within = 32
        
        # A banana-shaped target where NUTS works
        def rosenbrock():
            x = numpyro.sample('x', dist.Normal())
            y = numpyro.sample('y', dist.Normal(0.03*(jnp.square(x)-100)))
            return x,y
        mcmc = msc.many_short_chains(
            rosenbrock, 
            key_1, 
            n_super, 
            n_within,
            improve_init_params={'n_iter': 4}
        )
        self.assertLess(
            diagnostics.max_nested_rhat(mcmc=mcmc, n_super=n_super), 1.03
        )

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
        self.assertGreater(
            diagnostics.max_nested_rhat(mcmc=mcmc, n_super=n_super), 2
        )
        samples = mcmc.get_samples()
        self.assertNotAlmostEqual(0.7, (samples['x']>0).mean(), delta=0.1)

        # check n_super=1 works and diagnostic does not crash
        n_super = 1
        n_within = 2
        mcmc = msc.many_short_chains(
            rosenbrock, 
            key_3, 
            n_super, 
            n_within
        )
        self.assertAlmostEqual(
            diagnostics.max_nested_rhat(mcmc=mcmc, n_super=n_super), 1.0
        )

        # check using a potential function and passing init_params works
        def normal_potential(x):
            return 0.5*jnp.square(x).sum()

        dim = 4
        n_super = 8
        init_key, run_key = random.split(random.key(1),2)
        init_params = random.normal(init_key, (n_super, dim))
        mcmc = msc.many_short_chains(
            None, 
            run_key,
            n_super,
            kernel_params = {'potential_fn': normal_potential},
            init_params=init_params
        )
        self.assertLessEqual(
            diagnostics.max_nested_rhat(mcmc=mcmc, n_super=n_super),
            1.1
        )

if __name__ == '__main__':
    unittest.main()
