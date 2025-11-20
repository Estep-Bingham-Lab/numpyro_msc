from functools import partial

import jax
from jax import lax
from jax import numpy as jnp
from jax import random

import numpyro
from numpyro.infer import MCMC, NUTS

from numpyro_msc import utils

##############################################################################
# main function
##############################################################################

def many_short_chains(
        model,
        rng_key,
        n_super = 8,
        n_within = 8,
        n_adapt = 2**10,
        n_steps = 2**10,
        kernel_class = NUTS,
        kernel_params = {'max_tree_depth': 8},
        model_args = (),
        model_kwargs = {},
        mcmc_kwargs = {},
        run_kwargs = {},
        keep_last_step_only = True,
        improve_init_params = False
    ):
    """
    Many short chains sampling as described in [Ref]_.

    Vectorized sampling of `n_super` superchains, each with `n_within` chains.

    :param model: Target NumPyro model.
    :param rng_key: The PRNG key that the sampler should use for simulation.
    :param n_super: Number of superchains (`K` in [Ref]_).
    :param n_within: Number of chains within each superchain (`M` in [Ref]_).
    :param n_adapt: Number of adaptation steps.
    :param n_steps: Number of sampling steps.
    :param kernel_class: A constructor for the MCMC kernel.
    :param kernel_params: Optional parameters for the MCMC kernel.
    :param model_args: Optional `args` for the model.
    :param model_kwargs: Optional `kwargs` for the model.
    :param mcmc_kwargs: Optional `kwargs` for building the MCMC object.
    :param run_kwargs: Optional `kwargs` for building the `MCMC.run` function.
    :param keep_last_step_only: If `True`, only the last step of the sampling
        phase.
    :param improve_init_params: If `True`, run L-BFGS to improve each of the
        `n_super` initial points. It can also be a `dict` with settings passed
        to :func:`utils.optimize_fun`.
    :return: A :class:`numpyro.infer.MCMC` object.
    
    .. rubric:: References

    .. [Ref] Margossian, C. C., Hoffman, M. D., Sountsov, P., Riou-Durand, L., 
        Vehtari, A., & Gelman, A. (2024). Nested ̂R: Assessing the convergence 
        of Markov chain Monte Carlo when running many short chains. *Bayesian 
        Analysis*, 1(1), 1-28. 
    """
    init_key, run_key = random.split(rng_key)

    # maybe update mcmc_kwargs
    if keep_last_step_only:
        mcmc_kwargs = {'thinning': n_steps, **mcmc_kwargs}

    # Find initial points for superchains
    init_params = sample_n_super_init_params(
        kernel_class(model, **kernel_params), 
        n_super,
        n_within,
        init_key,
        model_args,
        model_kwargs,
        run_kwargs,
        improve_init_params
    )

    # Build kernel
    kernel = kernel_class(model, **kernel_params)

    # sample
    # Each chain has its own sets of kernel parameters. It may violate some
    # simplifying assumptions in the mathematical analysis, but all
    # the kernels should produce convergent chains. This is discussed in
    # the 2nd paragraph of Section 2 of the paper.
    mcmc = run(
        kernel, 
        run_key,
        n_adapt, 
        n_steps, 
        init_params,
        model_args,
        model_kwargs,
        mcmc_kwargs,
        run_kwargs
    )  

    return mcmc

##############################################################################
# low level utilities
##############################################################################

def improve_initial_params(
        model,
        super_init_params,
        rng_key,
        model_args,
        model_kwargs,
        opt_params
    ):
    params_info, potential_fn_gen = numpyro.infer.util.initialize_model(
        rng_key,
        model,
        dynamic_args=True,
        model_args=model_args,
        model_kwargs=model_kwargs,
    )[:2]
    dummy_init_params = params_info[0]
    potential_fn = potential_fn_gen(*model_args, **model_kwargs)
    opt_fun = partial(utils.optimize_fun, potential_fn, **opt_params)

    # find n_super and use it to determine if vectorization is needed
    first_param_shape = next(iter(super_init_params.values())).shape
    if first_param_shape == next(iter(dummy_init_params.values())).shape:
        n_super = 1
        maybe_vmap_pot_fn = potential_fn
        maybe_vmap_opt_fun = opt_fun
    else:
        n_super = first_param_shape[0]
        maybe_vmap_pot_fn = jax.vmap(potential_fn)
        maybe_vmap_opt_fun = jax.vmap(opt_fun)
    
    # print info
    print(f"Improving {n_super} random initial points via L-BFGS optimization")
    print("Starting energies")
    print(maybe_vmap_pot_fn(super_init_params))
    
    # optimize
    opt_super_init_params = maybe_vmap_opt_fun(super_init_params)

    # print info and return
    print(f"Final energies:")
    print(maybe_vmap_pot_fn(opt_super_init_params))
    return opt_super_init_params

def sample_n_super_init_params(
        kernel, 
        n_super,
        n_within,
        rng_key,
        model_args,
        model_kwargs,
        run_kwargs,
        solver_settings,
    ):
    """
    Sample `n_super` initial points, intended to be shared among all `n_within`
    chains inside a super chain.
    """
    rng_key, init_key = random.split(rng_key)
    init_keys = random.split(init_key, n_super) if n_super > 1 else rng_key # avoid creating a singleton dimension
    mcmc = MCMC(
        kernel,
        num_chains=n_super,
        num_warmup=0, 
        num_samples=1, 
        chain_method="vectorized",
        progress_bar=False
    )
    mcmc.run(init_keys, *model_args, **(run_kwargs | model_kwargs))
    super_init_params = getattr(mcmc.last_state, kernel.sample_field)
    
    # improve parameters using L-BFGS optimization
    if solver_settings:
        super_init_params = improve_initial_params(
            kernel.model,
            super_init_params,
            rng_key,
            model_args,
            model_kwargs,
            solver_settings if isinstance(solver_settings, dict) else {}
        )

    # repeat each super chain param `n_within` times, for a total of
    #   n_chains = n_super * n_within
    # initial points
    return jax.tree.map(
        lambda x: jnp.repeat(
            x if n_super>1 else lax.expand_dims(x, (0,)),
            n_within,
            axis=0
        ),
        super_init_params
    )

def run(
        kernel, 
        rng_key,
        num_warmup, 
        num_samples, 
        init_params,
        model_args,
        model_kwargs,
        mcmc_kwargs,
        run_kwargs
    ):
    """
    Sample `n_super` super chains from different `init_params`. Each chain
    inside a super chain is started from the same initial point.
    """
    assert 'init_params' not in run_kwargs, "`init_params` should not be passed via `run_kwargs`"
    run_kwargs = {'init_params': init_params} | run_kwargs
    n_chains = next(iter(init_params.values())).shape[0]
    run_keys = random.split(rng_key, n_chains)
    mcmc = MCMC(
        kernel, 
        num_chains=n_chains,
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        chain_method="vectorized",
        **mcmc_kwargs
    )
    mcmc.run(
        run_keys, 
        *model_args,
        **(run_kwargs | model_kwargs)
    )
    return mcmc

