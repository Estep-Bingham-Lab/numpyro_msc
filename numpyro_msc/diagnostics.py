from functools import partial
import math

import jax
from jax import numpy as jnp

##############################################################################
# Nested R-hat diagnostic
##############################################################################

# main function
def nested_rhats(mcmc=None, n_super=None, msc_samples=None):
    """
    Compute nested Rhat diagnostics for each dimension of every latent 
    variable in the model.

    :param mcmc: An instance of :class:`numpyro.infer.MCMC` produced by a call
        to :func:`many_short_chains`.
    :param n_super: The number of superchains used for running 
        :func:`many_short_chains`.
    :param msc_samples: Alternatively, the user can directly pass a `dict` with
        entries given by arrays of shape `(n_super, n_within, n_steps, ...)`.
    :return: A `dict` with the nested Rhat value for each latent variable.
    """
    # check we have enough info
    if msc_samples is None and (mcmc is None or n_super is None):
        raise ValueError(
            "Insufficient inputs: either pass `msc_samples` or both of " \
            "`mcmc` and `n_super`"
        )

    # check mcmc or msc_samples
    if msc_samples is None and mcmc is not None:
        msc_samples = jax.tree.map(
            partial(grouped_samples_to_msc, n_super=n_super),
            mcmc.get_samples(True)
        )
    
    # compute nested rhat for each element of the dict, flattening 
    # multidimensional arrays into vectors
    def tree_map_fn(xs):     
        # dispatch depending on the shape of the underlying latent var
        xs_shape = jnp.shape(xs)
        if len(xs_shape) == 3:
            # scalar latent var
            return nested_rhat_univariate(xs)
        if len(xs_shape) > 4:
            # 2D or higher latent var -> flatten into vector
            flattened_len = math.prod(xs_shape[3:])
            xs = jnp.reshape(xs, shape=(*xs_shape[:3], flattened_len))
        return jax.vmap(nested_rhat_univariate, in_axes=3)(xs)
    
    return jax.tree.map(tree_map_fn, msc_samples)

# univariate base case
def nested_rhat_univariate(xs):
    """
    Implements Definition 2.2 in Margossian et al. (2024).

    :param xs: An array of shape `(n_super, n_within, n_steps)`.
    :return: The nested ̂R value.
    """
    # check shape is (super,within,step) and min values
    xs_shape = jnp.shape(xs)
    assert len(xs_shape) == 3
    n_super, n_within, n_steps = xs_shape
    assert n_super > 0 and n_within > 0 and n_steps > 0

   # hatB, Eq 6
    var_of_superchain_means = (
        jnp.zeros((), xs.dtype)
        if n_super == 1 
        else xs.mean(axis=(-1,-2)).var(ddof=1)
    )

    # tildeB
    within_superchain_vars_of_means = (
        jnp.zeros_like(var_of_superchain_means)
        if n_within == 1
        # for each superchain, compute the variance of its chains' means
        else jax.vmap(lambda x: x.mean(axis=-1).var(ddof=1))(xs)
    )

    # tildeW
    within_superchain_means_of_vars = (
        jnp.zeros_like(var_of_superchain_means)
        if n_steps == 1 
        # for each superchain, compute the average of the intra chain variances
        else jax.vmap(lambda x: x.var(axis=-1, ddof=1).mean())(xs)
    )

    # hatW, Eq 7
    hatW = jnp.mean(
        within_superchain_vars_of_means + within_superchain_means_of_vars
    )

    # hatR_nu, Eq 8
    return jnp.sqrt(1+var_of_superchain_means/hatW)

# reshape: (n_super, n_within, n_step, ...) -> (n_chains, n_step, ...)
def msc_samples_to_grouped(xs):
    xs_shape = jnp.shape(xs)
    n_chains = xs_shape[0]*xs_shape[1]
    return jnp.reshape(xs, (n_chains, *xs_shape[2:]))

# reshape: (n_chains, n_step, ...) -> (n_super, n_within, n_step, ...)
def grouped_samples_to_msc(xs, n_super):
    xs_shape = jnp.shape(xs)
    n_within = xs_shape[0] // n_super
    assert n_super*n_within == xs_shape[0] # check that n_super was a true divisor
    return jnp.reshape(xs, (n_super, n_within, *xs_shape[1:]))

def max_nested_rhat(**kwargs):
    """
    Return the maximum nested Rhat across dimensions.

    :param kwargs: arguments to the :func:`nested_rhats` function.
    """
    n_rhats = nested_rhats(**kwargs)
    return float(jax.tree.reduce(max, jax.tree.map(jnp.max, n_rhats)))
