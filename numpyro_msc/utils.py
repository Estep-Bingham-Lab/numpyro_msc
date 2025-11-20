import jax

import optax

# simple scan-based optimization loop
def optimize_fun(target_fun, init_params, n_iter = 2**7, **lbfgs_params):
    """
    Fixed length loop for optimizing a function using :func:`optax.lbfgs` 
    updates.

    :param target_fun: A loss function to minimize.
    :param init_params: The starting pytree.
    :param n_iter: Number of updates to apply.
    :param **lbfgs_params: optional arguments passed to :func:`optax.lbfgs`.
    :return: An updated pytree.
    """
    # can reuse stuff because target is deterministic
    # see https://optax.readthedocs.io/en/stable/api/optimizers.html#lbfgs
    cached_value_and_grad = optax.value_and_grad_from_state(target_fun)

    # build solver and loop function
    solver = optax.lbfgs(**lbfgs_params)
    def scan_fn(carry, _):
        params, opt_state = carry
        value, grad = cached_value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=target_fun
        )
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None
    
    # optimization loop
    opt_params, _ = jax.lax.scan(
        scan_fn, 
        (init_params, solver.init(init_params)),
        length = n_iter
    )[0]
    
    return opt_params