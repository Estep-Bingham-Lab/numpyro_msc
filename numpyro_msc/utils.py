import jax

import optax

def adam_loop(target_fun, init_params, n_iter):
    solver = optax.adam(learning_rate=0.003)
    def scan_fn(carry, _):
        params, state = carry
        grad = jax.grad(target_fun)(params)
        updates, state = solver.update(grad, state, params)
        return (optax.apply_updates(params, updates), state), None      
    
    return jax.lax.scan(
        scan_fn, 
        (init_params, solver.init(init_params)),
        length = n_iter
    )[0][0]

def lbfgs_loop(target_fun, init_params, n_iter, lbfgs_params):
    # can reuse stuff because target is deterministic
    # see https://optax.readthedocs.io/en/stable/api/optimizers.html#lbfgs
    cached_value_and_grad = optax.value_and_grad_from_state(target_fun)
    solver = optax.lbfgs(**lbfgs_params)
    def scan_fn(carry, _):
        params, opt_state = carry
        value, grad = cached_value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=target_fun
        )
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None
    
    return jax.lax.scan(
        scan_fn, 
        (init_params, solver.init(init_params)),
        length = n_iter
    )[0][0]

def optimize_fun(target_fun, init_params, n_iter = 2**7, **lbfgs_params):
    """
    Minimize a function using a two step procedure. We first use ADAM to 
    find the basis of attraction of the mode. Then, we use the 
    quasi-Newton method L-BFGS to home in the local mode. Initializing with
    ADAM overcomes some difficulties encountered when deploying L-BFGS using
    single-precision (32 bit) floats [Ref]_.

    :param target_fun: A loss function to minimize.
    :param init_params: The starting pytree.
    :param n_iter: Number of L-BFGS updates to apply. We carry out 32 times
        this number of updates in the ADAM phase.
    :param **lbfgs_params: optional arguments passed to :func:`optax.lbfgs`.
    :return: An updated pytree.

    .. rubric:: References

    .. [Ref] Kiyani, E., Shukla, K., Urbán, J. F., Darbon, J., & Karniadakis, 
        G. E. (2025). Optimizing the optimizer for physics-informed neural 
        networks and Kolmogorov-Arnold networks. Computer Methods in Applied 
        Mechanics and Engineering, 446, 118308..
    """
    init_params = adam_loop(target_fun, init_params, 32*n_iter)
    return lbfgs_loop(target_fun, init_params, n_iter, lbfgs_params)

