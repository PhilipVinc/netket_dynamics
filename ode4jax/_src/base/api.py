from plum import dispatch

import jax
import jax.numpy as jnp

from .problem import AbstractProblem
from .solver import AbstractAlgorithm
from .integrator import AbstractIntegrator


@dispatch
def solve(problem: AbstractProblem, solver: AbstractAlgorithm, *args, **kwargs):
    return _solve(problem, solver, *args, **kwargs)

def init(problem: AbstractProblem, solver: AbstractAlgorithm, *args, **kwargs):
    # use to preprocess arguments
    integrator = _init(problem, solver, *args, **kwargs)
    _initialize(integrator)
    return integrator


@dispatch
def solve(integrator: AbstractIntegrator):
    return _solve(integrator)

def step(integrator: AbstractIntegrator, *args, **kwargs):
    return _step(integrator, *args, **kwargs)

@dispatch
def postamble(integrator: AbstractIntegrator):
    return integrator

# extension points
@dispatch.abstract
def _solve(problem, solver, *args, **kwargs):
    pass


@dispatch.abstract
def _init(problem: AbstractProblem, solver: AbstractAlgorithm, *args, **kwargs):
    pass


@dispatch.abstract
def _initialize(integrator: AbstractIntegrator):
    pass


@dispatch
def _step(integrator: AbstractIntegrator, dt: None):
    return _step(integrator)

@dispatch
def _step(integrator: AbstractIntegrator, dt, stop_at_dt = False):
    next_t = integrator.t + dt
    if stop_at_dt:
        integrator.add_tstop(next_t)

    return _step_until(integrator, next_t)
    

@jax.jit
def _step_until(integrator, tfin):
    def cond_fun(integrator):
        return jnp.logical_and(integrator.t < tfin, jnp.logical_not(integrator.error_code))

    return jax.lax.while_loop(cond_fun, _step, integrator)
