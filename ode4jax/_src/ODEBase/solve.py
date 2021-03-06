from typing import Any, Callable

from plum import dispatch

from jax.experimental import loops

import numpy as np
import jax
import jax.numpy as jnp

from jax.experimental import host_callback as hcb

from ode4jax.base import init, postamble

from .problem import ODEProblem
from .integrator import ODEIntegrator
from .algorithms import AbstractODEAlgorithm, perform_step


@dispatch
def _solve(problem: ODEProblem, alg: AbstractODEAlgorithm, *args, **kwargs):
    integrator = init(problem, alg, *args, **kwargs)
    return _solve(integrator)

@dispatch
def _solve(integrator: ODEIntegrator):
    def cond_fun(integrator):
        return jnp.logical_not(integrator.done)

    def while_fun(integrator):
        # def _cond_fun(integrator):
        #  return jnp.all(integrator.tdir * integrator.t < integrator.opts.next_tstops)
        #
        # def _while_fun(integrator):
        #  loopheader(integrator)
        #  #
        #  perform_step(integrator, integrator.cache)
        #  loopfooter(integrator)
        #
        # integrator = jax.lax.while_loop(_cond_fun, _while_fun, integrator)
        # integrator = handle_tstop(integrator)
        return _step(integrator)

    integrator = jax.lax.while_loop(cond_fun, while_fun, integrator)

    integrator = postamble(integrator)

    return integrator.solution


@dispatch
def _step(integrator: ODEIntegrator):
    # cache some data from previous step and check that the timestep is valid,
    # and recompute it.
    integrator = loopheader(integrator)
    integrator.error_code = check_error(integrator)

    # if itnegrator.error_code != 0 :
    #   postamble!(integrator)
    #   break return

    # perform the actual integration step. updates u but not t
    integrator = perform_step(integrator, integrator.alg, integrator.cache)

    #
    integrator = loopfooter(integrator)

    #
    integrator = handle_tstop(integrator)

    return integrator


# maybe remove
_jstep = jax.jit(_step)

###############################
###############################
###############################
# utils
def loopheader(integrator: ODEIntegrator):
    """
    Applied right after iterators/callback, at the beginning of a solve step.

    Evaluates if the last step should be accepted or rejected (only executed after
    the first step).
    """
    # Apply right after iterators / callbacks

    # If this is at least the second iteration, apply the update from the last
    # succesfull step
    cond = integrator.iter > 0

    def true_body(integrator):
        if integrator.opts.adaptive:

            def _accept_step(integrator):
                integrator.success_iter += 1
                integrator = apply_step(integrator)
                return integrator

            def _reject_step(integrator):
                integrator.opts.controller.step_reject_controller(
                    integrator, integrator.alg
                )
                return integrator

            integrator = jax.lax.cond(
                integrator.accept_step, _accept_step, _reject_step, integrator
            )
        else:
            integrator.success_iter += 1
            integrator = apply_step(integrator.replace())
        return integrator

    def false_body(integrator):
        integrator = integrator.replace()
        return integrator

    integrator = jax.lax.cond(cond, true_body, false_body, integrator)

    # say that we have advanced one iteration
    integrator.iter += 1
    # Fix the timestep To be in the range [dtmin, dtmax]
    integrator = fix_dt_at_bounds(integrator)
    # Adapt the timestep to hit the tstops
    integrator = modify_dt_for_tstops(integrator)
    integrator.force_stepfail = jnp.asarray(False)

    return integrator


def fix_dt_at_bounds(integrator):
    """
    Fix dt to fit inside [dtmin, dtmax]
    """
    cond = jnp.all(integrator.tdir > 0)

    def true_body(integrator):
        dt = jnp.minimum(integrator.opts.dtmax, integrator.dt)
        dt = jnp.maximum(dt, integrator.opts.dtmin)
        return integrator.replace(dt=dt)

    def false_body(integrator):
        dt = jnp.maximum(integrator.opts.dtmax, integrator.dt)
        dt = jnp.minimum(dt, integrator.opts.dtmin)
        return integrator.replace(dt=dt)

    return jax.lax.cond(cond, true_body, false_body, integrator)


def modify_dt_for_tstops(integrator):
    """
    Change the timestep to hit the next tstop if in range
    """
    cond = integrator.has_tstops

    def true_body(integrator):
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = integrator.first_tstop

        # If we are adaptive, change the dt
        if integrator.opts.adaptive:
            new_dt = integrator.tdir * jnp.minimum(
                jnp.abs(integrator.dt), jnp.abs(tdir_tstop - tdir_t)
            )
            integrator = integrator.replace(dt=new_dt)
            # missing case from julia integrator_utils:42
            # elseif iszero(integrator.dtcache) && integrator.dtchangeable
            #    integrator.dt = integrator.tdir * abs(tdir_tstop - tdir_t)
        # If we are not adaptive, then compute the dt to hit the point desired
        else:
            new_dt = integrator.tdir * jnp.minimum(
                jnp.abs(integrator.dtcache), jnp.abs(tdir_tstop - tdir_t)
            )
            integrator = integrator.replace(dt=new_dt)
        return integrator

    integrator = jax.lax.cond(
        integrator.has_tstops,
        true_body,
        lambda integrator: integrator.replace(),
        integrator,
    )
    return integrator


def apply_step(integrator: ODEIntegrator):
    # integrator.accept_step = jnp.asarray(False)
    integrator.uprev = integrator.u
    integrator.dt = integrator.dtpropose

    if integrator.alg.is_fsal:
        integrator.fsalfirst = integrator.fsallast

    ## fsal
    return integrator


###############################
###############################
###############################


def check_error(integrator: ODEIntegrator):
    # integratorinterface::347
    _dt_nan = jnp.isnan(integrator.dt)
    _maxiter = integrator.iter > integrator.opts.maxiters

    return jnp.logical_or(_dt_nan, _maxiter)


# this is slow
def get_error(integrator: ODEIntegrator):
    def passthrough(val, errcode):
        return errcode * (not val == 0) + (val == 0) * val

    _dt_nan = jnp.isnan(integrator.dt)
    _maxiter = integrator.iter > integrator.opts.maxiters

    res = jax.lax.cond(_dt_nan, lambda res: passthrough(res, 1), lambda res: res, 0)
    res = jax.lax.cond(_maxiter, lambda res: passthrough(res, 2), lambda res: res, 0)

    return res


###############################
###############################
###############################

# integrator_utils 220
def loopfooter(integrator: ODEIntegrator):
    """
    Update the current time if the step was succesfull, otherwise computes new adaptive timestep.

    Executes callbacks and save data
    """

    # set flags for fsal

    # if integrator.step_forcefail
    #
    ttmp = integrator.t + integrator.dt
    if integrator.opts.adaptive:
        #
        q = integrator.opts.controller.stepsize_controller(integrator, integrator.alg)
        integrator.accept_step = integrator.opts.controller.accept_step_controller(
            integrator
        )  # forcedttmin

        # if accept_step:
        def accept_fun(integrator):
            dtnew = integrator.opts.controller.step_accept_controller(
                integrator, integrator.alg, q
            )
            integrator.tprev = integrator.t
            integrator.t = ttmp
            # TODO round integrator.t to nearest tstop if needed
            integrator.dtpropose = calc_dt_propose(integrator, dtnew)
            # handle callbacks
            return integrator

        def reject_fun(integrator):
            return integrator

        integrator = jax.lax.cond(
            integrator.accept_step, accept_fun, reject_fun, integrator
        )

    else:
        integrator.tprev = integrator.t

        # missing advanced logic to fix integrator.t to current tstop
        integrator.t = ttmp
        integrator.accept_step = jnp.asarray(True)
        integrator.dtpropose = integrator.dt

    # TODO: handle callbacks
    # process saveat points, and eventually save the data.
    saveat(integrator)

    return integrator


def calc_dt_propose(integrator: ODEIntegrator, dtnew):
    dtpropose = integrator.tdir * jnp.minimum(
        jnp.abs(integrator.opts.dtmax), jnp.abs(dtnew)
    )
    dtpropose = integrator.tdir * jnp.maximum(
        jnp.abs(dtpropose), jnp.abs(integrator.opts.dtmin)
    )
    return dtpropose


from .dense import _ode_addsteps, ode_interpolant


def saveat(integrator):
    """
    Process saveat instructions.
    """
    if integrator.opts.saveat is None:
        return
    elif len(integrator.opts.saveat) == 0:
        return

    # TODO: this only handles one saved point per time-step.
    # in principle there could be more than one
    next_save_t = integrator.opts.saveat[integrator.opts.next_saveat_id]
    cond = next_save_t <= integrator.t + 2 * jnp.finfo(integrator.t.dtype).eps

    def do_save_body(solution):
        curt = integrator.tdir * next_save_t

        # if curt does not match integrator.t exactly...
        k = integrator.k
        # k = addsteps(integrator)
        theta = (curt - integrator.tprev) / integrator.dt
        idxs = 0  # integrator.opts.save_idxs
        save_val = ode_interpolant(theta, integrator, idxs, 0)
        solution = solution.replace()

        solution.set(integrator.saveiter, curt, save_val)
        # solution.set(integrator.saveiter, integrator.t, integrator.u)

        # if integrator.t == curt then
        # solution = solution.replace()
        # solution.set(integrator.saveiter, integrator.t, integrator.u)
        saveiter = integrator.saveiter + 1
        saveiter_dense = integrator.saveiter_dense + 1
        return (solution, saveiter, saveiter_dense)

    def not_save_body(solution):
        return solution.replace(), integrator.saveiter, integrator.saveiter_dense

    s, si, sid = jax.lax.cond(cond, do_save_body, not_save_body, integrator.solution)
    integrator.solution = s
    integrator.saveiter = si
    integrator.saveiter_dense = sid
    integrator.opts.next_saveat_id = integrator.opts.next_saveat_id + cond * 1


###############################
###############################
###############################


def handle_tstop(integrator):
    def has_tstop_body(integrator):
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = integrator.first_tstop

        cond = tdir_t == tdir_tstop

        def _true_body(integrator):
            return integrator.opts.next_tstop_id + 1

        def _false_body(integrator):
            return integrator.opts.next_tstop_id

        return jax.lax.cond(cond, _true_body, _false_body, integrator)

    def not_tstop_body(integrator):
        return integrator.opts.next_tstop_id

    next_tstop = jax.lax.cond(
        integrator.has_tstops, has_tstop_body, not_tstop_body, integrator
    )
    integrator.opts.next_tstop_id = next_tstop
    return integrator
