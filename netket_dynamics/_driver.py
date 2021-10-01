# Copyright 2020 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import singledispatch
from typing import Any, Callable, Optional, Tuple, Union
from netket.utils.types import Array

import numpy as np
import scipy.integrate as _scint
import scipy

import flax
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import netket.jax as nkjax

from netket.operator import AbstractOperator
from ._abstract_variational_driver import AbstractVariationalDriver
from netket.driver.vmc_common import info
from netket.vqs import VariationalState, MCState, MCMixedState
from netket.optimizer.qgt import QGTAuto

import ode4jax

# self is driver
@singledispatch
def dwdt(state, self, w, pars, t, *, stage=None):
    raise NotImplementedError(f"dwdt not implemented for {type(state)}")


@dwdt.register
def dwdt_mcstate(state: MCState, self, w, pars, t, *, stage=None):
    state.reset()

    self._loss_stats, self._loss_grad = state.expect_and_grad(
        self.generator(t),
        use_covariance=True,
    )
    self._loss_grad = tree_map(lambda x: -1.0j * x, self._loss_grad)

    self._S_intermediate = self.qgt(self.state)
    if stage == 0:
        self._S = self._S_intermediate

    x0 = self._dp if self.linear_solver_restart is False else None
    self._dp, self._sr_info = self._S_intermediate.solve(
        self.linear_solver, self._loss_grad, x0=x0
    )

    return self._dp


jstep = jax.jit(ode4jax.step)


class TimeEvolution(AbstractVariationalDriver):
    """
    Variational Time evolution using the time-dependent Variational Monte Carlo (t-VMC).
    """

    def __init__(
        self,
        operator: AbstractOperator,
        *,
        variational_state: VariationalState,
        algorithm,
        qgt=None,
        linear_solver=None,
        linear_solver_restart: bool = False,
        tspan: Optional[Tuple[float, float]] = None,
        t0: Optional[float] = None,
        tend: Optional[float] = None,
        **integrator_kwargs,
    ):
        """
        Construct the time evolution driver.

        Args:
            operator: the generator of the dynamics (hamiltonian, lindbladian...)
            variational_state: The variational state
            solver: the time evolution integrator (like runge kutta or euler)
            qgt: The QGT storage format.
            linear_solver: The solver for solving the linear system determining the time evolution
            linear_solver_restart: If False (default) uses last solution to speedup the solution of the linear system.
            tspan: Specify this as (t0, tend) or the two separately
            t0: initial time
            tend: end time. stop integrating at this time
        """
        if variational_state.hilbert != operator.hilbert:
            raise TypeError(
                f"""the variational_state has hilbert space {variational_state.hilbert} 
                                (this is normally defined by the hilbert space in the sampler), but
                                the operator has hilbert space {operator.hilbert}. 
                                The two should match."""
            )

        if t0 is None:
            t0 = 0.0
        if tend is None:
            tend = np.inf

        if linear_solver is None:
            linear_solver = jax.scipy.sparse.linalg.cg

        if qgt is None:
            qgt = QGTAuto(solver=linear_solver)

        # parse t_span/t0/t_end
        if tspan is not None:
            if t0 is not None or tend is not None:
                raise ValueError(
                    "If tspan is specified, you cannot also specify t0 and tend."
                )
            if not isinstance(tspan, tuple):
                raise TypeError("tspan must be a tuple.")

            t0, tend = tspan

        self._t0 = t0
        self._tend = tend

        super().__init__(
            variational_state, optimizer=None, minimized_quantity_name="generator"
        )

        if isinstance(operator, AbstractOperator):
            self._generator = lambda _: operator.collect()  # type: AbstractOperator
        else:
            self._generator = operator

        self.qgt = qgt  # type: SR
        self.linear_solver = linear_solver
        self.linear_solver_restart = linear_solver_restart

        self._dp = None  # type: PyTree

        u0 = flax.core.unfreeze(self.state.parameters)
        self._problem = ode4jax.ODEProblem(
            self._odefun, (self._t0, self._tend), u0, callback=True
        )

        self._algorithm = algorithm

        self._integrator = ode4jax.init(self._problem, algorithm, **integrator_kwargs)

    @property
    def generator(self) -> AbstractOperator:
        """
        The generator of the dynamics integrated by this driver.
        """
        return self._generator

    @property
    def algorithm(self):
        """
        The solver used to integrate the dynamics
        """
        return self._algorithm

    def _odefun(self, w, p, t, **s):
        """
        The ODE determining the dynamics passed to scipy solvers.

        Args:
            t: The current time (unused).
            w: The parameters as a vector.

        Returns:
            dwdt, the derivative at time t
        """
        # recreate the pytree of parameters from the ode state
        self.state.parameters = w
        # do a timestep
        Δw = dwdt(self.state, self, w, p, t)

        return flax.core.unfreeze(Δw)

    def advance(self, dt=None, n_steps=None):
        """
        Advance the time propagation by `n_steps` simulation steps
        of duration `self.dt`.

           Args:
               :n_steps (int): No. of steps to advance.
        """
        if (dt is None and n_steps is None) or (
            dt is not None and n_steps is not None
        ):
            raise ValueError("Both specified")

        if n_steps is not None:
            for i in range(n_steps):
                self._integrator = jstep(self._integrator)
        elif dt is not None:
            self._integrator = jstep(self._integrator, dt)

        # if self._integrator.status == "failed":
        #    raise ...

    def iter(self, delta_t, t_interval=None):
        """
        Returns a generator which advances the time evolution in
        steps of `step` for a total of `n_iter` times.

        Args:
            :n_iter (int): The total number of steps.
            :step (int=1): The size of each step.

        Yields:
            :(int): The current step.
        """
        t_end = self.t + delta_t
        while self.t < t_end:  # and self._integrator.status == "running":
            t0 = self.t
            yield self.t
            self._step_count += 1
            self._integrator = jstep(self._integrator, t_interval)

    def _log_additional_data(self, obs, step):
        obs["t"] = self.t

    @property
    def _default_step_size(self):
        # Essentially means
        return None

    @property
    def step_value(self):
        return self.t

    @property
    def dt(self):
        return self._integrator.step_size

    @dt.setter
    def dt(self, _dt):
        success = False
        try:
            self._integrator.step_size = _dt
            success = True
        except AttributeError:
            pass

        if not success:
            try:
                self._integrator.h_abs = _dt
                success = True
            except AttributeError:
                pass

        if not success:
            if self._integrator.h_abs == self._integrator.max_step:
                self._integrator.h_abs = _dt
                self.max_step = _dt
            else:
                self._integrator.h_abs = _dt
            success = True

    @property
    def t(self):
        return self._integrator.t

    @t.setter
    def t(self, t):
        self._integrator.t = jnp.array(t, dtype=self._integrator.t)

    @property
    def t_end(self):
        return self._integrator.opts.tstops[-1]

    # @t_end.setter
    # def t_end(self, t_end):
    #    self._integrator.t_bound = t_end

    @property
    def t0(self):
        return self._t0

    def __repr__(self):
        return f"TimeEvolution(step_count={self.step_count}, t={self.t})"

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian ", self._ham),
                ("algorithm   ", self.algorithm),
                ("SR solver   ", self.sr),
                ("State       ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
