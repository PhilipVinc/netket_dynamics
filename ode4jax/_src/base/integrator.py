# Copyright 2021 The NetKet Authors - All Rights Reserved.
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

from typing import Callable

from plum import dispatch

from builtins import RuntimeError, next
from functools import partial
from typing import Callable, Optional, Tuple, Type

import numpy as np
import jax
import jax.numpy as jnp

from netket.utils import struct
from netket.utils.types import Array, PyTree

from .solution import AbstractSolution
from .options import AbstractDEOptions

dtype = jnp.float64

@struct.dataclass(_frozen=False)
class AbstractIntegrator:
    solution: AbstractSolution
    opts: AbstractDEOptions

    force_stepfail: bool
    error_code: int

    @jax.jit
    def solve(self):
        from .api import solve
        return solve(self)

    def du(self):
        """
        Return the derivative at current t
        """
        raise NotImplementedError

    def proposed_dt(self):
        """
        gets the proposed dt for the next timestep
        """
        raise NotImplementedError

    def set_proposed_dt(self):
        """
        returns a new integrator with the proposed value of dt set
        """
        raise NotImplementedError

    def set_savevalues(self, force_save):
        """
            savevalues!(integrator::DEIntegrator,
              force_save=false) -> Tuple{Bool, Bool}
        Try to save the state and time variables at the current time point, or the
        `saveat` point by using interpolation when appropriate. It returns a tuple that
        is `(saved, savedexactly)`. If `savevalues!` saved value, then `saved` is true,
        and if `savevalues!` saved at the current time point, then `savedexactly` is
        true.
        The saving priority/order is as follows:
          - `save_on`
            - `saveat`
            - `force_save`
            - `save_everystep`
        """
        raise NotImplementedError

    def add_tstop(self, t):
        eps = 2*jnp.finfo(self.t).eps

        if t <= (self.t + eps):
            return 
        elif not self.has_tstops:
            self.opts.tstops = jnp.asarray(np.insert(self.opts.tstops, self.opts.next_tstop_id, t))
        elif t<=self.first_tstop:
            if np.abs(t-self.first_tstop) < eps:
                # ignore
                return
            self.opts.tstops = jnp.asarray(np.insert(self.opts.tstops, self.opts.next_tstop_id, t))
        else:
            self.opts.tstops = jnp.asarray(np.insert(self.opts.tstops, np.searchsorted(self.opts.tstops, t), t))


    def add_saveat(self, t):
        raise NotImplementedError

    def set_abstol(self, t):
        raise NotImplementedError

    def set_reltol(self, t):
        raise NotImplementedError

    def reinit(self, *args):
        """
        resets the integrator
        """
        raise NotImplementedError

    @property
    def has_tstops(self):
        return self.opts.next_tstop_id < self.opts.tstops.size

    @property
    def first_tstop(self):
        return self.opts.tstops[self.opts.next_tstop_id]


    # ITERATOR INTERFACE
    @property
    def done(self):
        is_solving = self.has_tstops
        no_error = jnp.logical_not(self.error_code)

        return jnp.logical_not(jnp.logical_and(is_solving, no_error))

    @jax.jit
    def step(self, dt=None):
        """
        Performs one step of integration
        """
        from .api import step
        if dt is None:
            return step(self)
        else:
            return step(self, dt)

    def __iter__(self):
        return _IteratorWrapper(self)

class _IteratorWrapper:
    """
    Wrapper to update the state during iteration
    """
    def __init__(self, integrator):
        self.integrator = integrator

    def __iter__(self):
        return self

    def __next__(self):
        it = self.integrator
        if it.done:
            raise StopIteration
        else:
            it = it.step()
            self.integrator = it

            return (it.t, it.u)

