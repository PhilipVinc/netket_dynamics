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

from builtins import RuntimeError, next
from functools import partial
from typing import Callable, Optional, Tuple, Type

import jax
import jax.numpy as jnp

from netket.utils import struct
from netket.utils.types import Array, PyTree

import pytreearray as pta


def unwrap_call(fun, u, *args, **kwargs):
    if isinstance(u, pta.PyTreeArray):
        wrap = True
        _u = u.tree
    else:
        wrap = False
        _u = u

    res = fun(_u, *args, **kwargs)

    if wrap:
        res = pta.PyTreeArray1(res)

    return res


def unpack_call(fun, args_and_kwargs):
    args, kwargs = args_and_kwargs
    return fun(*args, **kwargs)


@struct.dataclass
class AbstractFunction:
    f: Callable = struct.field(pytree_node=False)

    def __call__(self, u, p, t, /, **kwargs):
        """
        When called, unwrap from pytreearray that might cause problems with user
        code.
        """
        return unwrap_call(self.f, u, p, t, **kwargs)

    def __repr__(self):
        return f"{self.__name__} for function {self.f}"


@struct.dataclass
class StandardFunction(AbstractFunction):
    pass


@struct.dataclass
class CallbackFunction(AbstractFunction):
    def __call__(self, u, p, t, /, **kwargs):
        from jax.experimental import host_callback as hcb

        def _hcb_call(u, p, t, **kwargs):
            args = (u, p, t)
            pyfun = partial(unpack_call, self.f)
            res_shape = jax.tree_map(
                lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), u
            )
            return hcb.call(pyfun, (args, kwargs), result_shape=res_shape)

        return unwrap_call(_hcb_call, u, p, t, **kwargs)


def wrap_function(fun, /, *, callback=False):
    if callback is True:
        return CallbackFunction(fun)
    else:
        return StandardFunction(fun)
