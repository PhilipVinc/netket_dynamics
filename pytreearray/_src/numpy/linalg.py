from typing import Any, Union, Optional, Tuple

import jax.numpy as jnp

from functools import partial

from ..core import PyTreeArray

from .util import _wraps
from . import pta_numpy as pnp


@_wraps(jnp.sum)
def sum(
    a: PyTreeArray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype=None,
    out=None,
    keepdims=None,
    initial=None,
    where=None,
):
    if axis is not None:
        raise NotImplementedError

    return jax.tree_util.tree_reduce(
        jnp.add, jax.tree_map(partial(jnp.sum, dtype=dtype), a.tree)
    )


@_wraps(jnp.linalg.norm)
def norm(a: PyTreeArray, ord=None, axis=None):
    if ord is not None or axis is not None:
        raise NotImplementedError
    return jax.lax.sqrt(pnp.sum(pnp.abs(a) ** 2))
