from typing import Any, Union, Optional, Tuple

import jax.numpy as jnp

from functools import partial

from ..core import PyTreeArray

from .util import _wraps


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


@_wraps(jnp.sqrt)
def sqrt(a: PyTreeArray):
    return jax.tree_map(jnp.sqrt, a)


@_wraps(jnp.abs)
def abs(a: PyTreeArray):
    return jax.tree_map(jnp.abs, a)
