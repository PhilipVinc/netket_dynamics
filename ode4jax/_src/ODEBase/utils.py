from numbers import Number

from plum import dispatch

import jax
import jax.numpy as jnp

import pytreearray as pta


@dispatch
def strong_dtype(x):
    if isinstance(x, Number):
        x = jnp.asarray(x)

    return jnp.array(x, dtype=x.dtype)


@dispatch
def strong_dtype(x: pta.PyTreeArray):
    return jax.tree_map(strong_dtype, x)


def expand_dim(arr, sz):
    def _expand(x):
        return jnp.zeros((sz,) + x.shape, dtype=x.dtype)

    res = jax.tree_map(_expand, arr)

    if isinstance(arr, pta.PyTreeArray):
        res = pta.PyTreeArray2(res.tree)

    return res
