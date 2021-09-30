from plum import dispatch

import jax
import jax.numpy as jnp

from netket.utils import struct

from ode4jax.base import AbstractAlgorithm, AbstractIntegrator

from .generic_tableau import (
    AbstractODETableauAlgorithm,
    TrivialAlgorithmCache,
    get_current_adaptive_order,
)

from . import tableau


@struct.dataclass
class AbstractODERKAlgorithm(AbstractODETableauAlgorithm):
    pass


@struct.dataclass
class Euler(AbstractODERKAlgorithm):
    @property
    def tableau(self):
        return tableau.bt_feuler


@struct.dataclass
class Midpoint(AbstractODERKAlgorithm):
    @property
    def tableau(self):
        return tableau.bt_midpoint


@struct.dataclass
class Heun(AbstractODERKAlgorithm):
    @property
    def tableau(self):
        return tableau.bt_heun


@struct.dataclass
class RK4(AbstractODERKAlgorithm):
    @property
    def tableau(self):
        return tableau.bt_rk4


## adaptive methods


@struct.dataclass
class RK12(AbstractODERKAlgorithm):
    @property
    def tableau(self):
        return tableau.bt_rk12


@struct.dataclass
class RK23(AbstractODERKAlgorithm):
    @property
    def tableau(self):
        return tableau.bt_rk23


@struct.dataclass
class RK4Fehlberg(AbstractODERKAlgorithm):
    @property
    def tableau(self):
        return tableau.bt_rk4_fehlberg


@struct.dataclass
class RK45(AbstractODERKAlgorithm):
    @property
    def tableau(self):
        return tableau.bt_rk4_dopri
