import ode4jax
import jax
import jax.numpy as jnp

from matplotlib import pyplot as plt


def f(u, pars, t, **_):
    return 1.01 * u


u0 = 1 / 2
tspan = (0.0, 1.0)
prob = ode4jax.ODEProblem(f, tspan, u0)
sol = ode4jax.solve(prob, ode4jax.algorithms.RK4(), dt=0.01, reltol=1e-8, saveat=101)

exact = u0 * jnp.exp(1.01 * sol.t)

plt.plot(sol.t, sol.u, label="solution to the linear ODE")
plt.plot(sol.t, exact, "--", label="Exact solution")
plt.show()
