import ode4jax
import jax
import jax.numpy as jnp

from matplotlib import pyplot as plt

plt.ion()


def f(u, pars, t, **_):
    return 1.01 * u


u0 = 1 / 2
tspan = (0.0, 0.1)
prob = ode4jax.ODEProblem(f, tspan, u0)
it = ode4jax.init(
    prob, ode4jax.algorithms.RK4Fehlberg(), dt=0.01, reltol=1e-8, saveat=11
)

jstep = jax.jit(ode4jax.step)

while it.t < tspan[1]:
    it = jstep(it)

sol = it.solution

exact = u0 * jnp.exp(1.01 * sol.t)

plt.plot(sol.t, sol.u, label="solution to the linear ODE")
plt.plot(sol.t, exact, "--", label="Exact solution")
plt.show()
