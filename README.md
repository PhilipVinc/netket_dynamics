<div align="center">
<img src="https://www.netket.org/_static/logo_simple.jpg" alt="logo" width="400"></img>
</div>

# __NetKet Dynamics, ODE4Jax, PyTreeArrays__

Code to perform time-evolution with netket. Experimental repository.

Also includes `ode4jax` and `pytreearrays`.

## __ODE4Jax__

An experimental package with ODE solvers written in Jax that can be jit-compiled. 
It is heavily inspired from DifferentialEquations.jl. 
Supports arbitrary PyTrees. The code is laid out in order to be able to
support different and advanced solvers and hopefully attract more contributors
in the future.

```python
import ode4jax
from matplotlib import pyplot as plt

def f(u,pars, t, **_):
   return 1.01*u

u0 = 1/2
tspan = (0.0,1.0)
prob = ode4jax.ODEProblem(f, tspan, u0)
sol = ode4jax.solve(prob, ode4jax.RK4(), dt=0.01, reltol=1e-8, saveat=101)

plt.plot(sol.t, sol.u.tree)
```

## __NetKet Dynamics__

Requires Netket#master (or at least commit [a4e54e8](https://github.com/netket/netket/commit/a4e54e895ce931d3bb014bbb500c1848f8f54417) though it's likely to eventually require a more recent commit). So `pip install git+https:github.com/netket/netket`

Introduces a `TimeEvolution` driver using `ode4jax` to perform the time-evolution. See [this example](https://github.com/PhilipVinc/netket_dynamics/blob/master/examples/ising1d/ising1d.py) to see how to use it. The rest of the API is the same as other drivers...

Anyhow, a succint example might be:

```python
vs = MCState(...)
t_end = 1.0

te = nkd.TimeEvolution(ha1, variational_state=vs, algorithm=nkd.Euler(), dt=0.005)
te.run(t_end, out=...)
```




