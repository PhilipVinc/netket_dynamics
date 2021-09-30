<div align="center">
<img src="https://www.netket.org/_static/logo_simple.jpg" alt="logo" width="400"></img>
</div>

# __NetKet Dynamics, ODE4Jax, PyTreeArrays__

Code to perform time-evolution with netket. Experimental repository.

Also includes `ode4jax` and `pytreearrays`.

# __ODE4Jax__

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

