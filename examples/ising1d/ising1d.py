# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import netket as nk
import netket_dynamics as nkd

import matplotlib.pyplot as plt

# 1D Lattice
L = 10

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, use_visible_bias=True, dtype=complex)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisHamiltonian(hi, ha, n_chains=16)

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1000, n_discard_per_chain=100)

# Optimizer
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=1e-4)

# Variational monte carlo driver
gs = nk.VMC(ha, op, variational_state=vs)

# Create observable
Sx = sum([nk.operator.spin.sigmax(hi, i) for i in range(L)])

# Run the optimization for 300 iterations
gs.run(n_iter=150, out="example_ising1d_GS", obs={"Sx": Sx})
W_0 = vs.parameters
ket_0 = vs.to_qobj()

# Create solver for time propagation
ha1 = nk.operator.Ising(hilbert=hi, graph=g, h=0.5)
vs.parameters = W_0
te = nkd.TimeEvolution(ha1, variational_state=vs, algorithm=nkd.Euler(), dt=0.005)

log = nk.logging.JsonLog("example_ising1d_TE")
te.run(1.0, out=log, show_progress=True, obs={"SX": Sx})

plt.plot(log.data["t"], log.data["SX"])
plt.show()


## try to compute the exact slope
import qutip
import numpy as np

tvals = np.arange(0.0, 1.0, 0.01)
sol = qutip.sesolve(ha1.to_qobj(), ket_0, tvals, e_ops=[Sx.to_qobj()])

plt.plot(sol.times, sol.expect[0], label="Exact")
plt.plot(log.data["t"], log.data["SX"], label="NetKet")
plt.xlabel("time t")
plt.ylabel("⟨Sx⟩")
plt.legend()
plt.show()
