# The Bethe Ansatz as a Quantum Circuit

This repository contains the code to reproduce the numerical implementations presented in the manuscript ["The Bethe Ansatz as a Quantum Circuit"](https://arxiv.org/abs/2309.14430).


## Dependences

- `Pyhton>=3.11.7`

- `qibo==0.2.5`


## Usage
[`XXZ_model.py`](https://github.com/AlejandroSopena/The-Bethe-Ansatz-as-a-Quantum-Circuit/blob/main/XXZ_model.py) contains a class to generate the unitary matrices $P_k$ (unitaries for $k < M$ and isometries $P_k|0\rangle$ for $k\geq M$) derived from the $\Lambda$ tensors.
```python
from XXZ_model import XXZ_model

nspins = 4
nmagnons = 2
delta = 0.5
roots = [-0.574, 0.574]

t = XXZ_model(nspins, nmagnons, delta)
t.get_roots(roots)
t.get_indexes()

P_xxz = []
for k in range(1, nspins):
    P_xxz.append(t.get_Pk_matrix(k=k, a=True, b=True))
```

[`XXZ_model_QR.py`](https://github.com/AlejandroSopena/The-Bethe-Ansatz-as-a-Quantum-Circuit/blob/main/XXZ_model_QR.py) contains the functions to generate the matrices $P_k$ derived from the $\Gamma$ tensors using the QR decomposition.
```python
from XXZ_model_QR import get_P_G

nspins = 4
roots = [-0.574, 0.574]
delta = 0.5
roots = [-0.574, 0.574]

P_xxz_qr = get_P_G(nspins, roots, delta)[0]
```

[`XX_model.py`](https://github.com/AlejandroSopena/The-Bethe-Ansatz-as-a-Quantum-Circuit/blob/main/XX_model.py) contains a class to generate the quantum circuit to prepare Bethe eigensates of the XX model. These circuits are efficient in the number of qubits and magnons.

```python
from XX_model import XX_model

nspins = 4
nmagnons = 2
roots = [-0.561, 0.561]

t = XX_model(nspins, nmagnons)
t.get_roots(roots)
t.P_list()
t.get_circuit()
circ = t.circuit
state_xx_efficient = circ().state()
```



[`bethe_circuit.py`](https://github.com/AlejandroSopena/Algebraic-Bethe-Circuits/blob/main/bethe_circuit.py) defines the class `BetheCircuit` which implements the Bethe Ansatz for the XXZ model with both the non-unitary matrices $R$ and the unitary matrices $P_k$.
```python
import numpy as np
from qibo.quantum_info import fidelity
from bethe_circuit import BetheCircuitTN

nspins = 4
nmagnons = 2
roots = [-0.574, 0.574]
delta = 0.5

v = BetheCircuitTN(nspins, nmagnons)
state_mps = v.mps_state(roots, delta)().state()
state_mps = [state_mps[i] for i in range(0, len(state_mps), 2**nmagnons)]
state_mps /= np.linalg.norm(state_mps)

state_lambda = v.unitary_circuit(P_xxz)().state()
state_gamma = v.unitary_circuit(P_xxz_qr)().state()

fidelity_mps_lambda = fidelity(state_mps, state_lambda)
fidelity_mps_gamma = fidelity(state_mps, state_gamma)

print('fidelity lambda', fidelity_mps_lambda)
print('fidelity gamma', fidelity_mps_gamma)
```

The fidelity of the XX eigenstate can be computed as
```python
nspins = 4
nmagnons = 2
roots = [-0.561,0.561]
delta = 0

v = BetheCircuitTN(nspins, nmagnons)
state_mps = v.mps_state(roots,delta)().state()
state_mps =  [state_mps[i] for i in range(0,len(state_mps),2**nmagnons)]
state_mps /= np.linalg.norm(state_mps)

fidelity_mps_xx = fidelity(state_mps,state_xx_efficient)
print('fidelity xx', fidelity_mps_xx)
```