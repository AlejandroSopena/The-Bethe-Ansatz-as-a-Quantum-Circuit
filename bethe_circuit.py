import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibo import gates


class BetheCircuitTN:
    """
    Class to build the Bethe eigenstates of the XXZ model using the unitary matrices P_k.
    It is also posible to prepare the eigenstates using the R-matrix.
    """

    def __init__(self, nspins, nmagnons):
        """
        Args:
            nspins (int): number of qubits.
            nmagnons (int): number of magnons.
        """
        self.nspins = nspins
        self.nmagnons = nmagnons

    def unitary_circuit(self, Ps):
        """
        Unitary quantum circuit to build Bethe eigenstates.

        Args:
            Ps (list): Unitary matrices Pk obtained through the Gamma or Lambda tensors.

        Returns:
            (qibo.models.Circuit): Quantum circuit that prepares the corresponding Bethe eigenstate.
        """
        self.circuit = Circuit(self.nspins)
        unitaries = Ps[::-1]
        for n in range(self.nmagnons):
            self.circuit.add(gates.X(n))
        for n in range(self.nspins-self.nmagnons):
            qubits = list(np.arange(n, n+self.nmagnons+1, 1))[::-1]
            self.circuit.add(gates.Unitary(unitaries[n], *qubits))
        j = 0
        for n in range(self.nspins-self.nmagnons, self.nspins-1):
            qubits = list(
                np.arange(self.nspins-self.nmagnons+j, self.nspins, 1))[::-1]
            self.circuit.add(gates.Unitary(unitaries[n], *qubits))
            j += 1
            
        return self.circuit

    def mps_state(self, roots, delta, type='Rapidities'):
        """
        Prepare a Bethe eigenstate of the XXZ model using the MPS representation of the Algebraic Bethe Ansatz.

        Args:
            roots (list of complex): root of the Bethe equations.
            delta (float): anisotropy in the XXZ model.
            type (str, optional): Specifies the type of the roots. Defaults to 'Rapidities'. Other option is 'Momentum'.

        Returns:
            (qibo.models.Circuit): The tensor network (circuit) representing the eigenstate.
        """

        self.tensornet = Circuit(self.nspins+self.nmagnons)
        for n in range(self.nmagnons):
            self.tensornet.add(gates.X(n))
        for i, m in enumerate(range(self.nmagnons, 0, -1)):
            for n in range(self.nspins):
                self.tensornet.add(gates.Unitary(
                    self._r_matrix_xxz(roots[i], delta, type), n+m-1, n+m))

        return self.tensornet

    def _r_matrix_xxz(self, root, delta, type='Rapidities'):
        """
        R matrix used in the 6-vertex model to construct the state of interest. 

        Args:
            root (complex): root of the Bethe equations.
            delta (float): anisotropy in the XXZ model.
            type (str, optional): Specifies the type of the root. Defaults to 'Rapidities'. Other option is 'Momentum'.

        Returns:
            (array): The R matrix. 
        """

        r_matrix = np.eye(4, dtype=np.complex128)

        if type == 'Rapidities':
            if delta == 1:
                b = (root-1j) / (root+1j)
                c = 2j / (root+1j)
            elif delta > 1:
                gamma = np.arccosh(delta)
                b = np.sin(gamma/2 * (root-1j)) / np.sin(gamma/2 * (root+1j))
                c = 1j * np.sinh(gamma) / np.sin(gamma/2 * (root+1j))
            else:
                gamma = np.arccos(delta)
                b = np.sinh(gamma/2 * (root-1j)) / np.sinh(gamma/2 * (root+1j))
                c = 1j * np.sin(gamma) / np.sinh(gamma/2 * (root+1j))
        elif type == 'Momentum':
            b = np.exp(1j*root)
            c = np.sqrt(1+b**2-2*b*delta)

        r_matrix[1, 1] = r_matrix[2, 2] = c
        r_matrix[1, 2] = r_matrix[2, 1] = b

        return r_matrix
