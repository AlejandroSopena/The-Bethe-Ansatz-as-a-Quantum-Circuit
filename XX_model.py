import numpy as np
from qibo import Circuit, gates


class XX_model:
    def __init__(self, nqubits, nmagnons):
        self.nqubits = nqubits
        self.nmagnons = nmagnons
        self.delta = 0
        self.roots = None
        self.Plist = None
        self.circuit = None

    def get_roots(self, lambdas, type='Rapidities'):
        gamma = np.arccos(self.delta)
        roots = []
        for l in lambdas:
            if type == 'Rapidities':
                roots.append(np.sinh(gamma*(l-1j)/2)/np.sinh(gamma*(l+1j)/2))
            elif type == 'Momentum':
                roots.append(np.exp(1j*l))
        self.roots = roots

    def Ck(self, k, x, y):
        if k == 0:
            c = 1
        else:
            c = self.Ck(k-1, x, y)*x*np.conjugate(y)+1
        return c

    def Djk(self, j, k):
        d = np.zeros((j, j), complex)
        for r in range(j):
            for c in range(j):
                d[r, c] = self.Ck(k, self.roots[c], self.roots[r])
        det = np.linalg.det(d)

        return det

    def Fjk(self, j, k):
        f = np.zeros((j, j), complex)
        for r in range(j-1):
            for c in range(j):
                f[r, c] = self.Ck(k, self.roots[c], self.roots[r])
        for c in range(j):
            f[j-1, c] = 1
        det = np.linalg.det(f)

        return det

    def ajk(self, j, k):
        phase = np.prod(
            (self.roots[0:j-1]-self.roots[j-1])/np.abs(self.roots[0:j-1]-self.roots[j-1]))
        a_jk = (1/(phase*np.prod(np.conjugate(self.roots[0:j-1]))))*self.Fjk(
            j, k)/np.sqrt(self.Djk(j-1, k-1)*self.Djk(j, k))

        return a_jk

    def bjk(self, j, k):
        b_jk = self.roots[j-1]*np.sqrt(self.Djk(j-1, k) *
                                       self.Djk(j, k-1)/(self.Djk(j-1, k-1)*self.Djk(j, k)))

        return b_jk

    def M(self, j, k):
        M = np.zeros((4, 4), complex)
        M[0, 0] = 1
        M[3, 3] = 1
        M[1, 1] = self.ajk(j, k)
        M[2, 2] = np.conjugate(M[1, 1])
        M[2, 1] = self.bjk(j, k)
        M[1, 2] = -np.conjugate(M[2, 1])

        return M

    def Pk(self, k):
        P = []
        for j in range(1, min(k, self.nmagnons)+1):
            P.append(self.M(j, k))

        return P[::-1]

    def P_list(self):
        P = []
        for k in range(1, self.nqubits):
            P.append(self.Pk(k))
        self.Plist = P[::-1]

    def get_circuit(self):
        self.circuit = Circuit(self.nqubits)
        for n in range(self.nmagnons):
            self.circuit.add(gates.X(n))
        for n in range(self.nqubits-self.nmagnons):
            qubits = list(np.arange(n, n+self.nmagnons+1, 1))[::-1]
            Pn = self.Plist[n]
            for j in range(len(qubits)-1):
                self.circuit.add(gates.Unitary(Pn[j], *qubits[j:j+2]))

        j = 0
        for n in range(self.nqubits-self.nmagnons, self.nqubits-1):
            qubits = list(
                np.arange(self.nqubits-self.nmagnons+j, self.nqubits, 1))[::-1]
            Pn = self.Plist[n]
            for k in range(len(qubits)-1):
                self.circuit.add(gates.Unitary(Pn[k], *qubits[k:k+2]))
            j += 1

    def check(self, j, k):
        check = np.abs(self.Fjk(j, k))**2+self.Djk(j-1, k)*self.Djk(j, k-1)*np.prod(np.abs(
            self.roots[0:j])**2)-self.Djk(j, k)*self.Djk(j-1, k-1)*np.prod(np.abs(self.roots[0:j-1])**2)

        return check
