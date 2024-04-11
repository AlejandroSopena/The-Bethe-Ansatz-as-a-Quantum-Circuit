import numpy as np
from itertools import permutations
from sympy import LeviCivita


def choose(n, m):

    return np.math.factorial(n)/(np.math.factorial(m)*np.math.factorial(n-m))


def binarize(decimal, L):
    binary = np.zeros(L)
    bin = ''
    for i in range(L):
        binary[L - 1 - i] = decimal % 2
        decimal = decimal // 2
        bin = bin + str(binary.astype(int)[L - 1 - i])

    return bin[::-1]


def P_flip(P):
    rows = np.shape(P)[0]
    L1 = int(np.log2(rows))
    columns = np.shape(P)[1]
    L2 = int(np.log2(columns))
    P1 = np.zeros((rows, columns), dtype='complex')
    for i in range(rows):
        for j in range(columns):
            P1[int(binarize(i, L2)[::-1], 2),
               int(binarize(j, L1)[::-1], 2)] = P[i, j]

    return P1


class XXZ_model:
    def __init__(self, nqubits, nmagnons, delta):
        self.nqubits = nqubits
        self.nmagnons = nmagnons
        self.delta = delta
        self.roots = None
        self.roots1 = None
        self.Plist = None
        self.circuit = None
        self.Ds = None
        self.Ws = None
        self.Fs = None
        self.indexes = None
        self.k = None

    def get_roots(self, lambdas, type='Rapidities'):
        gamma = np.arccos(self.delta)
        roots = []
        roots1 = []
        for l in lambdas:
            if type == 'Rapidities':
                roots.append(np.sinh(gamma*(l-1j)/2)/np.sinh(gamma*(l+1j)/2))
                roots1.append(1j*np.sin(gamma)/np.sinh(gamma*(l+1j)/2))
            elif type == 'Momentum':
                b = np.exp(1j*l)
                roots.append(b)
                roots1.append(np.sqrt(1+b**2-2*b*self.delta)
                              )
        self.roots = np.array(roots)
        self.roots1 = np.array(roots1)

    def _get_index(self, r):
        def gen_l(l, j):
            l = np.array(l)
            r = len(l)
            l_list = [np.array(l)]
            if j != r-1:
                while l[j] < l[j+1] - 1:
                    l[j] += 1
                    l_list.append(np.array(l))

            else:
                while l[j] < self.nmagnons-1:
                    l[j] += 1
                    l_list.append(np.array(l))
            return l_list

        l = list(range(r))
        j = len(l) - 1
        result1 = gen_l(l, j)
        for j in reversed(range(0, len(l) - 1)):
            r_list = []
            for rr in result1:
                r_list.append(gen_l(rr, j))
            r_list = np.concatenate(r_list)
            result1 = r_list
        return result1

    def get_indexes(self):
        indexes = []
        for r in range(1, self.nmagnons+1):
            indexes.append(self._get_index(r))
        self.indexes = indexes

    def a(self, x, y):
        return 1 + x*y - 2*self.delta*y
    
    def _get_index1(self, r, k):
        def gen_l(l, j):
            l = np.array(l)
            r = len(l)
            l_list = [np.array(l)]
            if j != r-1:
                while l[j] < l[j+1] - 1:
                    l[j] += 1
                    l_list.append(np.array(l))

            else:
                while l[j] < k:
                    l[j] += 1
                    l_list.append(np.array(l))
            return l_list

        l = list(range(r))
        j = len(l) - 1
        result1 = gen_l(l, j)
        for j in reversed(range(0, len(l) - 1)):
            r_list = []
            for rr in result1:
                r_list.append(gen_l(rr, j))
            r_list = np.concatenate(r_list)
            result1 = r_list
        return result1

    def Crk_xy(self, r, k, x, y):
        def get_s(i, j):
            return 1+x[i]*x[j]-2*self.delta*x[j]

        def get_s_conj(i, j):
            return 1+np.conjugate(y[i]*y[j])-2*self.delta*np.conjugate(y[j])

        if r == 0:
            c_rk = 1
        elif k == 0:
            if r == 1:
                c_rk = 1
            else:
                c_rk = 0
        elif k + 1 < r:
            c_rk = 0
        else:
            a_s = list(permutations(list(range(r))))
            b_s = list(permutations(list(range(r))))
            n_s = self._get_index1(r, k)
            c_rk = 0
            for n in n_s:
                c_rk_n = 0
                for a in a_s:
                    c_a = 0
                    for b in b_s:
                        sign = int(LeviCivita(*a))*int(LeviCivita(*b))
                        s = 1
                        for p in range(r):
                            s_p = 1
                            for q in range(r):
                                if p > q:
                                    s_p *= get_s(a[p], a[q]) * \
                                        get_s_conj(b[p], b[q])
                            s *= s_p

                        result = 1
                        for m in range(r):
                            result *= (x[a[m]]*np.conjugate(y[b[m]]))**(n[m])

                        c_a_b = sign*s*result
                        c_a += c_a_b
                    c_rk_n += c_a
                c_rk += c_rk_n
        return c_rk

    def _get_Crk(self, r, k):
        from joblib import Parallel, delayed

        def choose(n, m):
            return np.math.factorial(n)/(np.math.factorial(m)*np.math.factorial(n-m))
        j = int(choose(self.nmagnons, r))

        def _get_Crk(r, k, n, m):
            index = self.indexes[r-1]
            x = index[n]
            y = index[m]
            c_n_m = self.Crk_xy(r, k, self.roots[x], self.roots[y])
            return c_n_m
        C = Parallel(n_jobs=30, prefer='threads')(delayed(_get_Crk)
                                                  (r, k, n, m) for n in range(j) for m in range(j))
        C = np.reshape(C, (j, j))
        return C

    def m_k(self, n):
        return min(n+1, self.nmagnons)

    def _get_Xr(self, r, k):
        if r == 0:
            X = np.array([[1]])
        else:
            dim = int(choose(self.m_k(k), r))
            index = self.indexes[r-1]
            diag = [np.prod(self.roots[index[j]]) for j in range(dim)]
            X = np.diag(diag)
        return X

    def _get_Sr(self, r):
        if r == 0:
            S = np.ones((1, self.nmagnons))
        else:
            dim1 = int(choose(self.nmagnons, r-1))
            dim2 = int(choose(self.nmagnons, r))
            S = np.zeros((dim1, dim2), complex)
            y = self.roots
            for j in range(dim2):
                index_j = self.indexes[r-1][j]
                for i in range(dim1):
                    if r == 1:
                        S[i, j] = 1
                    else:
                        index_i = self.indexes[r-2][i]
                        for n in range(1, r+1):
                            index_jn = np.concatenate(
                                [index_j[0:n-1], index_j[n::]])
                            # print(len(index_jn))
                            # jns = self.indexes[r-2]
                            if (index_jn == index_i).all():
                                delta = 1
                            else:
                                delta = 0
                            ajn = np.prod([self.a(y[s], y[index_j[n-1]])
                                          for s in index_i])
                            S[i, j] += (-1)**(n+1)*delta*ajn
        return S

    def _get_Sreps(self, r, eps, k):
        if eps == 0:
            dim = int(choose(self.m_k(k), r))
            Seps = np.eye(dim, dim)
        elif eps == 1:
            S = self._get_Sr(r)
            rows = int(choose(self.m_k(k), r-1))
            cols = int(choose(self.m_k(k), r))
            Seps = S[0:rows, 0:cols]
        return Seps

    def _get_Ark(self, r, k):
        def choose(n, m):
            return np.math.factorial(n)/(np.math.factorial(m)*np.math.factorial(n-m))
        rows = int(choose(self.m_k(k), r))
        cols = int(choose(self.m_k(k+1), r))
        A = np.zeros((rows, cols), complex)
        c = self._get_Crk(r, k)
        for a in range(rows):
            for b in range(cols):
                c_a_to_b = np.copy(c)
                c_a_to_b[a, :] = c[b, :]
                if a == 0:
                    det_a = 1
                else:
                    det_a = np.linalg.det(c[0:a, 0:a])
                det_aplus1 = np.linalg.det(c[0:a+1, 0:a+1])
                det_aplus1_ab = np.linalg.det(c_a_to_b[0:a+1, 0:a+1])
                A[a, b] = det_aplus1_ab / np.sqrt(det_a*det_aplus1)
        return A

    def _get_Brk(self, r, k):
        def choose(n, m):
            return np.math.factorial(n)/(np.math.factorial(m)*np.math.factorial(n-m))
        j = int(choose(self.m_k(k), r))
        B = np.zeros((j, j), complex)
        c = self._get_Crk(r, k)
        for b in range(j):
            for a in range(b+1):
                if a == b:
                    if a == 0:
                        det_a = 1
                    else:
                        det_a = np.linalg.det(c[0:a, 0:a])
                    det_aplus1 = np.linalg.det(c[0:a+1, 0:a+1])
                    B[a, b] = np.sqrt(det_a/det_aplus1)
                else:
                    c_a_to_b = np.copy(c)
                    c_a_to_b[a, :] = c_a_to_b[b, :]
                    if b == 0:
                        det_b_ab = 1
                        det_b = 1
                    else:
                        det_b_ab = np.linalg.det(c_a_to_b[0:b, 0:b])
                        det_b = np.linalg.det(c[0:b, 0:b])
                    det_bplus1 = np.linalg.det(c[0:b+1, 0:b+1])
                    B[a, b] = - det_b_ab / np.sqrt(det_b*det_bplus1)
        return B

    def _get_Lambda_rkeps(self, r, eps, k):
        X = self._get_Xr(r-eps, k)
        S = self._get_Sreps(r, eps, k)
        Lambda = X@S
        return Lambda

    def _get_Prkeps(self, r, eps, k, a, b):
        A = self._get_Ark(r-eps, k-1)
        X = self._get_Xr(r-eps, k)
        S = self._get_Sreps(r, eps, k)
        B = self._get_Brk(r, k)
        Preps = X@S
        if a:
            Preps = A@Preps
        if b:
            Preps = Preps@B

        return Preps

    def _get_index_elements(self):
        elements = []
        n = self.nmagnons
        for i in range(1, 2**(n+1)-1):
            ones = bin(i).count('1')
            column = []
            for j in range(1, 2**(n+1)):
                if bin(j).count('1') == ones:
                    column.append((i, j))
            elements.append(column)
        return elements

    def _rowcol_to_ijkeps(self):
        elements = self._get_index_elements()
        indexes = {r: {0: [], 1: []} for r in range(1, self.nmagnons+1)}
        for k, index in enumerate(elements):
            row = k + 1
            r = bin(row).count('1')
            if format(row, f'0{self.nmagnons}b')[-1] == '1':
                eps = 1
            else:
                eps = 0
            indexes[r][eps].append(index)
        return indexes

    def get_Pk_matrix(self, k, a, b):
        self.k = k

        def choose(n, m):
            return np.math.factorial(n)/(np.math.factorial(m)*np.math.factorial(n-m))
        n = min(self.k, self.nmagnons)
        P = np.zeros((2**(n+1), 2**(n+1)), complex)
        P[0, 0] = 1
        indexes = self._rowcol_to_ijkeps()
        for r in range(1, self.m_k(k)+1):
            range_i = [choose(self.m_k(self.k), r),
                       choose(self.m_k(self.k), r-1)]
            for eps in range(2):
                if r == k + 1 and eps == 0:
                    pass
                else:
                    Preps = self._get_Prkeps(r, eps, k, a, b)
                    for i in range(int(range_i[eps])):
                        for j in range(int(choose(self.m_k(self.k), r))):
                            element = indexes[r][eps][i][j]
                            if element[0] < 2**(n+1) and element[1] < 2**(n+1):
                                P[element] = Preps[i, j]
        return P

    def get_Lmabda_k_matrix(self, k):
        self.k = k

        def choose(n, m):
            return np.math.factorial(n)/(np.math.factorial(m)*np.math.factorial(n-m))
        n = min(self.k, self.nmagnons)
        L = np.zeros((2**(self.nmagnons+1), 2**self.nmagnons), complex)
        L[0, 0] = 1
        indexes = self._rowcol_to_ijkeps()
        for r in range(1, self.m_k(k)+1):
            range_i = [choose(self.m_k(self.k), r),
                       choose(self.m_k(self.k), r-1)]
            for eps in range(2):
                if r == k + 1 and eps == 0:
                    pass
                else:
                    Lambda_reps = self._get_Lambda_rkeps(r, eps, k)
                    for i in range(int(range_i[eps])):
                        for j in range(int(choose(self.m_k(self.k), r))):
                            element = indexes[r][eps][i][j]
                            if element[0] < 2**(self.nmagnons+1) and element[1] < 2**self.nmagnons:
                                L[element] = Lambda_reps[i, j]
        return L
