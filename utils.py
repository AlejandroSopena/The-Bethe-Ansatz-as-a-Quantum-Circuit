import numpy as np

def unitarize_pink(P):
    cols=np.shape(P)[1]
    for k in range(int(cols/2),cols):
        P[:,k] = np.random.rand((cols))
        for j in range(k):
            P[:,k]=P[:,k]-np.dot(P[:,k],P[:,j].T.conj())*P[:,j]/np.linalg.norm(P[:,j])**2  
        P[:,k]=P[:,k]/np.linalg.norm(P[:,k])
    return P

def check_unitariry(P):
    check = np.allclose(np.eye(len(P)), P.dot(P.T.conj()),rtol=1e-05, atol=1e-08)
    return check