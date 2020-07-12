import numpy as np
import gram_sch as gm

def qr_fact(A):
    """
    QR Factorization of matrix A
    """
    Q = gm.orthogonalize(A)
    R = R_matrix(A,Q)
    return Q,R
    
def R_matrix(A,Q):
    """
    Creating the R matrix in a QR facotrization
    """
    #n = np.size(A,0)
    n = np.size(A,1)
    R = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j<i:
                pass
            else:
                R[i,j] = np.dot(A[:,j], Q[:,i])
    return R

def backsub(b, R):
    """
    Backsub algorithm used for finding the value of x in the eqauation Ax = b
    """
    n = np.size(b,0)
    x = np.zeros(n)
    for i in reversed(range(n)):
        sub = 0
        for j in reversed(range(n)):
            sub += R[i,j]*x[j]
        x[i] = (b[i]-sub)/R[i,i]
    return x

if __name__ == "__main__":
    #A = np.array([[2,-2,18],[2,1,0],[1,2,0]])
    #A = np.array([[2,2,0],[1,3,1],[2,1,1]])
    #A = np.array([[2,0],[-1,1],[0,2]])
    A = np.array([[0.97,1.86, 0.41],[1.23, 2.18, 0.53],[0.8,1.24,0.62],[1.29,0.98,0.51],[1.1, 1.23, 0.69],[0.67,0.34,0.54],[0.87,0.26,0.62],[1.1,0.16,0.48],[1.92,0.22,0.71],[1.29,0.12,0.62]])
    b = np.ones(np.size(A,0))*10**3

    Q, R = qr_fact(A)

    print(A)
    print("R = ")
    print(R)
    print("Q = ")
    print(Q)
    Q_t = np.transpose(Q)
    print(np.dot(Q,R))
    # Q_t = np.transpose(Q)
    b_Q = np.dot(Q_t, b)
    x = backsub(b_Q ,R)
    print(x)