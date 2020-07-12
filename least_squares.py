import numpy as np
import qr_fact as qr

def least_squares(A,b):
    Q, R = qr.qr_fact(A)
    Q_t = np.transpose(Q)
    n = np.size(b,1)
    x = np.zeros((np.size(A,1),n))
    for i in range(n):
        b_Q = np.dot(Q_t, b[:,i])
        x[:, i] = qr.backsub(b_Q ,R)
    return x

if __name__ == "__main__":
    A = np.array([[0.97,1.86, 0.41],[1.23, 2.18, 0.53],[0.8,1.24,0.62],[1.29,0.98,0.51],[1.1, 1.23, 0.69],[0.67,0.34,0.54],[0.87,0.26,0.62],[1.1,0.16,0.48],[1.92,0.22,0.71],[1.29,0.12,0.62]])
    b1 = np.ones((np.size(A,0),1))*10**3
    b2 = np.ones((np.size(A,0),1))*20**3
    b = np.concatenate((b1,b2), axis = 1)
    x = least_squares(A,b)
    print(x)