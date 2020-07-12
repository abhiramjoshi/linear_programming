import numpy as np

def orthogonalize(A):
    A_f = A.astype(float)
    n = np.size(A,1) 
    Q = np.zeros_like(A_f,float)
    qi = np.zeros(n)
    term = False
    for i in range (n):
        qi = A_f[:,i]
        for j in reversed(range(i)): 
            qi -= np.dot(A_f[:,i],Q[:,j])*Q[:,j]
            if np.all(abs(qi) < 1e-14, axis = 0):
                print('Given Vectors are linearly dependant')
                term = True
                break
        if term == True:
            break
        qi = qi/np.linalg.norm(qi)    
        Q[:,i] = qi
    return Q

if __name__ == "__main__":
    #a = np.array([[2,-2,18],[2,1,0],[2,1,0],[1,2,1]])
    a = np.array([[-3,-4],[4,6],[1,1]])
    Q = orthogonalize(a)
    print(Q)
