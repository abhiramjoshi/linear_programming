#gram_schmidt algorithm for determining a orthonormal for set a1,a2...an
#the algortihm takes in input set of vectors and outputs a set of equal lenght of orthonormal vectors.
#Alogorithm will terminate early if vector set a is linearly dependant
import numpy as np
import matplotlib.pyplot as plt

def randomvectors():
    x = np.random.randint(1,100)
    y = np.random.randint(1,100)
    #a = np.random.uniform(-x,x,(2,2))   
    a = np.array([[2,-2,18],[2,1,0],[1,2,0]])
    a = a.astype(float)
    return a

def orthogonalize(a):
    a = a.astype(float)
    k = np.size(a,0)
    n = np.size(a,1)
    q = np.zeros((k,n))
    qi = np.zeros((1,n))
    term = False
    for i in range(k):
        qi = a[:,i].copy()
        for j in range(i):
            qi -= np.dot(q[j], a[i])*q[j]
            #qi = removefloat(qi)
            if np.all(abs(qi) < 1e-15, axis = 0):
                print('Given Vectors are linearly dependant')
                term = True
                break
        if term == True:
            break
        qi = normalize(qi)
        q[:,i] = qi
    return q

def normalize(x):
    mag = np.linalg.norm(x)
    x_norm = x/mag
    return x_norm    

def removefloat(a):
    for i in range(np.size(a,0)):
        a[i] = round(a[i],2)
        # if abs(a[i])<1e-15:
        #     a[i]=0
    return a

def plotvectors(x):
    if np.size(x, 1) == 2:
        origin = [0], [0] # origin point
        plt.quiver(*origin, x[:,0], x[:,1], angles='xy', scale_units='xy', scale=1, color = ['r','g'])
        max_x = np.max(x[:,0])+1
        max_y = np.max(x[:,1])+1
        if max_x > max_y:
            gen_max = max_x
        else:
            gen_max = max_y
        plt.xlim(-gen_max,gen_max)
        plt.ylim(-gen_max,gen_max)
        plt.show()
    else:
        pass    

def remove_zero_row(data):
    data = data[~np.all(data == 0, axis=1)]
    return data    

def gram_matrix(a):
    q = orthogonalize(a)
    return q


if __name__ == '__main__':
    a = randomvectors()
    print(a)
    plotvectors(a)
    q = orthogonalize(a)
    plotvectors(q)
    print(remove_zero_row(q))
    #print(np.transpose(q))
    plt.show()