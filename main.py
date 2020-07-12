import pandas as pd
import numpy as np
import least_squares as ls
import matplotlib.pyplot as plt
import k_means as km

if __name__ == "__main__":
    save_path = 'toy_data.tsv' 
    data = pd.read_csv(save_path,delimiter = '\t', header = None)
    labels = data[0].to_numpy()
    labels = np.reshape(labels,(np.size(labels), 1))
    x = data[[1,2]].to_numpy()
    # x1 = np.reshape(x[:,0],(np.size(labels), 1))
    # x2 = np.reshape(x[:,1],(np.size(labels), 1))
    # print(x[:,0])
    ones = np.ones((len(data),1))
    A = np.concatenate((ones,x), 1)
    theta = ls.least_squares(A, labels)
    theta_0 = theta[0]
    theta = np.delete(theta, 0, axis = 0)
    print(theta, theta_0)
    # plt.scatter(x[0:100,0], x[0:100,1])
    # plt.scatter(x[100:,0], x[100:,1])
    # axes = plt.gca()
    # x_vals = np.array(axes.get_xlim())
    # y_vals = theta_0 + (-1*theta[0]) * x_vals
    # plt.plot(x_vals, y_vals, '--')
    # plt.show()
    # z = np.dot(theta, x1[120]) + theta_0
    # print(z)
    theta_numpy, res, rank, s = np.linalg.lstsq(A, labels, rcond= None)
    print(theta_numpy)
    km.k_means(x, labels)