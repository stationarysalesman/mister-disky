import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import MDAnalysis as mda

def calcNormal(points):
    """Use least-squares regression to get a plane from a set of points. 
        Return the normal of that plane."""
    A,z = np.hsplit(points, [2])
    l = len(points)
    ones = np.ones(l).reshape(l, 1)
    A = np.hstack((A,ones))
    x,residuals,rank,s = np.linalg.lstsq(A, z)
    x = np.append(x[:2], 1)
    n = np.divide(x, np.linalg.norm(x))
    return n


# test
points = np.array(((1,0,0),(0,1,0),(0,0,0)))
n = calcNormal(points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs,ys,zs = np.hsplit(points,[1,2])
ax.scatter(xs,ys,zs)
ax.scatter(n[0],n[1],n[2])
plt.show()
