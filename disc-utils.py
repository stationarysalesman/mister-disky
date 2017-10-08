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

def calcAxes(points):
    """Calculate a local coordinate frame for an object given points in a plane         of that object."""
    y = calcNormal(points)
    print "y is: " + str(y)
    theta = np.pi / 4
    # Rotation matrix for negative z-axis
    R = np.array((
        (1,0,0),
        (0,-np.cos(theta),np.sin(theta)),
        (0,-np.sin(theta),-np.cos(theta)))) 
    v = np.matmul(R, y)
    print v
    cross = np.cross(v,y)
    x = np.divide(cross, np.linalg.norm(cross))
    z = np.cross(x,y)
    return (x, y, z)

# test
"""
points = np.array(((1,0,0),(0,1,0),(0,0,0)))
x,y,z = calcAxes(points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs,ys,zs = np.hsplit(points,[1,2])
ax.scatter(xs,ys,zs)
ax.plot([0,x[0]],[0,x[1]],[0,x[2]])
ax.plot([0,y[0]],[0,y[1]],[0,y[2]], color='red')
ax.plot([0,z[0]],[0,z[1]],[0,z[2]], color='green')
plt.show()
"""
