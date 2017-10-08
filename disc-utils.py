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
    theta = np.pi / 4
    # Rotation matrix for negative z-axis
    R = np.array((
        (1,0,0),
        (0,-np.cos(theta),np.sin(theta)),
        (0,-np.sin(theta),-np.cos(theta)))) 
    v = np.matmul(R, y)
    cross = np.cross(v,y)
    x = np.divide(cross, np.linalg.norm(cross))
    z = np.cross(x,y)
    return (x, y, z)

xGlobal = np.array((1,0,0))
yGlobal = np.array((0,1,0))
zGlobal = np.array((0,0,1))

def calcRotation(x,y,z):
    """Calculate the angle between a local coordinate frame, defined by x,y,z, 
        and a global coordinate frame defined by x=(1,0,0), y=(0,1,0), and 
        z=(0,0,1)."""
    thetaX = np.arccos(np.vdot(x, xGlobal))
    thetaY = np.arccos(np.vdot(y, yGlobal))
    thetaZ = np.arccos(np.vdot(z, zGlobal))

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

# test reading
u = mda.Universe('ND.psf','ND.dcd')
selection_str = 'resname DLPE DMPC DPPC GPG LPPC PALM PC PGCL POPC POPE'
lipids = u.select_atoms(selection_str, updating=True)
dt = 10e-12
xs = np.zeros(len(u.trajectory))
xangles = np.zeros(len(u.trajectory))
yangles = np.zeros(len(u.trajectory))
zangles = np.zeros(len(u.trajectory))
for i,ts in enumerate(u.trajectory):
    x,y,z = calcAxes(lipids.positions)
    xangles[i] = np.arccos(np.vdot(x, xGlobal))
    yangles[i] = np.arccos(np.vdot(y, yGlobal))
    zangles[i] = np.arccos(np.vdot(z, zGlobal))
    xs[i] = dt * i 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xs, xangles, color='red')
ax.plot(xs, yangles, color='green')
ax.plot(xs, zangles, color='blue')
plt.show()   

xRotDisp = np.sum(np.absolute(np.diff(xangles)))
yRotDisp = np.sum(np.absolute(np.diff(yangles)))
zRotDisp = np.sum(np.absolute(np.diff(zangles)))
delT = dt * len(u.trajectory)
delTdelX = np.divide(delT, xRotDisp)
delTdelY = np.divide(delT, yRotDisp)
delTdelZ = np.divide(delT, zRotDisp)
print('Rotational correlation times (x,y,z): {} {} {}'.format(delTdelX,
                                                              delTdelY,
                                                              delTdelZ))
