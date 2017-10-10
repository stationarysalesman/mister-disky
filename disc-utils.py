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


def calcXAxis(lipids, helixResidues):
    """Calculate a faux-X-axis; that is, a vector pointing from the 
    center of mass of the lipids to the center of mass of a single 
    helical turn in the MSP."""
    lipidR = calcCenterOfMass(lipids)
    helixR = calcCenterOfMass(helixResidues)
    x = (helixR - lipidR) 
    return x / np.linalg.norm(x)
    
def calcCenterOfMass(atoms):
    """Calculate the center of mass of a given set of atoms."""
    M = 0.0
    s = np.zeros((3)) 
    for a in atoms:
        s += a.mass * a.position 
        M += a.mass
    s /= M
    return s
 
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
lipid_str = 'resname DLPE DMPC DPPC GPG LPPC PALM PC PGCL POPC POPE'
helix_str = 'bynum 1352:1389' # a single turn of MSP1, near x-axis at start
lipids = u.select_atoms(lipid_str, updating=True)
helixResidues = u.select_atoms(helix_str, updating=True)
dt = 10e-12
xs = np.zeros(len(u.trajectory))
pitch = np.zeros(len(u.trajectory))
yaw = np.zeros(len(u.trajectory))
roll = np.zeros(len(u.trajectory))
for i,ts in enumerate(u.trajectory):
    # Get local coordinate axes 
    z = calcNormal(lipids.positions)
    x = calcXAxis(lipids, helixResidues)
    y = np.cross(z, x)
    
    # Compute the yaw, i.e. rotation of x-axis about the global y-axis
    xhat = np.copy(x)
    xhat[1] = 0.0 # project onto xz-plane
    xhat /= np.linalg.norm(xhat)
    yaw[i] = np.arccos(np.vdot(xhat, xGlobal)) # a dot b = |a||b|cos(theta)
   
    # Compute the pitch, i.e. rotation of z-axis about the global x-axis 
    zhat = np.copy(z)
    zhat[0] = 0.0 # project onto zy-plane
    zhat /= np.linalg.norm(zhat)
    pitch[i] = np.arccos(np.vdot(zhat, zGlobal))
    
    # Compute the roll, i.e. rotation of y-axis about the global z-axis
    yhat = np.copy(y)
    yhat[2] = 0.0 # project onto xy-plane
    yhat /= np.linalg.norm(yhat)
    roll[i] = np.arccos(np.vdot(yhat, yGlobal))
   
    # X points for graphing 
    xs[i] = dt * i 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xs, roll, color='red', label='Roll Angle')
ax.plot(xs, pitch, color='green', label='Pitch Angle')
ax.plot(xs, yaw, color='yellow', label='Yaw Angle')
plt.show()   

"""
xRotDisp = np.sum(np.absolute(np.diff(xangles)))
yRotDisp = np.sum(np.absolute(np.diff(yangles)))
delT = dt * len(u.trajectory)
delTdelX = np.divide(delT, xRotDisp)
delTdelY = np.divide(delT, yRotDisp)
"""

