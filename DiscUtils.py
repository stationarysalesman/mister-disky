import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import MDAnalysis as mda

# Globals / constants
xGlobal = np.array((1,0,0))
yGlobal = np.array((0,1,0))
zGlobal = np.array((0,0,1))


def autocorrelation(v):
    """Compute the autocorrelation of a function, given by the 
    discrete values in vector v."""
    nvals = len(v) 
    autocorr = list() 
    for deltaT in range(4000):
        s = 0.0
        for i in range(nvals-deltaT):
            s += np.vdot(v[i], v[i+deltaT])
        s /= (nvals-deltaT) 
        autocorr.append(s)
    return autocorr


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
 
def calcRotation(x,y,z):
    """Calculate the angle between a local coordinate frame, defined by x,y,z, 
        and a global coordinate frame defined by x=(1,0,0), y=(0,1,0), and 
        z=(0,0,1)."""
    thetaX = np.arccos(np.vdot(x, xGlobal))
    thetaY = np.arccos(np.vdot(y, yGlobal))
    thetaZ = np.arccos(np.vdot(z, zGlobal))

