from DiscUtils import *

# Trajectory paths
dcdPath = '../dec-cg/ND_combined_10.dcd'
psfPath = '../dec-cg/ND.psf'
dt = 20e-15 * 1000 * 10

def P_2(v, t):
    # the second order Legendre polynomial
    intervals = len(v) / t
    total = 0.0
    for t_0 in range(intervals - 1):
        total += (np.vdot(v[t_0], v[t_0+t]) ** 2)
    avg = total / (intervals - 1)
    return (3/2.) * avg - (1/2.)

def main():
    u = mda.Universe(psfPath, dcdPath)
    nframes = len(u.trajectory)
    # coarse grain atoms 
    x1 = u.select_atoms('bynum 21', updating=True)
    x2 = u.select_atoms('bynum 232', updating=True)
    y1 = u.select_atoms('bynum 117', updating=True)
    y2 = u.select_atoms('bynum 347', updating=True)

    xs = np.zeros(nframes)
    x_axis_positions = np.zeros((nframes, 3))
    y_axis_positions = np.zeros((nframes, 3))
    z_axis_positions = np.zeros((nframes, 3))
    u_positions = np.zeros((nframes, 3))
    for i,ts in enumerate(u.trajectory):
        # Get local coordinate axes 
        x1_pos = x1[0].position
        x2_pos = x2[0].position
        x = (x1_pos - x2_pos) / (np.linalg.norm(x1_pos - x2_pos))
        y1_pos = y1[0].position
        y2_pos = y2[0].position
        y = (y1_pos - y2_pos) / (np.linalg.norm(y1_pos - y2_pos))
        z = np.cross(x, y) 
        
        x_axis_positions[i] = x 
        y_axis_positions[i] = y 
        z_axis_positions[i] = z / np.linalg.norm(z) 
       
        # The unit vector is attached to one of the x-axis points
        # and subtends a 45 degree angle with the horizon
        tvec = (x + y) / np.linalg.norm(x+y)
        tpos = x1_pos + tvec 
        u_positions[i] = (tpos - x1_pos) / np.linalg.norm(tpos - x1_pos)

        # X points for graphing 
        #xs[i] = dt * i 
        xs[i] = dt * i / (1e-9) 

    n = 400
    v = np.zeros(n)
    v[0] = 0
    for t in range(1,n):
        v[t] = P_2(u_positions,t) 
    f = open('t_vs_p2.csv', 'w')
    f.write('time,p2\n')
    for i,val in enumerate(v):
        f.write('{},{}\n'.format(i*dt,val))
    f.close() 

if __name__ == '__main__':
    main()

