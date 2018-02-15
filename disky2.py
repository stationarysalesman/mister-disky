from DiscUtils import *

# Trajectory paths
dcdPath = '../dec-cg/ND_combined_10.dcd'
psfPath = '../dec-cg/ND.psf'
dt = 20e-15 * 1000 * 10

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
        
        # X points for graphing 
        #xs[i] = dt * i 
        xs[i] = dt * i / (1e-9) 

    init_pos = z_axis_positions[0]
    #n = 1
    #z_angles_smoothed = z_axis_angles[::n]
    z_axis_angles = np.zeros(len(z_axis_positions)) 
    for i in range(1,len(z_axis_positions)):
        z_axis_angles[i] = np.arccos(np.vdot(z_axis_positions[i], init_pos))
    #z_deltheta = np.absolute(np.diff(z_axis_angles))
    f = open('thetavstime.csv', 'w')
    f.write('time,theta\n')
    for i,theta in enumerate(z_axis_angles):
        f.write('{},{}\n'.format(i*dt,theta))
    f.close() 

if __name__ == '__main__':
    main()

