from DiscUtils import *
from scipy import optimize

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
        z_axis_positions[i] = z 
        
        # X points for graphing 
        #xs[i] = dt * i 
        xs[i] = dt * i / (1e-9) 

    # The autocorrelation of a unit vector, defined as the correlation of 
    # the angle it makes with its initial position, at time dt
    x_initial_pos = x_axis_positions[0]/np.linalg.norm(x_axis_positions[0]) 
    y_initial_pos = y_axis_positions[0]/np.linalg.norm(y_axis_positions[0]) 
    z_initial_pos = z_axis_positions[0]/np.linalg.norm(z_axis_positions[0]) 

    x_axis_angles = np.zeros((nframes))
    y_axis_angles = np.zeros((nframes))
    z_axis_angles = np.zeros((nframes))
    for i in range(nframes):
        x_axis_angles[i] = np.arccos(np.vdot( \
            x_axis_positions[i]/np.linalg.norm(x_axis_positions[i]), \
            x_initial_pos))
        y_axis_angles[i] = np.arccos(np.vdot( \
            y_axis_positions[i]/np.linalg.norm(y_axis_positions[i]), \
            y_initial_pos))
        z_axis_angles[i] = np.arccos(np.vdot( \
            z_axis_positions[i]/np.linalg.norm(z_axis_positions[i]), \
            z_initial_pos))

    print('x angles: {}'.format(x_axis_angles))
    print('y angles: {}'.format(y_axis_angles))
    print('z angles: {}'.format(z_axis_angles))


    # Smooth by only taking every nth point
    n = 1 
    xs_smooth = xs[::n]
    x_angles_smooth = x_axis_angles[::n]
    y_angles_smooth = y_axis_angles[::n]
    z_angles_smooth = z_axis_angles[::n]
    print('num points: {}'.format(len(xs_smooth))) 
    
    # Autocorrelation function
    x_autocorr = autocorrelation(x_angles_smooth)
    y_autocorr = autocorrelation(y_angles_smooth)
    z_autocorr = autocorrelation(z_angles_smooth)
    
    # Fit autocorrelation to exponential decays
    def z_function(x):
        diff = 0
        for i, ac in enumerate(z_autocorr):
            val = np.exp(-2 * x * (i * dt))
            diff += (ac - val)
        return np.absolute(diff)

    #minimum = optimize.fmin(z_function, 1e-7)
    #print('min is {}'.format(minimum))
 
    # Plot autocorrelation
    xs_graph = xs_smooth
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(xs_graph, x_autocorr, color='red', label='X Autocorrelation')
    ax3.plot(xs_graph, y_autocorr, color='green', label='Y Autocorrelation')
    ax3.plot(xs_graph, z_autocorr, color='blue', label='Z Autocorrelation')
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(handles3, labels3)
    
    
    # Show the plot
    plt.show()   


if __name__ == '__main__':
    main()

