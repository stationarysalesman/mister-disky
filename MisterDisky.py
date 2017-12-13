from DiscUtils import *

# Trajectory paths
dcdPath = '../dec-cg/ND_combined_10.dcd'
psfPath = '../dec-cg/ND.psf'
dt = 20e-15 * 1000 * 10

def main():
    u = mda.Universe(psfPath, dcdPath)

    # hybrid atoms
    """ 
    x1 = u.select_atoms('bynum 1771', updating=True)
    x2 = u.select_atoms('bynum 2791', updating=True)
    y1 = u.select_atoms('bynum 3211', updating=True)
    y2 = u.select_atoms('bynum 2247', updating=True)
    xs = np.zeros(len(u.trajectory))
    """
    # coarse grain atoms 
    x1 = u.select_atoms('bynum 21', updating=True)
    x2 = u.select_atoms('bynum 232', updating=True)
    y1 = u.select_atoms('bynum 117', updating=True)
    y2 = u.select_atoms('bynum 347', updating=True)

    xs = np.zeros(len(u.trajectory))
    pitch = np.zeros(len(u.trajectory))
    yaw = np.zeros(len(u.trajectory))
    roll = np.zeros(len(u.trajectory))
    for i,ts in enumerate(u.trajectory):
        # Get local coordinate axes 
        x1_pos = x1[0].position
        x2_pos = x2[0].position
        x = (x1_pos - x2_pos) / (np.linalg.norm(x1_pos - x2_pos))
        y1_pos = y1[0].position
        y2_pos = y2[0].position
        y = (y1_pos - y2_pos) / (np.linalg.norm(y1_pos - y2_pos))
        z = np.cross(x, y)
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
        #xs[i] = dt * i 
        xs[i] = dt * i / (1e-9) 

    # Smooth by only taking every nth point
    n = 100 
    """ 
    xs_smooth = [np.sum(xs[i:i+n])/n for i in range(len(xs)-n)]
    roll_smooth = [np.sum(roll[i:i+n])/n for i in range(len(xs)-n)]
    pitch_smooth = [np.sum(pitch[i:i+n])/n for i in range(len(xs)-n)]
    yaw_smooth = [np.sum(yaw[i:i+n])/n for i in range(len(xs)-n)]
    """ 
    xs_smooth = xs[::n]
    roll_smooth = roll[::n]
    pitch_smooth = pitch[::n]
    yaw_smooth = yaw[::n]
    print('num points: {}'.format(len(xs_smooth))) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs_smooth, roll_smooth, color='red', label='Roll Angle')
    ax.plot(xs_smooth, pitch_smooth, color='green', label='Pitch Angle')
    ax.plot(xs_smooth, yaw_smooth, color='blue', label='Yaw Angle')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()

    rolldelThetas = np.absolute(np.diff(roll_smooth))
    pitchdelThetas = np.absolute(np.diff(pitch_smooth))
    yawdelThetas = np.absolute(np.diff(yaw_smooth))
   
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(xs_smooth[1:], rolldelThetas, color='red', label='Roll $\Delta \\theta$')
    ax2.plot(xs_smooth[1:], pitchdelThetas, color='green', label='Pitch $\Delta \\theta$')
    ax2.plot(xs_smooth[1:], yawdelThetas, color='blue', label='Yaw $\Delta \\theta$')
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2, labels2) 
    plt.show()

    rollRotDisp = np.sum(rolldelThetas)
    pitchRotDisp = np.sum(pitchdelThetas)
    yawRotDisp = np.sum(yawdelThetas)
    delT = dt * len(u.trajectory)
    delTdelRoll = np.divide(delT, rollRotDisp)
    delTdelPitch = np.divide(delT, pitchRotDisp)
    delTdelYaw = np.divide(delT, yawRotDisp)
    print('Roll rotational correlation time: {}'.format(delTdelRoll))
    print('Pitch rotational correlation time: {}'.format(delTdelPitch))
    print('Yaw rotational correlation time: {}'.format(delTdelYaw))

    # Autocorrelation function
    rhat_roll = autocorrelation(rolldelThetas)
    rhat_pitch = autocorrelation(pitchdelThetas)
    rhat_yaw = autocorrelation(yawdelThetas)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(xs_smooth[1:], rhat_roll, color='red', label='Roll Autocorrelation')
    ax3.plot(xs_smooth[1:], rhat_pitch, color='green', label='Pitch Autocorrelation')
    ax3.plot(xs_smooth[1:], rhat_yaw, color='blue', label='Yaw Autocorrelation')
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(handles3, labels3)
    
    
    # Show the plot
    plt.show()   


if __name__ == '__main__':
    main()

