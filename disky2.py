from DiscUtils import *
import matplotlib.pyplot as plt


# Trajectory paths
dcdPath = '../Current_D1_trajectory/D1_combined_pbc_fixed.dcd'
psfPath = '../Current_D1_trajectory/D1_combined.psf'
dt = 2e-10 

def P_2(v, t):
    # the second order Legendre polynomial
    intervals = len(v) - t
    total = 0.0
    for t_0 in range(intervals):
        total += (np.vdot(v[t_0], v[t_0+t]) ** 2)
    avg = total / (intervals)
    return (3/2.) * avg - (1/2.)


def cos2theta_fun(initpos, pos):
    return np.power(np.vdot(initpos,pos),2)


def main():
    u = mda.Universe(psfPath, dcdPath)
    nframes = len(u.trajectory)
    # coarse grain atoms 
    x1 = u.select_atoms('bynum 21', updating=True)
    x2 = u.select_atoms('bynum 232', updating=True)
    y1 = u.select_atoms('bynum 121', updating=True)
    y2 = u.select_atoms('bynum 332', updating=True)

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
        tvec = (x + y + z) / np.linalg.norm(x + y + z)
        tpos = x1_pos + tvec 
        u_positions[i] = (tpos - x1_pos) / np.linalg.norm(tpos - x1_pos)
        u_positions[i][0] = -u_positions[i][0]
        # X points for graphing 
        #xs[i] = dt * i 
        xs[i] = dt * i / (1e-9) 

    print('initial pos of unit vector: {}'.format(u_positions[0]))
    print('initial pos of z-axis: {}'.format(z_axis_positions[0]))
    n = 6000 
    v = np.zeros(n)
    v[0] = 0
    theta = np.arccos(np.vdot(u_positions[0], z_axis_positions[0]))
    cos2theta = np.power(np.cos(theta), 2)
    sin2theta = np.power(np.sin(theta), 2) 
    sin4theta = np.power(np.sin(theta), 4)
    alpha1 = (1/4.) * np.power((3 * cos2theta-1), 2)
    alpha2 = 3 * sin2theta * cos2theta
    alpha3 = (3/4.) * sin4theta
    initpos = u_positions[0]
    for t in range(1,n):
        v[t] = P_2(u_positions,t) 

    
    # Fit to sum of exponentials
    def exponential_model(t,tau1, tau2, tau3):
        return alpha1 * np.exp(-t/tau1) + alpha2 * np.exp(-t/tau2) + alpha3 * np.exp(-t/tau3)

    v_fit = v[1:] # fitting cutoffs 
    def fitfun(taus):
        l = len(v_fit)
        s = 0.0
        for t in range(l):
           s += np.power(v_fit[t]-exponential_model(t*dt,taus[0],taus[1],taus[2]), 2)
        s /= l 
        return np.sqrt(s)

    from scipy import optimize
    minimum = optimize.minimize(fitfun, [1e-7, 1e-7, 1e-7], method='Nelder-Mead', options={'xatol':1e-11})
    print('Parameters of minimization: {}'.format(minimum.x))
    print('Alphas: {}, {}, {}'.format(alpha1,alpha2,alpha3))
    # Make some magic pictures
    fig = plt.figure()
    ax = fig.add_subplot(111)
    final_taus = minimum.x
    xs = np.zeros(len(v_fit))
    fit_ys = np.zeros(len(v_fit))
    calc_ys = np.zeros(len(v_fit))
    for i,val in enumerate(v_fit):
        input_val = i * dt
        fit_ys[i] = exponential_model(input_val, final_taus[0], final_taus[1], final_taus[2])
        calc_ys[i] = val 
        xs[i] = input_val 
    ax.plot(xs, calc_ys, 'b', xs, fit_ys, 'r')
    plt.show()
    
 
    f = open('t_vs_p2.csv', 'w')
    f.write('time,p2\n')
    for i,val in enumerate(v):
        f.write('{},{}\n'.format(i*dt,val))
    f.close() 

if __name__ == '__main__':
    main()

