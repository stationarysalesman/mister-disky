from statsmodels.graphics.tsaplots import plot_acf
from pandas import Series
from matplotlib import pyplot as plt
import numpy as np

series = Series.from_csv('thetavstime.csv', header=0)
ac = np.zeros(len(series))
for i in range(len(ac)):
    ac[i] = series.autocorr(i)

ac = np.log(ac)
l = [x for x in range(len(ac))]
xs = np.array(l, dtype=float)
xs = xs * 2e-10
n=300
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xs[:n], ac[:n], color='blue', label='Autocorrelation')
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles,labels)
plt.show()

from scipy.stats import linregress
slope,intercept,r_value,p_value,std_error = linregress(xs[:n],ac[:n])
print('Slope: {}'.format(slope))
print('intercept: {}'.format(intercept))
print('r^2: {}'.format(np.power(r_value,2)))
