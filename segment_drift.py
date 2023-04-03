import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.special import factorial
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

import datetime
now = datetime.datetime.now()
print("Date and time ",str(now))
#
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.special import factorial
%matplotlib inline
#
def poisson_func(k, lam):
    '''
    Poisson distribution function.
    Given number of occurences and mean, returns distribution function value.
    '''
    poiss = lam**k*np.exp(-lam)/sp.special.factorial(k, exact=True)
    return poiss
#
# Number of (single) DNA strands
n_strands = 92
#
# Length of DNA strand
L_DNA = 1e8 # base pairs
#
# Average number of breaks per strand, the number of segments generated and their lengths
mean_breaks = 20
n_segs = np.zeros(n_strands).astype(int)
seg_lengths = np.zeros((n_strands, (mean_breaks + 5*np.sqrt(mean_breaks)).astype(int)))
#
# Set up histgram of number of breaks
num_range = np.zeros(2)
num_range[0] = np.maximum(mean_breaks - 5*np.sqrt(mean_breaks), 0.0).astype(int)
num_range[1] = (mean_breaks + 5*np.sqrt(mean_breaks)).astype(int)
n_num_bins = ((num_range[1] - num_range[0])).astype(int)
num_bins, num_bin_wid = np.linspace(num_range[0], num_range[1], n_num_bins + 1, retstep = True)
num_bin_cents = 0.5*(num_bins[0:n_num_bins] + num_bins[1:n_num_bins + 1])
num_bin_err = num_bin_wid/np.sqrt(12)*np.ones(n_num_bins)
cum_num_hist = np.zeros(n_num_bins)
#
# Set up histogram of segment lengths
n_len_bins = 40
len_bins, len_bin_wid = np.linspace(0.0, 10*L_DNA/mean_breaks, n_len_bins + 1, retstep = True)
len_bin_cents = 0.5*(len_bins[0:n_len_bins] + len_bins[1:n_len_bins + 1])
len_bin_err = len_bin_wid/np.sqrt(12)*np.ones(n_len_bins)
cum_len_hist = np.zeros(n_len_bins)
#
# Initialise random number generator
rng = np.random.default_rng()
#
# Simulate breaks and segements for all strands
for n in range(0, n_strands):
    n_breaks = rng.poisson(mean_breaks)
    breaks = np.zeros(n_breaks)*L_DNA
    n_segs[n] = n_breaks + 1
    breaks = rng.uniform(0.0, 1.0, n_breaks)
    ends = np.zeros(n_segs[n] + 1)
    ends[1:n_segs[n]] = np.sort(breaks)
    ends[n_segs[n]] = 1.0
    seg_lengths[n, 0:n_segs[n]] = L_DNA*(ends[1:n_segs[n] + 1] - ends[0:n_segs[n]])
    #
    num_hist_here, _ = np.histogram(n_breaks, num_bins)
    cum_num_hist += num_hist_here
    #
    len_hist_here, _ = np.histogram(seg_lengths[n, 0:n_segs[n]], len_bins)
    cum_len_hist += len_hist_here
#
# Function describing distribution of number of breaks
n_poiss_plot = 100
k_poiss_plot = np.linspace(num_range[0], num_range[1], n_poiss_plot)
poiss_func_plot = n_strands*num_bin_wid*poisson_func(mean_breaks, k_poiss_plot)
#
# Function describing segment length distribution
len_func_plot = (mean_breaks*(mean_breaks + 1)*n_strands*
                 len_bin_wid/L_DNA*np.exp(-mean_breaks*len_bin_cents/L_DNA))
#
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
#
ax[0].set_title('Broken strand numbers')
ax[0].errorbar(num_bin_cents, cum_num_hist, xerr = num_bin_err, yerr = np.sqrt(cum_num_hist),
               linestyle = '', marker = '+', color = 'b')
ax[0].plot(k_poiss_plot, poiss_func_plot, linestyle = '-', marker = '', color = 'r')
ax[0].set_xlabel('Number')
ax[0].set_ylabel('Relative frequency')
ax[0].set_ylim(0.0, 1.1*(np.amax(cum_num_hist) + np.amax(np.sqrt(cum_num_hist))))
ax[0].grid(color = 'g')
#
ax[1].set_title('Broken strand lengths')
ax[1].errorbar(len_bin_cents, cum_len_hist, xerr = len_bin_err, yerr = np.sqrt(cum_len_hist),
               linestyle = '', marker = '+', color = 'b')
ax[1].plot(len_bin_cents, len_func_plot, linestyle = '-', marker = '', color = 'r')
ax[1].set_xlabel('Length (bp)')
ax[1].set_ylabel('Relative frequency')
ax[1].set_xlim(0.0, L_DNA)
ax[1].set_ylim(0.5, 1.1*(np.amax(cum_len_hist) + np.amax(np.sqrt(cum_len_hist))))
ax[1].grid(color = 'g')
#
ax[2].set_title('Broken strand lengths')
ax[2].errorbar(len_bin_cents, cum_len_hist, xerr = len_bin_err, yerr = np.sqrt(cum_len_hist),
               linestyle = '', marker = '+', color = 'b')
ax[2].plot(len_bin_cents, len_func_plot, linestyle = '-', marker = '', color = 'r')
ax[2].set_xlabel('Length (bp)')
ax[2].set_ylabel('Relative frequency')
ax[2].set_xlim(0.0, L_DNA)
ax[2].set_ylim(0.5, 1.5*(np.amax(cum_len_hist) + np.amax(np.sqrt(cum_len_hist))))
ax[2].set_yscale('log')
ax[2].grid(color = 'g')
#
plt.tight_layout()
#plt.savefig('Segmentvsdrift.png')
plt.show
#
then = now
now = datetime.datetime.now()
print(" ")
print("Date and time",str(now))
print("Time since last check is",str(now - then))

from mpl_toolkits.mplot3d import Axes3D
#
use_sphere = True
#
# Radius of nucleus
rad_nuc = 2.0 # microns
rad_nuc_min = 1.5 # microns
sig_rad_nuc = 0.5 # microns
#
# Radius of cell
rad_cell = 10.0 # microns
rad_cell_min = 16.0 # microns
sig_rad_cell = 1.0 # microns
#
# Colot table for plots
n_color_tab = 8
color_tab = np.array(['r', 'orange', 'y', 'g', 'c', 'b', 'm', 'k'])
#
fig = plt.figure(figsize = (10, 8))
#
ax2d_cell = fig.add_subplot(2, 2, 1)
ax2d_cell.set_title("Initial positions of strands - cell")
ax2d_cell.set_xlabel("x")
ax2d_cell.set_ylabel("y")
#
ax3d_cell = fig.add_subplot(2, 2, 2, projection='3d')
ax3d_cell.set_title("Initial positions of strands - cell")
ax3d_cell.set_xlabel("x")
ax3d_cell.set_ylabel("y")
ax3d_cell.set_zlabel("z")
#
ax2d_pic = fig.add_subplot(2, 2, 3)
ax2d_pic.set_title("Initial positions of strands - image")
ax2d_pic.set_xlabel("x")
ax2d_pic.set_ylabel("y")
#
ax3d_pic = fig.add_subplot(2, 2, 4, projection='3d')
ax3d_pic.set_title("Initial positions of strands - image")
ax3d_pic.set_xlabel("x")
ax3d_pic.set_ylabel("y")
ax3d_pic.set_zlabel("z")
#
# Image dimensions
n_rows = 1040 # number of rows of pixels (y coord)
n_cols = 1392 # number of columns of pixels (x coord)
depth = 20
#
# Set up number of cells and their positions
n_cells = 15
x_cell = np.zeros(n_cells)
y_cell = np.zeros(n_cells)
z_cell = np.zeros(n_cells)
#
# Set location of cells
if n_cells == 1:
    x_cell[0] = n_cols/2
    y_cell[0] = n_rows/2
    z_cell[0] = depth/2
else:
    x_cell = np.random.uniform(0, n_cols, n_cells)
    y_cell = np.random.uniform(0, n_rows, n_cells)
    z_cell = np.random.uniform(0, depth, n_cells)

#
# Colors for plotting strands
colors = np.zeros(n_strands).astype(str)
#
for n_cell in range(0, n_cells):
    #
    # Simulate uniform radial and phi initial distribution of strands
    rad_arr = np.sqrt(np.random.uniform(0, max(np.random.normal(rad_cell, sig_rad_cell, 1),
                                              rad_cell_min)**2, n_strands))
    phi_arr = np.random.uniform(0, 2*np.pi, n_strands)
    #
    # Determine initial coordinates of strands (simulate uniform intial theta distribution if required)
    if use_sphere:
        theta_arr = np.arccos(np.random.uniform(-1, 1, n_strands))
        x_arr = x_cell[n_cell] + rad_arr*np.sin(theta_arr)*np.cos(phi_arr)
        y_arr = y_cell[n_cell] + rad_arr*np.sin(theta_arr)*np.sin(phi_arr)
        z_arr = z_cell[n_cell] + rad_arr*np.cos(theta_arr)
    else:
        x_arr = x_cell[n_cell] + rad_arr*np.cos(phi_arr)
        y_arr = y_cell[n_cell] + rad_arr*np.sin(phi_arr)
        z_arr = x_cell[n_cell] + np.random.uniform(0, depth, n_strands)
    #
    # Plot intial positions of all strands in image
    col_inds = (np.linspace(0, n_strands - 1, n_strands)%n_color_tab).astype(int)
    ax2d_pic.scatter(x_arr, y_arr, color = color_tab[col_inds], s = 1.0)
    ax3d_pic.scatter(x_arr, y_arr, z_arr, color = color_tab[col_inds], s = 1.0)
    #
    if n_cell > 0:
        continue
    #
    # Plot initial positions of strands in one cell 
    ax2d_cell.scatter(x_arr - x_cell[n_cell], y_arr - y_cell[n_cell], color = color_tab[col_inds], s = 10.0)
    ax3d_cell.scatter(x_arr - x_cell[n_cell], y_arr - y_cell[n_cell], 
                      z_arr - z_cell[n_cell], color = color_tab[col_inds], s = 10.0)    
#
scale_cell = 2.0
ax2d_cell.set_xlim(-scale_cell*rad_cell, scale_cell*rad_cell)
ax2d_cell.set_ylim(-scale_cell*rad_cell, scale_cell*rad_cell)
#
ax3d_cell.set_xlim(-scale_cell*rad_cell, scale_cell*rad_cell)
ax3d_cell.set_ylim(-scale_cell*rad_cell, scale_cell*rad_cell)
ax3d_cell.set_zlim(-scale_cell*rad_cell, scale_cell*rad_cell)
#
expand_pic = 0.2
ax2d_pic.set_xlim(-expand_pic*n_cols, (1 + expand_pic)*n_cols)
ax2d_pic.set_ylim(-expand_pic*n_cols, (1 + expand_pic)*n_rows)
#
ax3d_pic.set_xlim(-expand_pic*n_cols, (1 + expand_pic)*n_cols)
ax3d_pic.set_ylim(-expand_pic*n_cols, (1 + expand_pic)*n_rows)
ax3d_pic.set_zlim(-(1 + expand_pic)*rad_cell, depth + (1 + expand_pic)*rad_cell)
#
plt.tight_layout()
#plt.savefig('3Dimages.png')
plt.show()

import datetime
now = datetime.datetime.now()
print("Date and time ",str(now))
#
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#
def vDagarose(L, E):
    '''
    Given length of DNA strand (in kbp) and electric field (in V/cm) returns 
    velocity in agarose (in cm/s * 10^4 or equivalently microns/s).
    '''
    if not hasattr(vDagarose, "kB"):
        vDagarose.kB, vDagarose.kC, vDagarose.kL, vDagarose.alpha = 2.24, 0.56, 0.59, 1.34
        print(" ")
        print("Parameters used to descibe vD in agarose")
        print("kB = {:.2f}".format(vDagarose.kB))
        print("kC = {:.2f}".format(vDagarose.kC))
        print("kL = {:.2f}".format(vDagarose.kL))
        print("alpha = {:.2f}".format(vDagarose.alpha))
    #
    vD = vDagarose.kB*E/(1 + vDagarose.kC*L) + vDagarose.kL*E**vDagarose.alpha
    #
    return vD
#
# Calculate fitted function values x direction
nPlot = 100
xBot = 0.1
xTop = 1000.0
xPlot = np.exp(np.linspace(np.log(xBot), np.log(xTop), nPlot))
nY = 5
yVals = np.array([0.05, 0.15, 0.3, 0.5, 0.8]) # V/cm
#
# Calculate fitted function values y direction
yBot = 0.0
yTop = 1.0
yPlot = np.linspace(yBot, yTop, nPlot)
nX = 11
xVals = np.array([0.31, 0.59, 0.86, 1.07, 1.28, 1.97, 2.24, 4.28, 6.59, 9.31, 23.01]) # kbp
#
# Plot data
nColTab = 6
colTab = ['b', 'r', 'c', 'm', 'y', 'k']
#
fig = plt.figure(figsize = (16, 6))
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Velocity model')
ax.set_xlabel('Length (kbp)')
ax.set_ylabel('Velocity (cm/s $\\times$ $10^4$)')
nCol = 0
for plot in range(nY, 0, -1):
    fitPlotX = vDagarose(xPlot, yVals[plot - 1]*np.ones(nPlot))
    ax.plot(xPlot, fitPlotX, linestyle = '-', color = colTab[nCol], label = "E = " + str(yVals[plot - 1]) + " V/cm")
    nCol += 1
    if nCol > nColTab - 1:
        nCol = 0
ax.set_xlim(xBot, xTop)
ax.set_ylim(0, 2.2)
ax.set_xscale('log')
ax.grid(color = 'g')
ax.legend()
#
ax = fig.add_subplot(1, 2, 2)
ax.set_title('Velocity model')
ax.set_xlabel('Field (V/cm)')
ax.set_ylabel('Velocity (cm/s $\\times$ $10^4$)')
nCol = 0
for plot in range(0, nX):
    fitPlotY = vDagarose(xVals[plot]*np.ones(nPlot), yPlot)
    ax.plot(yPlot, fitPlotY, linestyle = '-', color = colTab[nCol], label = "Len = " + str(xVals[plot]) + " kbp")
    nCol += 1
    if nCol > nColTab - 1:
        nCol = 0
ax.set_xlim(yBot, yTop)
ax.set_ylim(0, 2.2)
ax.grid(color = 'g')
ax.legend()
#
# plt.savefig("FitPlot.png")
plt.show()
#
then = now
now = datetime.datetime.now()
print(" ")
print("Date and time",str(now))
print("Time since last check is",str(now - then))

E_field = 1.0 # V/cm
drift_time = 25*60
n_strands = 1
L_DNA = 1e8
mean_breaks = 20

n_segs = np.zeros(n_strands).astype(int)

for n in range(0, n_strands):
    n_breaks = rng.poisson(mean_breaks)
    n_segs[n] = n_breaks + 1
    seg_lengths = np.zeros(mean_breaks + 5*np.sqrt(mean_breaks).astype(int))
    drift_dist = np.zeros(max(n_segs))
    breaks = np.zeros(n_breaks)
    breaks = rng.uniform(0.0, 1.0, n_breaks)
    ends = np.zeros(n_segs[n] + 1)
    ends[1:n_segs[n]] = np.sort(breaks)
    ends[n_segs[n]] = 1.0
    seg_lengths[0:n_segs[n]] = L_DNA*(ends[1:n_segs[n] + 1] - ends[0:n_segs[n]])
    list_drift_dist = drift_dist[0:n_segs[n]]
    list_seg_lengths = seg_lengths[0:n_segs[n]]

    drift_dist[0:n_segs[n]] = vDagarose(seg_lengths[0:n_segs[n]]/1000, E_field)*drift_time # um

    bp_to_micro = 1e6*0.04/1e8
    y_offset = np.arange(n_segs[n])
    right = list_drift_dist
    left = np.maximum(right - list_seg_lengths*bp_to_micro, 0.0)
    for n_seg in range(0, len(list_drift_dist)):
        plt.plot([left[n_seg], right[n_seg]], [y_offset[n_seg], y_offset[n_seg]])
    plt.title('Segment vs drift distance')
    plt.xlabel("Segment position ($\mu$m)")
    plt.ylabel("y_offset")
    plt.show()
