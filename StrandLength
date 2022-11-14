import datetime
now = datetime.datetime.now()
print("Date and time ",str(now))
#
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
%matplotlib inline
#
def poisson_func(k, lam):
    '''
    Poisson distribution function.
    Given number of occurences and mean, returns distribution function value.
    '''
    poiss = lam**k*np.exp(-lam)/sp.special.factorial(k)
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
    breaks = np.zeros(n_breaks)
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
plt.show
#
then = now
now = datetime.datetime.now()
print(" ")
print("Date and time",str(now))
print("Time since last check is",str(now - then))
