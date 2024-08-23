import os
import argparse
import json
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.utils_infodynamics import detrending_method
import yaml
from utils.experiments import Experiment
import json
from tqdm import tqdm


def normalize(x, lower=-1, upper=1):
    """ transform x to x_ab in range [a, b]
    """
    x_norm = (upper - lower)*((x - np.min(x)) / (np.max(x) - np.min(x))) + lower
    return x_norm

def adaptive_filter(y, span=56):
    #if len(y) % 2:
    #   y=y[:-1]

    w = int(4 * np.floor(len(y)/span) + 1)
    y_dt = np.mat([float(j) for j in y])
    _, y_smooth = detrending_method(y_dt, w, 1)
    
    return y_smooth.T

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

    return ax

def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.
    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers
    """ 
    if ax is None:
        ax = plt.gca()

    bootindex = sp.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = np.polyfit(xs, ys + resamp_resid, 1)                   
        # Plot bootstrap cluster
        ax.plot(xs, np.polyval(pc, xs), "r-", linewidth=2, alpha=3.0 / float(nboot))

    return ax

def adaptiveline(x1, x2, outpath="adaptline.png"):
    _, ax = plt.subplots(2,1,figsize=(14,6),dpi=300)
    c = ["g", "r", "b"]
    ax[0].plot(normalize(x1, lower=0),c="gray")
    for i, span in enumerate([128, 56, 32]):
        n_smooth = normalize(adaptive_filter(x1, span=span), lower=0)
        ax[0].plot(n_smooth,c=c[i])
    ax[0].set_ylabel("$\\mathbb{N}ovelty$", fontsize=14)
    
    ax[1].plot(normalize(x2, lower=-1),c="gray")
    for i, span in enumerate([128, 56, 32]):
        r_smooth = normalize(adaptive_filter(x2, span=span), lower=-1)
        ax[1].plot(r_smooth,c=c[i])
    ax[1].set_ylabel("$\\mathbb{R}esonance$", fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath)
    #plt.close()



def move_avg(l,n = 5): 
    cumsum, moving_aves = [0], []
    for i, x in enumerate(l, 1):
        cumsum.append(cumsum[i-1] + x)
        if i >= n:
            moving_ave = (cumsum[i] - cumsum[i-n])/n
            moving_aves.append(moving_ave)
    return moving_aves
# NOTES: consider padding?


def gaussian_kernel(arr, sigma=False, fwhm=False):
    y_vals = np.array(arr) 
    x_vals = np.arange(arr.shape[0])
    if sigma == fwhm:
        print("[INFO] Define parameters \u03C3 xor FWHM")
        pass
    elif fwhm:
        sigma = fwhm / np.sqrt(8 * np.log(2))
    else:
        sigma = sigma
        fwhm = sigma * np.sqrt(8 * np.log(2))
    
    print("[INFO] Applying Gaussian kernel for \u03C3 = {} and FWHM = {} ".format(round(sigma,2), round(fwhm,2)))
    
    smoothed_vals = np.zeros(y_vals.shape)
    for x_position in tqdm(x_vals):
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_vals[x_position] = sum(y_vals * kernel)
    
    return smoothed_vals

def plot_signal(signal, smoothed_vals_adaptive, smoothed_vals_avg, outpath: str):
    
    fig = plt.figure(figsize= (12,6))
    ax1 = fig.add_subplot(221)
    ax1.set_title("novelty")   
    ax1.plot(signal, markersize=8, alpha=0.6)
    ax2 = fig.add_subplot(222)
    ax2.set_title("smoothed (adaptive filter)")
    ax2.plot(smoothed_vals_adaptive, markersize=8, alpha=0.4)
    ax3 = fig.add_subplot(223)
    ax3.set_title("zero-centered")
    ax3.bar(range(len(signal)), signal-np.mean(signal))
    ax4 = fig.add_subplot(224)
    ax4.set_title("moving avg")
    ax4.plot(smoothed_vals_avg, markersize=8, alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath) 









##########################################


# Load specific run of the experiment
with open(
    os.path.join(
        "configs",
        "infodynamics_configs",  # Important that this is infodynamics_configs
        "distilbert-base-uncased-MNLI_infodynamics_config.yaml",  # Has to contain a timestamp!
    ),
    "r",
) as file:
    experiment_config = yaml.safe_load(file)

experiment = Experiment(**experiment_config)


NTR_folder_path = os.path.join(experiment.current_run_dir, f"NTR-results_window-{experiment.window}_step_cutoff-{experiment.step_cutoff}")

# Load NTR results
with open(os.path.join(NTR_folder_path, f"NTR-results_window-{experiment.window}_step_cutoff-{experiment.step_cutoff}.json"), "r") as fin:
    ntr_results = json.load(fin)



resonance = ntr_results["resonance"]
novelty = ntr_results["novelty"]
transience = ntr_results["transience"]



signal_name = "novelty"
moving_avg_n = 10000
adaptive_filter_span = 56


smoothed_novelty_adaptive = adaptive_filter(novelty, span=adaptive_filter_span)
smoothed_resonance_adaptive = adaptive_filter(resonance, span=adaptive_filter_span)
plt.plot(smoothed_novelty_adaptive)
plt.plot(smoothed_resonance_adaptive)

smoothed_novelty_avg = move_avg(novelty, n=moving_avg_n)
smoothed_resonance_avg = move_avg(resonance, n=moving_avg_n)
plt.plot(smoothed_novelty_avg)
plt.plot(smoothed_resonance_avg)



plot_signal(novelty, smoothed_novelty_adaptive, smoothed_novelty_avg, outpath=os.path.join(NTR_folder_path, f"{signal_name}_moving_avg-{moving_avg_n}_adaptive_filter-{adaptive_filter_span}.png"))

adaptiveline(novelty, resonance, outpath=os.path.join(NTR_folder_path, f"{signal_name}_adaptive_filter.png"))


#########
######### need to change the adaptiveline plot to only incluce the adaptive filter with span 56 (and maybe also the moving average with n=10000, but faded out) and the original signal (most faded)
#########




# Plot the signal and the smoothed signal in the same plot
fig = plt.figure(figsize=(12,6),dpi=300)
plt.plot(normalize(novelty, lower=0), alpha=0.1)
plt.plot(normalize(smoothed_novelty_avg, lower=0), label=f"Moving average (n={moving_avg_n})", alpha=0.5)
plt.plot(normalize(smoothed_novelty_adaptive, lower=0) , label=f"Adaptive filter (span={adaptive_filter_span})")
plt.ylabel("$\\mathbb{N}ovelty$ (normalized)")
plt.legend()
plt.savefig(os.path.join(NTR_folder_path, f"novelty_w_smoothed_signals_adaptive_filter-{adaptive_filter_span}_moving_avg-{moving_avg_n}.png"))


_, ax = plt.subplots(2,1,figsize=(14,6),dpi=300)
ax[0].plot(normalize(novelty, lower=0),c="gray", alpha = 0.5)
#ax[0].plot(normalize(smoothed_novelty_avg, lower=0))
ax[0].plot(normalize(smoothed_novelty_adaptive, lower=0))
ax[0].set_ylabel("$\\mathbb{N}ovelty$", fontsize=14)

ax[1].plot(normalize(resonance, lower=-1),c="gray", alpha = 0.5)
#ax[1].plot(normalize(smoothed_resonance_avg, lower=0))
ax[1].plot(normalize(smoothed_resonance_adaptive, lower=-1))
ax[1].set_ylabel("$\\mathbb{R}esonance$", fontsize=14)
plt.tight_layout()
plt.savefig(outpath)
#plt.close()





############### CHECKING N VALUES FOR MOVING AVERAGE
# smoothed_signal_avg_100 = move_avg(novelty, n=100)
# smoothed_signal_avg_1000 = move_avg(novelty, n=1000)
# smoothed_signal_avg_10000 = move_avg(novelty, n=10000)
# smoothed_signal_avg_100000 = move_avg(novelty, n=100000)
# # plot them in subfigures
# fig = plt.figure(figsize=(12,6),dpi=300)
# ax1 = fig.add_subplot(221)
# ax1.plot(normalize(novelty), alpha=0.3)
# ax1.plot(normalize(smoothed_signal_avg_100), label=f"n=100")
# ax1.set_ylabel("$\\mathbb{N}ovelty$ (normalized)")
# ax1.legend()
# ax2 = fig.add_subplot(222)
# ax2.plot(normalize(novelty), alpha=0.3)
# ax2.plot(normalize(smoothed_signal_avg_1000), label=f"n=1000")
# ax2.set_ylabel("$\\mathbb{N}ovelty$ (normalized)")
# ax2.legend()
# ax3 = fig.add_subplot(223)
# ax3.plot(normalize(novelty), alpha=0.3)
# ax3.plot(normalize(smoothed_signal_avg_10000), label=f"n=10000")
# ax3.set_ylabel("$\\mathbb{N}ovelty$ (normalized)")
# ax3.legend()
# ax4 = fig.add_subplot(224)
# ax4.plot(normalize(novelty), alpha=0.3)
# ax4.plot(normalize(smoothed_signal_avg_100000), label=f"n=100000")
# ax4.set_ylabel("$\\mathbb{N}ovelty$ (normalized)")
# ax4.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(NTR_folder_path, f"{signal_name}_moving_avg.png"))




############### CHECKING SPAN VALUES FOR ADAPTIVE FILTER
smoothed_signal_adaptive_128 = adaptive_filter(novelty, span=128)
smoothed_signal_adaptive_56 = adaptive_filter(novelty, span=56)
smoothed_signal_adaptive_32 = adaptive_filter(novelty, span=32)


fig = plt.figure(figsize=(12,6),dpi=300)
plt.plot(normalize(novelty), alpha=0.3)
plt.plot(normalize(smoothed_novelty_avg), label=f"Moving average (n={moving_avg_n})")
plt.plot(normalize(smoothed_signal_adaptive_32) , label=f"Adaptive filter (span=32)")
plt.plot(normalize(smoothed_signal_adaptive_56), label=f"Adaptive filter (span=56)")
plt.plot(normalize(smoothed_signal_adaptive_128), label=f"Adaptive filter (span=128)")
plt.ylabel("$\\mathbb{N}ovelty$ (normalized)")
plt.legend()

fig = plt.figure(figsize=(12,6), dpi=300)
plt.plot(normalize(smoothed_novelty_avg), label=f"Moving average (n={moving_avg_n})")
plt.plot(normalize(smoothed_signal_adaptive_32) , label=f"Adaptive filter (span=32)")
plt.ylabel("$\\mathbb{N}ovelty$ (normalized)")
plt.legend()
plt.savefig(os.path.join(NTR_folder_path, "novelty_adaptive_filter_32.png"))

fig = plt.figure(figsize=(12,6), dpi=300)
plt.plot(normalize(smoothed_novelty_avg), label=f"Moving average (n={moving_avg_n})")
plt.plot(normalize(smoothed_signal_adaptive_56), label=f"Adaptive filter (span=56)")
plt.ylabel("$\\mathbb{N}ovelty$ (normalized)")
plt.legend()
plt.savefig(os.path.join(NTR_folder_path, "novelty_adaptive_filter_56.png"))

fig = plt.figure(figsize=(12,6), dpi=300)
plt.plot(normalize(smoothed_novelty_avg), label=f"Moving average (n={moving_avg_n})")
plt.plot(normalize(smoothed_signal_adaptive_128), label=f"Adaptive filter (span=128)")
plt.ylabel("$\\mathbb{N}ovelty$ (normalized)")
plt.legend()
plt.savefig(os.path.join(NTR_folder_path, "novelty_adaptive_filter_128.png"))







