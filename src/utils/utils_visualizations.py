"""
Different algorithms for visualization of time series data. 
Taken and adapted from https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/news_uncertainty.py commit commit 1fb16bc
"""
import os
import json
import numpy as np
from numpy import *
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    ax.fill_between(x2, y2 + ci, y2 - ci, color="green", edgecolor="green")

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

    bootindex = np.random.randint

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
    #plt.savefig(outpath)
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

def plot_signal(signal, smoothed_vals_adaptive, smoothed_vals_avg, outpath: str, title: str):
    
    fig = plt.figure(figsize= (12,6))
    # add title to whole figure
    plt.title(title)
    ax1 = fig.add_subplot(221)
    ax1.set_title("novelty", fontsize=10)   
    ax1.plot(signal, markersize=8, alpha=0.6)
    ax2 = fig.add_subplot(222)
    ax2.set_title("smoothed (adaptive filter)", fontsize=10)
    ax2.plot(smoothed_vals_adaptive, markersize=8, alpha=0.4)
    ax3 = fig.add_subplot(223)
    ax3.set_title("zero-centered", fontsize=10)
    ax3.bar(range(len(signal)), signal-np.mean(signal))
    ax4 = fig.add_subplot(224)
    ax4.set_title("moving avg", fontsize=10)
    ax4.plot(smoothed_vals_avg, markersize=8, alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath) 



def regline(x, y, title: str, bootstrap=True):
    p, cov = np.polyfit(x, y, 1, cov=True)
    slope, intercept = p
    slope_se = np.sqrt(cov[0, 0])  
    y_model = np.polyval(p, x)
    # statistics
    n = y.size
    m = p.size
    dof = n - m
    t = stats.t.ppf(0.975, n - m)
    # estimates of error
    resid = y - y_model                           
    chi2 = np.sum((resid / y_model)**2) 
    chi2_red = chi2 / dof
    s_err = np.sqrt(np.sum(resid**2) / dof)    
    slope_ci_lower = slope - t*slope_se
    slope_ci_upper = slope + t*slope_se
    print(f"SE = {slope_se}")
    print(f"95% CIs [{slope_ci_lower}, {slope_ci_upper}]")
    # plot
    fig, ax = plt.subplots(figsize=(8, 7.5),dpi=300)
    ax.plot(x, y, ".", color="coral", markersize=8,markeredgewidth=1, markeredgecolor="coral", markerfacecolor="None")
    ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="$\\beta_1 = {}$".format(round(p[0], 2)))
    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = np.polyval(p, x2)
    # # confidence interval option
    # if bootstrap:
    #     plot_ci_bootstrap(x, y, resid, ax=ax)
    # else:
    #     plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
    # prediction interval
    pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))   
    ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    ax.plot(x2, y2 - pi, "--", color="0.5", label="95% PI")
    ax.plot(x2, y2 + pi, "--", color="0.5")
    # borders
    ax.spines["top"].set_color("0.5")
    ax.spines["bottom"].set_color("0.5")
    ax.spines["left"].set_color("0.5")
    ax.spines["right"].set_color("0.5")
    ax.get_xaxis().set_tick_params(direction="out")
    ax.get_yaxis().set_tick_params(direction="out")
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    # labels
    plt.suptitle(f"{title}", fontsize="22")
    plt.xlabel("Novelty$_z$", fontsize="20")
    plt.ylabel("Resonance$_z$", fontsize="20")
    plt.xlim(np.min(x) - .25, np.max(x) + .25)
    # custom legend
    handles, labels = ax.get_legend_handles_labels()
    display = (0, 1)
    anyArtist = plt.Line2D((0, 1), (0, 0), color="#ea5752")
    legend = plt.legend(
        [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
        [label for i, label in enumerate(labels) if i in display], # + ["95% CI"]
        loc=5, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=2, mode="expand", fontsize=20
    )  
    frame = legend.get_frame().set_edgecolor("0.5")
    mpl.rcParams['axes.linewidth'] = 1
    # save figure
    plt.tight_layout()



def detrending_coeff(win_len , order):

#win_len = 51
#order = 2
	n = (win_len-1)/2
	A = mat(ones((win_len,order+1)))
	x = np.arange(-n , n+1)
	for j in range(0 , order + 1):
		A[:,j] = mat(x ** j).T

	coeff_output = (A.T * A).I * A.T
	return coeff_output , A


def detrending_method(data , seg_len , fit_order) :
	nrows,ncols = shape(data)
	if nrows < ncols :
		data = data.T

	# seg_len = 1001,odd number
	nonoverlap_len = int((seg_len - 1) / 2)
	data_len = shape(data)[0]
	# calculate the coefficient,given a window size and fitting order
	coeff_output , A = detrending_coeff(seg_len , fit_order)
	A_coeff = A * coeff_output

	for seg_index in range(1 , 2) :
		#left trend
		#seg_index = 1

		xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) , seg_index * (seg_len - 1) + 2)
		xi_left = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		left_trend = (A_coeff * seg_data).T

		# mid trend

		if seg_index * (seg_len - 1) + 1 + nonoverlap_len > data_len :
			xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) + nonoverlap_len , data_len + 1)
			xi_mid = mat(xi)
			xi_max = xi.max()
			xi_min = xi.min()
			seg_data = data[xi_min - 1 : xi_max , 0]
			nrows_seg = shape(seg_data)[0]

			if nrows_seg < seg_len :
				coeff_output1 , A1 = detrending_coeff(nrows_seg , fit_order)
				A_coeff1 = A1 * coeff_output1
				mid_trend = (A_coeff1 * seg_data).T
			else :
				mid_trend = (A_coeff * seg_data).T

			xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
			xx2 = mid_trend[0 , 0 : int((seg_len + 1) / 2)]
			w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
			xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

			record_x = xi_left[0 , 0 : nonoverlap_len]
			record_y = left_trend[0 , 0 : nonoverlap_len]
			mid_start_index = mat([(j) for j in range(shape(xi_mid)[1]) if xi_mid[0 , j] == xi_left[0 , shape(xi_left)[1] - 1] + 1])
			nrows_mid = shape(mid_start_index)[0]
			mid_start_index = mid_start_index[0 , 0]

			if nrows_mid == 0 :
				record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 3) / 2)-1 : shape(xi_left)[1]]))
				record_y = hstack((record_y , xx_left[0 , 1 : shape(xx_left)[1]]))
			else :
				record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 1)/ 2 )-1 : shape(xi_left)[1]] , xi_mid[0 , mid_start_index : shape(xi_mid)[1]]))
				record_y = hstack((record_y , xx_left[0 : shape(xx_left)[1]] , mid_trend[0 , int((seg_len + 3) / 2) - 1 : shape(mid_trend)[1]]))

			detrended_data = data - record_y.T

			return  detrended_data, record_y

		else :
			xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) + nonoverlap_len , seg_index * (seg_len - 1) + nonoverlap_len + 2)
			xi_mid = mat(xi)
			xi_max = xi.max()
			xi_min = xi.min()
			seg_data = data[xi_min-1 : xi_max , 0]
			nrows_seg = shape(seg_data)[0]
			mid_trend = (A_coeff * seg_data).T

		#right trend

			if (seg_index + 1) * (seg_len - 1) + 1 > data_len :
				xi = np.arange(seg_index * (seg_len - 1) + 1 , data_len + 1)
				xi_right = mat(xi)
				xi_max = xi.max()
				xi_min = xi.min()
				seg_data = data[xi_min - 1 : xi_max , 0]
				nrows_seg = shape(seg_data)[0]

				if nrows_seg < seg_len :
					coeff_output1 , A1 = detrending_coeff(nrows_seg , fit_order)
					A_coeff1 = A1 * coeff_output1
					right_trend = (A_coeff1 * seg_data).T
				else :
					right_trend = (A_coeff * seg_data).T

				xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
				xx2 = mid_trend[0 , 0 : int((seg_len + 1) / 2)]
				w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
				xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

				xx1 = mid_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
				xx2 = right_trend[0 , 0 : int((seg_len + 1) / 2)]
				w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
				xx_right = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

				record_x = xi_left[0 , 0 : nonoverlap_len]
				record_y = left_trend[0 , 0 : nonoverlap_len]

				record_x = np.hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 1) / 2) - 1 : shape(xi_left)[1]] , xi_mid[0 , int((shape(xi_mid)[1] + 1) / 2) : shape(xi_mid)[1]]))
				record_y = hstack((record_y , xx_left[0 , 0 : shape(xx_left)[1]] , xx_right[0 , 1 : shape(xx_right)[1]]))

				right_start_index = mat([(j) for j in range(shape(xi_right)[1]) if xi_right[0 , j] == xi_mid[0 , shape(xi_mid)[1] - 1] + 1])
				right_start_index =right_start_index[0 , 0]
				record_x = hstack((record_x,xi_right[0 , right_start_index : shape(xi_right)[1]]))
				record_y = hstack((record_y,right_trend[0 , right_start_index : shape(right_trend)[1]]))
				detrended_data = data - record_y.T

				return  detrended_data , record_y

			else :
				xi = np.arange(seg_index * (seg_len - 1) + 1 , (seg_index + 1) * (seg_len - 1) + 2)
				xi_right = mat(xi)
				xi_max = xi.max()
				xi_min = xi.min()
				seg_data = data[xi_min - 1 : xi_max,0]
				right_trend = (A * coeff_output * seg_data).T

				xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
				xx2 = mid_trend[0 , 0 : int((seg_len + 1) / 2)]
				w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
				xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

				xx1 = mid_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
				xx2 = right_trend[0 , 0 : int((seg_len + 1) / 2)]
				w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
				xx_right = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

				record_x = xi_left[0 , 0 : nonoverlap_len]
				record_y = left_trend[0 , 0 : nonoverlap_len]

				record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 1) / 2) - 1 : shape(xi_left)[1]] , xi_mid[0 , int((shape(xi_mid)[1] + 1) /2 ) : shape(xi_mid)[1]]))
				record_y = hstack((record_y , xx_left[0 , 0 : shape(xx_left)[1]] , xx_right[0 , 1 : shape(xx_right)[1]]))


	for seg_index in range(2 , int((data_len - 1) / (seg_len - 1))) :
		#left_trend
		#seg_index = 1
		xi = np.arange((seg_index - 1) * (seg_len - 1) + 1 , seg_index * (seg_len - 1) + 2)
		xi_left = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		left_trend = (A_coeff * seg_data).T

		# mid trend

		xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) + nonoverlap_len , seg_index * (seg_len -1) + nonoverlap_len + 2)
		xi_mid = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		mid_trend = (A_coeff * seg_data).T

		# right trend

		xi = np.arange(seg_index * (seg_len - 1) + 1 , (seg_index + 1) * (seg_len - 1) + 2)
		xi_right = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		right_trend = (A_coeff * seg_data).T

		xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
		xx2 = mid_trend[0 , 0 : int((seg_len + 1) / 2)]
		w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
		xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

		xx1 = mid_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
		xx2 = right_trend[0 , 0 : int((seg_len + 1) / 2)]
		w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
		xx_right = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

		record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 3) / 2) - 1 : shape(xi_left)[1]] , xi_mid[0 , int((shape(xi_mid)[1] + 1) / 2) : shape(xi_mid)[1]]))
		record_y = hstack((record_y , xx_left[0 , 1 : shape(xx_left)[1]] , xx_right[0 , 1 : shape(xx_right)[1]]))

#last part of data

	for seg_index in range(int((data_len - 1) / (seg_len - 1)) , int((data_len - 1) / (seg_len - 1)) + 1) :
	# left trend
	#seg_index = 1

		xi = np.arange((seg_index - 1) * (seg_len - 1) + 1 , seg_index * (seg_len - 1) + 2)
		xi_left = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		left_trend = (A_coeff * seg_data).T

		# mid trend

		if seg_index * (seg_len - 1) + 1 + nonoverlap_len > data_len :
			xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) + nonoverlap_len , data_len+ 1)
			xi_mid = mat(xi)
			xi_max = xi.max()
			xi_min = xi.min()
			seg_data = data[xi_min - 1 : xi_max , 0]
			nrows_seg = shape(seg_data)[0]

			if nrows_seg < seg_len :
				coeff_output1 , A1 = detrending_coeff(nrows_seg , fit_order)
				A_coeff1 = A1 * coeff_output1
				mid_trend = (A_coeff1 * seg_data).T
			else :
				mid_trend = (A_coeff * seg_data).T

			xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
			xx2  =mid_trend[0 , 0 : int((seg_len + 1) / 2)]
			w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
			xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )
			mid_start_index = mat([(j) for j in range(shape(xi_mid)[1]) if xi_mid[0 , j] == xi_left[0 , shape(xi_left)[1] - 1] + 1])
			nrows_mid = shape(mid_start_index)[0]
			mid_start_index = mid_start_index[0 , 0]

			if nrows_mid == 0 :

				record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 3) / 2) - 1 : shape(xi_left)[1]]))
				record_y = hstack((record_y , xx_left[0 , 1 : shape(xx_left)[1]]))

			else :
				record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 3) / 2) - 1 : shape(xi_left)[1]] , xi_mid[0 , mid_start_index : shape(xi_mid)[1]]))
				record_y = hstack((record_y , xx_left[0 , 1 : shape(xx_left)[1]] , mid_trend[0 , int((seg_len + 3) / 2) - 1 : shape(mid_trend)[1]]))

			detrended_data = data - record_y.T


			return detrended_data , record_y

		else :
			xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) + nonoverlap_len , seg_index * (seg_len - 1) + nonoverlap_len + 2)
			xi_mid = mat(xi)
			xi_max = xi.max()
			xi_min = xi.min()
			seg_data = data[xi_min - 1 : xi_max , 0]
			mid_trend = (A_coeff * seg_data).T

		# right trend
		xi = np.arange(seg_index * (seg_len - 1) + 1 , data_len + 1)
		xi_right = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		nrows_seg = shape(seg_data)[0]

		if nrows_seg < seg_len :
			coeff_output1 , A1 = detrending_coeff(nrows_seg , fit_order)
			A_coeff1 = A1 * coeff_output1
			right_trend = (A_coeff1 * seg_data).T
		else:
			right_trend = (A_coeff * seg_data).T

		xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
		xx2  =mid_trend[0 , 0 : int((seg_len + 1) / 2)]
		w = np.arange(0 , nonoverlap_len + 1)/nonoverlap_len
		xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

		xx1 = mid_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
		xx2 = right_trend[0 , 0 : int((seg_len + 1) / 2)]
		w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
		xx_right = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

		record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 3) / 2) - 1 : shape(xi_left)[1]] , xi_mid[0 , int((shape(xi_mid)[1] + 1) / 2) : shape(xi_mid)[1]]))
		record_y = hstack((record_y , xx_left[0 , 1 : shape(xx_left)[1]] , xx_right[0 , 1 : shape(xx_right)[1]]))

		right_start_index = mat([(j) for j in range(shape(xi_right)[1]) if xi_right[0 , j] == xi_mid[0 , shape(xi_mid)[1] - 1] + 1])
		nrows_mid = shape(right_start_index)[1]

		if nrows_mid == 1 :
			right_start_index = right_start_index[0,0]
			record_x = hstack((record_x , xi_right[0 , right_start_index : shape(xi_right)[1]]))
			record_y = hstack((record_y , right_trend[0 , right_start_index : shape(right_trend)[1]]))

		detrended_data = data - record_y.T

		return detrended_data , record_y
