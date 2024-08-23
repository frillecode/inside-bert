import os
import time
from tqdm import tqdm
import ndjson
import numpy as np
from numpy import *
from scipy import stats
from wasabi import msg
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# KL-divergence
def kld(p, q):
    """KL-divergence for two probability distributions.
    Taken from https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/tekisuto/metrics/entropies.py
    commit 1fb16bc91b99716f52b16100cede99177ac75f55"""
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, (p - q) * np.log10(p / q), 0))


# JSD-divergence
def jsd(p, q, base=np.e):
    """Pairwise Jensen-Shannon Divergence for two probability distributions.
    Taken from https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/tekisuto/metrics/entropies.py
    commit 1fb16bc91b99716f52b16100cede99177ac75f55"""
    # convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    # normalize p, q to probabilities
    p, q = p / p.sum(), q / q.sum()
    m = 1.0 / 2 * (p + q)
    return stats.entropy(p, m, base=base) / 2.0 + stats.entropy(q, m, base=base) / 2.0


# InfoDynamics class
class InfoDynamics:
    def __init__(self, data, time, window=3, weight=0, sort=False):
        """
        Class for estimation of information dynamics of time-dependent probabilistic document representations.
        Taken from https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/tekisuto/models/infodynamics.py
        commit 1fb16bc91b99716f52b16100cede99177ac75f55

        - data: list/array (of lists), bow representation of documents
        - time: list/array, time coordinate for each document (identical order as data)
        - window: int, window to compute novelty, transience, and resonance over
        - weight: int, parameter to set initial window for novelty and final window for transience
        - sort: bool, if time should be sorted in ascending order and data accordingly
        """
        self.window = window
        self.weight = weight
        if sort:
            self.data = np.array([text for _, text in sorted(zip(time, data))])
            self.time = sorted(time)
        else:
            self.data = np.array(data)
            self.time = time
        self.m = self.data.shape[0]

    def novelty(self, meas=kld):
        print("[INFO] Calculating novelty")
        start_time = time.time()
        N_hat = np.zeros(self.m)
        N_sd = np.zeros(self.m)
        for i, x in enumerate(tqdm(self.data)):

            submat = self.data[(i - self.window) : i,]
            tmp = np.zeros(submat.shape[0])
            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
            else:
                tmp = np.zeros([self.window]) + self.weight

            N_hat[i] = np.mean(tmp)
            N_sd[i] = np.std(tmp)

        self.nsignal = N_hat
        self.nsigma = N_sd
        print(f"[INFO] Calculating novelty took {time.time() - start_time} seconds")

    def transience(self, meas=kld):
        print("[INFO] Calculating transience")
        start_time = time.time()
        T_hat = np.zeros(self.m)
        T_sd = np.zeros(self.m)
        for i, x in enumerate(tqdm(self.data)):
            submat = self.data[i + 1 : (i + self.window + 1),]
            tmp = np.zeros(submat.shape[0])
            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
            else:
                tmp = np.zeros([self.window])

            T_hat[i] = np.mean(tmp)
            T_hat[-self.window :] = np.zeros([self.window]) + self.weight
            T_sd[i] = np.std(tmp)

        self.tsignal = T_hat
        self.tsigma = T_sd
        print(f"[INFO] Calculating transience took {time.time() - start_time} seconds")

    def resonance(self, meas=kld):
        print("[INFO] Calculating resonance")
        start_time = time.time()
        if not hasattr(self, "nsignal"):
            self.novelty(meas)
        if not hasattr(self, "tsignal"):
            self.transience(meas)
        self.rsignal = self.nsignal - self.tsignal
        self.rsignal[: self.window] = np.zeros([self.window]) + self.weight
        self.rsignal[-self.window :] = np.zeros([self.window]) + self.weight
        self.rsigma = (self.nsigma + self.tsigma) / 2
        self.rsigma[: self.window] = np.zeros([self.window]) + self.weight
        self.rsigma[-self.window :] = np.zeros([self.window]) + self.weight
        print(f"[INFO] Calculating resonance took {time.time() - start_time} seconds")


# Calculate Novelty, Transience & Resonance
def calc_ntr(probability_matrix, window, visualize=False):
    """Calculate Novelty, Transience & Resonance in a given window

    Parameters
    ----------
    probability_matrix : np.array
        array of shape (n_documents, n_labels)
    window : int
        n documents to look before/after document[i]
    visualize : bool, optional
        enable diagnostics plot for novelty? By default False

    Returns
    -------
    entropies.InfoDynamics
        trained instance of infodynamics class
    """
    print(f"[INFO] Calculating NTR with window size {window}")

    start_time = time.time()

    idmdl = InfoDynamics(data=probability_matrix, time=None, window=window, sort=False)

    idmdl.novelty(meas=jsd)
    idmdl.transience(meas=jsd)
    idmdl.resonance(meas=jsd)

    print(f"[INFO] Calculating NTR took {time.time() - start_time} seconds")

    if visualize:
        print("[INFO] Visualizing NTR")
        plt.plot(idmdl.nsignal)

    return idmdl


# Remove first & last {window} documents
def curb_incomplete_signal(timeseries, window):
    """remove first & last {window} documents"""
    print(f"[INFO] Curbing incomplete signal (removing first and last {window} documents)")
    return timeseries[window:-window]


# Calculate slope of resonance ~ novelty linear model
def calculate_resonance_novelty_slope(resonance, novelty):
    """get slope of resonance ~ novelty linear model
    a) standardize
    b) fit a simple linear regression
    c) extract beta coefficient

    Parameters
    ----------
    resonance : np.array-like
    novelty : np.array-like

    Returns
    -------
    float
        slope of lm(resonance ~ novelty)
    """
    print("[INFO] Calculating slope of resonance ~ novelty linear model")

    start_time = time.time()
    # reshape
    novelty = novelty.reshape(-1, 1)
    resonance = resonance.reshape(-1, 1)

    # standardize resonance & novelty
    z_novelty = StandardScaler().fit_transform(novelty)

    z_resonance = StandardScaler().fit_transform(resonance)

    # fit model
    lm = LinearRegression(fit_intercept=False)
    lm.fit(X=z_novelty, y=z_resonance)

    # capture slope
    slope = lm.coef_[0][0]
    # r2
    resonance_pred = lm.predict(z_novelty)
    r2 = r2_score(z_resonance, resonance_pred)
    # p-value

    print(f"[INFO] Calculating slope took {time.time() - start_time} seconds")

    return slope, r2


# # Reshape logits
# def reshape_logits(logits):
#     """Takes a np.array of shape (n_steps, batch_size, n_labels) and reshapes it to (n_documents, n_labels)

#     Parameters:
#         logits (np.array): array of shape (n_steps, batch_size, n_labels)
#     Returns:
#         (np.array): array of shape (n_documents, n_labels)
#     """
#     return logits.reshape(-1, logits.shape[-1])


# load all logits from a directory
def load_and_reshape_logits_from_dir(path, step_cutoff=None):
    """Load all logits from a directory, and reshape them to (n_documents, n_labels)

    Parameters:
        path (str): path to directory containing logits
    Returns:
        (np.array): array of shape (n_documents, n_labels)
    """
    print(f"[INFO] Loading and reshaping logits from {path}")
    start_time = time.time()
    logits = []
    for file in os.listdir(path):
        if file.endswith(".npy"):
            # only add logits from steps below step_cutoff
            if step_cutoff is not None:
                step = int(file.split("_")[2].split('.')[0])
                if step <= step_cutoff:
                    logits.append(np.load(os.path.join(path, file)))
            else:
                logits.append(np.load(os.path.join(path, file)))

    print(f"[INFO] Loaded {len(logits)} files")
    print(f"[INFO] Loading and reshaping took {time.time() - start_time} seconds")
    return np.concatenate(logits)


# def check_shape_logits(logits):
#     for i, inner_list in enumerate(logits):
#         if len(inner_list) != 32:
#             print(f"List at index {i} does not have length 32")
#         for j, element in enumerate(inner_list):
#             if not isinstance(element, np.ndarray):
#                 print(f"Element at index {j} in list {i} is not a numpy array")s


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