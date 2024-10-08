import os
import time
from tqdm import tqdm
import numpy as np
from numpy import *
from scipy import stats
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


def check_shape_logits(logits):
    for i, inner_list in enumerate(logits):
        if len(inner_list) != 64:
            print(f"List at index {i} does not have length 32")
        for j, element in enumerate(inner_list):
            if not isinstance(element, np.ndarray):
                print(f"Element at index {j} in list {i} is not a numpy array")

