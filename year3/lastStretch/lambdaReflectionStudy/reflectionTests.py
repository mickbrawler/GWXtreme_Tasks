import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
from GWXtreme import eos_model_selection as ems
import scipy.stats
from scipy.interpolation import make_interp_spline
from sklearn.neighbors import KernelDensity

# There's two ways to test if lambda_1's (really any lambda) kde can and should be reflected

# LOAD Data!!!
filename = "../files/NSBH/gw230529_phenom_lowSpin.json"
with open(filename,"r") as f: data = json.load(f)['posterior']['content']

modsel = ems.Model_selection(filename,kdedim=3,logq=True) #uncomment later
margPostData = modsel.margPostData #uncomment later

def method12():
    # One is simply using the resample logic on GWXtreme

    Lambda_1 = margPostData[:,0]
    kde = modsel.kde
    yhigh = modsel.yhigh
    logq = modsel.logq
    logyhigh = modsel.logyhigh

    # logic from get_trials() that gets you "synthetic" data
    new_margPostData = np.array([])
    counter = 0
    while len(new_margPostData) < len(margPostData):
        prune_adjust_factor = 1.1 + counter/10.
        N_resample = int(len(margPostData)*prune_adjust_factor)
        new_margPostData = kde.resample(size=N_resample).T
        unphysical = (new_margPostData[:, 0] < 0) +\
                     (new_margPostData[:, 1] > (yhigh if not logq else logyhigh) )
        if not logq: unphysical + (new_margPostData[:, 1] < 0)
        else: pass
        new_margPostData = new_margPostData[~unphysical]
        counter += 1
    indices = np.arange(len(new_margPostData))
    chosen = np.random.choice(indices, len(margPostData))
    new_margPostData = new_margPostData[chosen]

    # parameter we care about for this study
    resampledLambda1 = new_margPostData[:,0]

    plt.clf()
    plt.hist(np.log10(Lambda_1),density=True,color='red',bins=80,alpha=0.25,label="lambda_1")
    plt.hist(np.log10(resampledLambda1),density=True,color='blue',bins=80,alpha=0.25,label="resampled")
    plt.legend()
    plt.xlabel("Lambda 1")
    #plt.savefig("low_zero_method12.png")
    plt.savefig("low_neginf_method12.png")


def method3(scistat=True):
    # Second is to compare the hist of lambda_1 to the pdf of the kde of the lambda_1 distribution
    
    bw = 0.75 # is used by documentation
    bw = len(margPostData)**(-1/6) # is used by GWXtreme
    lambda_1 = data['lambda_1']
    lambda_1_plot = np.linspace(min(lambda_1),max(lambda_1),1000)

    # Use scipy.stats logic
    if scistat:
        kde = scipy.stats.gaussian_kde(lambda_1, bw_method=bw, weights=None)
        pdf = kde(lambda_1_plot)
        plt.clf()
        plt.hist(lambda_1,density=True,color='red',bins=80,alpha=0.25,label="lambda_1")
        plt.plot(lambda_1_plot,pdf,color='blue',alpha=0.50,label="pdf")
        plt.legend()
        plt.xlabel("Lambda 1")
        plt.savefig("scistat_method3.png")
    
    # Use scilearn.neighbors logic
    if scistat != True:
        Lambda_1 = np.array(lambda_1).reshape(-1,1)
        Lambda_1_plot = lambda_1_plot.reshape(-1,1)
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(Lambda_1)
        pdf = np.exp(kde.score_samples(Lambda_1_plot))
        plt.clf()
        plt.hist(lambda_1,density=True,color='red',bins=80,alpha=0.25,label="lambda_1")
        plt.plot(lambda_1_plot,pdf,color='blue',alpha=0.50,label="pdf")
        plt.legend()
        plt.xlabel("Lambda 1")
        plt.savefig("scilearn_method3.png")

    #note: Both these pacakges have an RBF function that'll fit gaussians too (spline)

def twoD_kdeTest():
    # Trying to do the same lambdat studying Ghosh did for GWXtreme's 2D kde development
    # A sanity check of sorts.

    # LOAD Data!!!
    filename = "/home/michael/projects/eos/GWXtreme_Tasks/year3/LSAMP_Poster/anaryaShare/Files/posterior_samples_narrow_spin_prior_170817.dat"

    modsel = ems.Model_selection(filename) 
    margPostData = modsel.margPostData

    LambdaT = margPostData[:,0]
    kde = modsel.kde
    yhigh = modsel.yhigh

    # logic from get_trials() that gets you "synthetic" data
    new_margPostData = np.array([])
    counter = 0
    while len(new_margPostData) < len(margPostData):
        prune_adjust_factor = 1.1 + counter/10.
        N_resample = int(len(margPostData)*prune_adjust_factor)
        new_margPostData = kde.resample(size=N_resample).T
        unphysical = (new_margPostData[:, 0] < 0) +\
                     (new_margPostData[:, 1] > yhigh) +\
                     (new_margPostData[:, 1] < 0)
        new_margPostData = new_margPostData[~unphysical]
        counter += 1
    indices = np.arange(len(new_margPostData))
    chosen = np.random.choice(indices, len(margPostData))
    new_margPostData = new_margPostData[chosen]

    # parameter we care about for this study
    resampledLambdaT = new_margPostData[:,0]

    plt.clf()
    plt.hist(np.log10(LambdaT),density=True,color='red',bins=80,alpha=0.25,label="LambdaT")
    plt.hist(np.log10(resampledLambdaT),density=True,color='blue',bins=80,alpha=0.25,label="resampled")
    plt.legend()
    plt.xlabel("LambdaT")
    plt.savefig("GW170817_LambdaT_test.png")


def ksTest():
    # Use Kolmogorov-Smirnov test to get a quantitative value for the discrepancy
    # between the two histograms for method12.

    Lambda_1 = margPostData[:,0]
    kde = modsel.kde
    yhigh = modsel.yhigh
    logq = modsel.logq
    logyhigh = modsel.logyhigh

    # logic from get_trials() that gets you "synthetic" data
    new_margPostData = np.array([])
    counter = 0
    while len(new_margPostData) < len(margPostData):
        prune_adjust_factor = 1.1 + counter/10.
        N_resample = int(len(margPostData)*prune_adjust_factor)
        new_margPostData = kde.resample(size=N_resample).T
        unphysical = (new_margPostData[:, 0] < 0) +\
                     (new_margPostData[:, 1] > (yhigh if not logq else logyhigh) )
        if not logq: unphysical + (new_margPostData[:, 1] < 0)
        else: pass
        new_margPostData = new_margPostData[~unphysical]
        counter += 1
    indices = np.arange(len(new_margPostData))
    chosen = np.random.choice(indices, len(margPostData))
    new_margPostData = new_margPostData[chosen]

    # parameter we care about for this study
    resampledLambda1 = new_margPostData[:,0]

    # Perform ks-test between the distributions' values
    rawKS = scipy.stat.ks_2samp(resampledLambda1,Lambda_1)
    
    # Perform ks-test between the distributions' hist heights
    histDefault = np.histogram(Lambda_1,bins=80,density=True)
    histResample = np.histogram(resampledLambda1,bins=80,density=True)
    histKS = scipy.stat.ks_2samp(resampledLambda1,Lambda_1)

    # Perform ks-test between the distributions' hists' smoothed out (spline)
hist, bin_edges = np.histogram(data, bins=20)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
spl = make_interp_spline(bin_centers, hist, k=3)
y_smooth = spl(x_smooth)
plt.plot(x_smooth, y_smooth)

    # We can histogram the 
    plt.hist(np.log10(Lambda_1),density=True,color='red',bins=80,alpha=0.25,label="lambda_1")
    plt.hist(np.log10(resampledLambda1),density=True,color='blue',bins=80,alpha=0.25,label="resampled")
    plt.legend()
    plt.xlabel("Lambda 1")
    #plt.savefig("low_zero_method12.png")
    plt.savefig("low_neginf_method12.png")
