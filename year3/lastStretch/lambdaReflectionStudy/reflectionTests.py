import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
from GWXtreme import eos_model_selection as ems
import scipy.stats
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
    print(len(Lambda_1))
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
    print(len(resampledLambda1))

    plt.clf()
    plt.hist(np.log10(Lambda_1),density=True,color='red',bins=80,alpha=0.25,label="lambda_1")
    plt.hist(np.log10(resampledLambda1),density=True,color='blue',bins=80,alpha=0.25,label="resampled")
    plt.legend()
    plt.xlabel("Lambda 1")
    #plt.savefig("low_zero_method12.png")
    plt.savefig("low_neginf_method12.png")

def method3():
    # Second is to compare the hist of lambda_1 to the pdf of the kde of the lambda_1 distribution
    
    bw = 0.75 # is used by documentation
    bw = len(margPostData)**(-1/6) # is used by GWXtreme
    print("bw val: {}".format(bw))
    lambda_1 = data['lambda_1']
    lambda_1_plot = np.linspace(min(lambda_1),max(lambda_1),1000)

    # scipy stats
    kde = scipy.stats.gaussian_kde(lambda_1, bw_method=bw, weights=None)
    pdf = kde(lambda_1_plot)
    plt.clf()
    plt.hist(lambda_1,density=True,color='red',bins=80,alpha=0.25,label="lambda_1")
    plt.plot(lambda_1_plot,pdf,color='blue',alpha=0.50,label="pdf")
    plt.legend()
    plt.xlabel("Lambda 1")
    plt.savefig("scistat_method3.png")
    
    # scilearn
    #lambda_1 = lambda_1[:, np.newaxis]
    #lambda_1_plot = lambda_1_plot[:, np.newaxis]
    #kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(lambda_1)
    #pdf = np.exp(kde.score_samples(lambda_1))
    #plt.clf()
    #plt.hist(lambda_1,density=True,color='red',alpha=0.25,label="lambda_1")
    #plt.plot(lambda_1_plot,pdf,color='blue',alpha=0.50,label="pdf")
    #plt.legend()
    #plt.xlabel("Lambda 1")
    #plt.savefig("scilearn_method3.png")


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
    plt.hist(np.log10(LambdaT),density=True,color='red',alpha=0.25,label="LambdaT")
    plt.hist(np.log10(resampledLambdaT),density=True,color='blue',alpha=0.25,label="resampled")
    plt.legend()
    plt.xlabel("LambdaT")
    plt.savefig("GW170817_LambdaT_test.png")
