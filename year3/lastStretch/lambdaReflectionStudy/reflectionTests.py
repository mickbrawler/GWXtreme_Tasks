import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
from GWXtreme import eos_model_selection as ems
import scipy.stats
from sklearn.neighbors import KernelDensity

# There's three ways to test if lambda_1's (really any lambda) kde can and should be reflected

# LOAD Data!!!
filename = "../files/NSBH/gw230529_phenom_lowSpin.json"
with open(filename,"r") as f: data = json.load(f)['posterior']['content']

modsel = ems.Model_selection(filename,kdedim=3,logq=True)
margPostData = modsel.margPostData

def method12():
    # One is simply using the resample logic on GWXtreme
    # Second is doing similar but including the logic in the bounded_3d_kde code that accounts for boundaries

    Lambda_1 = margPostData[:,0]
    kde = modsel.kde
    yhigh = modsel.yhigh
    logq = modsel.logq
    logyhigh = modsel.logyhigh
    #new_margPostData = modsel.kde.resample(size=len(margPostData)).T


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


    resampledLambda1 = new_margPostData[:,0]

    plt.clf()
    plt.hist(Lambda_1,density=True,color='red',alpha=0.25,label="lambda_1")
    plt.hist(resampledLambda1,density=True,color='blue',alpha=0.25,label="resampled")
    plt.legend()
    plt.xlabel("Lambda 1")
    plt.savefig("low_zero_method12.png")
    # I think method 1 is already method 2 if kdedim=3. The only way to do method 1
    # as I thought would be to remove the boundary logic... I see why Anarya was surprised.
    # I had misunderstood that resample was part of an external functionality.

def method3():
    # Third is to compare the hist of lambda_1 to the pdf of the kde of the lambda_1 distribution
    
    bw = 0.75 # is used by documentation
    bw = len(margPostData)**(-1/6) # is used by GWXtreme
    lambda_1 = data['lambda_1']
    lambda_1_plot = np.linspace(min(lambda_1),max(lambda_1),1000)

    # scipy stats
    kde = scipy.stats.gaussian_kde(lambda_1, bw_method=bw, weights=None)
    pdf = kde(lambda_1_plot)
    plt.clf()
    plt.hist(lambda_1,density=True,color='red',alpha=0.25,label="lambda_1")
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


