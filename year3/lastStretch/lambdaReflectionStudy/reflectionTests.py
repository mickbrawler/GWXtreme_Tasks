import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
from scipy.interpolate import make_interp_spline
from sklearn.neighbors import KernelDensity

# There's two ways to test if lambda_1's (really any lambda) kde can and should be reflected

# LOAD Data!!!
filename = "../files/NSBH/gw230529_phenom_lowSpin.json"
with open(filename,"r") as f: data = json.load(f)['posterior']['content']

logq=True
load=True
if load!=True: 
    from GWXtreme import eos_model_selection as ems
    modsel = ems.Model_selection(filename,kdedim=3,logq=logq) #uncomment later
    margPostData = modsel.margPostData #uncomment later

Tag = "q"
if logq: Tag = "logq"

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
    plt.xlabel("log10(Lambda 1)")
    plt.savefig("{}_method12.png".format(Tag))


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
        plt.xlabel("log10(Lambda 1)")
        plt.savefig("{}_scistat_method3.png".format(Tag))
    
    # Use scilearn.neighbors logic
    if scistat != True:
        Lambda_1 = np.array(lambda_1).reshape(-1,1)
        Lambda_1_plot = lambda_1_plot.reshape(-1,1)
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(Lambda_1)
        pdf = np.exp(kde.score_samples(Lambda_1_plot))
        plt.clf()
        plt.hist(np.log10(lambda_1),density=True,color='red',bins=80,alpha=0.25,label="lambda_1")
        plt.plot(np.log10(lambda_1_plot),pdf,color='blue',alpha=0.50,label="pdf")
        plt.legend()
        plt.xlabel("log10(Lambda 1)")
        plt.savefig("{}_scilearn_method3.png".format(Tag))

    #note: Both these packages have an RBF function that'll fit gaussians too (spline)

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
    plt.xlabel("log10(LambdaT)")
    plt.savefig("GW170817_LambdaT_test.png")


def ksTest():
    # Use Kolmogorov-Smirnov test to get a quantitative value for the discrepancy
    # between the two histograms for method12.

    if load != True: # Requires older python/scipy version :)
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
        np.savetxt("resampleL_L_samples.txt",np.array([resampledLambda1,Lambda_1]).T)

    else: # Requires recent python/scipy version :)
        resampledLambda1, Lambda_1 = np.loadtxt("./resampleL_L_samples.txt").T
        # Perform ks-test between the distributions' values
        rawKS = scipy.stats.ks_2samp(resampledLambda1,Lambda_1)

        # The below would only be necessary for a "homemade" ks-test script
        # Perform ks-test between the distributions' hist heights
        histDef, bin_edgesDef = np.histogram(Lambda_1,bins=80,density=True)
        bin_centDef = (bin_edgesDef[:-1] + bin_edgesDef[1:]) / 2

        histResamp, bin_edgesResamp = np.histogram(resampledLambda1,bins=80,density=True)
        bin_centResamp = (bin_edgesResamp[:-1] + bin_edgesResamp[1:]) / 2

        #histKS = scipy.stats.kstest(histResamp,histDef) # CHECK if ks-test can do hist heights?
        # PROB NOT

        # Perform ks-test between the distributions' hists' smoothed out (spline)
        bin_centDef_smooth = np.linspace(bin_centDef.min(), bin_centDef.max(), 1000)
        splDef = make_interp_spline(bin_centDef, histDef, k=3) # CHECK what is k?
        histDef_smooth = splDef(bin_centDef_smooth)

        bin_centResamp_smooth = np.linspace(bin_centResamp.min(), bin_centResamp.max(), 1000)
        splResamp = make_interp_spline(bin_centResamp, histResamp, k=3)
        histResamp_smooth = splResamp(bin_centResamp_smooth)

        #splineKS = scipy.stats.kstest(histResamp_smooth,histDef_smooth) # CHECK if ks-test can do hist splines??
        # PROB NOT

        plt.clf()
        plt.bar(bin_edgesDef[:-1],histDef,width=np.diff(bin_edgesDef),align='edge',color='red',alpha=0.25)
        plt.bar(bin_edgesResamp[:-1],histResamp,width=np.diff(bin_edgesResamp),align='edge',color='blue',alpha=0.25)
        plt.plot(bin_centDef_smooth,histDef_smooth,color='red',label="lambda_1")
        plt.plot(bin_centResamp_smooth,histResamp_smooth,color='blue',label="resampled")
        plt.vlines(rawKS.statistic_location,ymin=0,ymax=max([max(histDef),max(histResamp),max(histDef_smooth),max(histResamp_smooth)]),label="sign:{}".format(rawKS.statistic_sign),color="black")
        plt.xlim(0,2)
        plt.legend()
        plt.xlabel("Lambda 1")
        plt.title("ks:{:.2f}, p:{}".format(rawKS[0],rawKS[1]))
        plt.savefig("{}_spline_method12.png".format(Tag))

