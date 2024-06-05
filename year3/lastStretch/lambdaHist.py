import numpy as np
import lal
import lalsimulation as lalsim
import matplotlib.pyplot as plt

labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]

def calcLambda_parametrized():
    # Recycled code from GWXtreme.

    for ii in range(len(labels)):

        #filename='data/constraints/{}_16simulationsInference1000samp_gammas.txt'.format(labels[ii])
        filename='data/constraints/{}_GW170817inference_gammas.txt'.format(labels[ii])
        samples = np.loadtxt(filename)
        Lambdas = []
        m=1.4
        for sample in samples:

            g0, g1, g2, g3 = sample
            eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g0, g1, g2, g3)
            fam = lalsim.CreateSimNeutronStarFamily(eos)

            rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
            kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
            cc = m*lal.MRSUN_SI/rr
            Lambda = (2/3)*kk/(cc**5)
            Lambdas.append(Lambda)

        #np.savetxt("data/lambdaHists/{}_16simulationsInference1000samp_Lambdas.txt".format(labels[ii]),np.array(Lambdas).T)
        np.savetxt("data/lambdaHists/{}_GW170817inference_Lambdas.txt".format(labels[ii]),np.array(Lambdas).T)


def plotLambda_parametrized(eosname="APR4_EPP"):

    m=1.4
    eos = lalsim.SimNeutronStarEOSByName(eosname)
    fam = lalsim.CreateSimNeutronStarFamily(eos)

    rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
    kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
    cc = m*lal.MRSUN_SI/rr
    eosLambda = (2/3)*kk/(cc**5)

    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)
    Labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    for ii in range(len(Labels)):

        #filename='data/lambdaHists/{}_16simulationsInference1000samp_Lambdas.txt'.format(labels[ii])
        filename='data/lambdaHists/{}_GW170817inference_Lambdas.txt'.format(labels[ii])
        Lambdas = np.loadtxt(filename).T

        plt.hist(Lambdas, label=Labels[ii], alpha=0.45, fill=True)

    plt.axvline(x=eosLambda, label=eosname)

    plt.xlabel("$\Lambda$(1.4)",fontsize=20)
    plt.yticks([])
    plt.legend()
    #plt.savefig("plots/lambdaHists/16simulations1000samp_Lambdas.png", bbox_inches='tight')
    plt.savefig("plots/lambdaHists/GW170817_Lambdas.png", bbox_inches='tight')
