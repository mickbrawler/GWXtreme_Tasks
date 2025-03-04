import numpy as np
import lal
import os.path
import lalsimulation as lalsim
import matplotlib.pyplot as plt

#labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]

labels = ["3D-KDE-PhenomNRT"]

def calcLambda_parametrized():
    # Recycled code from GWXtreme.

    for ii in range(len(labels)):

        #filename='data/BNS/constraints/{}_GW170817inference_gammas_10000samp.txt'.format(labels[ii])
        #if os.path.isfile(filename) != True: filename='data/BNS/constraints/{}_GW170817inference_gammas.txt'.format(labels[ii])
        #filename='data/BNS/constraints/{}_16simulationsInference_10000samp_gammas.txt'.format(labels[ii])
        filename='./data/NSBH/constraints/{}_18simulationsInference_10000samp_gammas.txt'.format(labels[ii])
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

        #np.savetxt("data/BNS/lambdaHists/{}_GW170817inference_Lambdas_10000samp.txt".format(labels[ii]),np.array(Lambdas).T)
        #np.savetxt("data/BNS/lambdaHists/{}_16simulationsInference_Lambdas_10000samp.txt".format(labels[ii]),np.array(Lambdas).T)
        np.savetxt("data/NSBH/lambdaHists/{}_18simulationsInference_Lambdas_10000samp.txt".format(labels[ii]),np.array(Lambdas).T)


#labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
#labels = ["3D-KDE-PhenomNRT", "lalsim_nest-PhenomNRT"]
#labels = ["2D-KDE-TaylorF2", "3D-KDE-PhenomNRT", "lalsim_nest-PhenomNRT"]
#labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT", "lalsim_nest-PhenomNRT"]

labels = ["3D-KDE-PhenomNRT"]

def plotLambda_parametrized(eosname="APR4_EPP"):

    m=1.4
    eos = lalsim.SimNeutronStarEOSByName(eosname)
    fam = lalsim.CreateSimNeutronStarFamily(eos)

    rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
    kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
    cc = m*lal.MRSUN_SI/rr
    eosLambda = (2/3)*kk/(cc**5)

    #Colors = ['#ffffb3','#bebada','#fb8072']
    #Colors = ['#beaed4','#fdc086']
    #Colors = ['#ffffb3','#beaed4','#fdc086']
    #Colors = ['#ffffb3','#bebada','#beaed4','#fdc086']

    Colors = ['#fb8072']

    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    #plt.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    #Labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE IMRPhenomPv2_NRTidal"]
    #Labels = ["3D KDE IMRPhenomPv2_NRTidal", "full-parameter-space Nest runs"]
    #Labels = ["2D KDE TaylorF2", "3D KDE IMRPhenomPv2_NRTidal", "full-parameter-space Nest runs"]
    #Labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE IMRPhenomPv2_NRTidal", "full-parameter-space Nest runs"]

    Labels = ["3D KDE IMRPhenomPv2_NRTidal"]
    for ii in range(len(Labels)):

        #filename='data/BNS/lambdaHists/{}_GW170817inference_Lambdas_10000samp.txt'.format(labels[ii])
        #if os.path.isfile(filename) != True: filename='data/BNS/lambdaHists/{}_GW170817inference_Lambdas.txt'.format(labels[ii])
        #filename='data/BNS/lambdaHists/{}_16simulationsInference_Lambdas_10000samp.txt'.format(labels[ii])
        filename='data/NSBH/lambdaHists/{}_18simulationsInference_Lambdas_10000samp.txt'.format(labels[ii])
        Lambdas = np.loadtxt(filename).T
        plt.hist(Lambdas, label=Labels[ii], alpha=0.45, fill=True, density=True, color=Colors[ii], histtype='step')

    plt.axvline(x=eosLambda, label=eosname, color="black")

    plt.xlabel("$\Lambda$(1.4)",fontsize=20)
    plt.yticks([])
    plt.legend()
    #plt.savefig("plots/BNS/lambdaHists/GW170817_Lambdas1_10000samp.png", bbox_inches='tight')
    #plt.savefig("plots/BNS/lambdaHists/GW170817_Lambdas2_10000samp.png", bbox_inches='tight')
    #plt.savefig("plots/BNS/lambdaHists/GW170817_Lambdas3_10000samp.png", bbox_inches='tight')
    #plt.savefig("plots/BNS/lambdaHists/GW170817_Lambdas4_10000samp.png", bbox_inches='tight')

    #plt.savefig("plots/BNS/lambdaHists/16simulations_Lambdas_10000samp.png", bbox_inches='tight')
    plt.savefig("plots/NSBH/lambdaHists/18simulations_Lambdas_10000samp.png", bbox_inches='tight')

