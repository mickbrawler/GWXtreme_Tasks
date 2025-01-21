import numpy as np
import lal
import os.path
import lalsimulation as lalsim
import matplotlib.pyplot as plt


#labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
labels = ["3D-KDE-PhenomNRT"]

def calcMaxMass_parametrized():
    # Recycled code from GWXtreme.

    for ii in range(len(labels)):

        #filename='data/BNS/constraints/{}_GW170817inference_gammas_10000samp.txt'.format(labels[ii])
        #if os.path.isfile(filename) != True: filename='data/BNS/constraints/{}_GW170817inference_gammas.txt'.format(labels[ii])
        #filename='data/BNS/constraints/{}_16simulationsInference_10000samp_gammas.txt'.format(labels[ii])
        filename='./data/NSBH/constraints/{}_18simulationsInference_10000samp_gammas.txt'.format(labels[ii])
        samples = np.loadtxt(filename)
        maxMasses = []
        m=1.4
        for sample in samples:

            g0, g1, g2, g3 = sample
            eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g0, g1, g2, g3)
            fam = lalsim.CreateSimNeutronStarFamily(eos)
            maxMass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI

        #np.savetxt("data/BNS/massHists/{}_GW170817inference_MaxMasses_10000samp.txt".format(labels[ii]),np.array(maxMasses).T)
        #np.savetxt("data/BNS/massHists/{}_16simulationsInference_MaxMasses_10000samp.txt".format(labels[ii]),np.array(maxMasses).T)
        #np.savetxt("data/BNS/massHists/{}_GW230529inference_MaxMasses_10000samp.txt".format(labels[ii]),np.array(maxMasses).T)
        np.savetxt("data/NSBH/massHists/{}_18simulationsInference_MaxMasses_10000samp.txt".format(labels[ii]),np.array(maxMasses).T)


#labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
#labels = ["3D-KDE-PhenomNRT", "lalsim_nest-PhenomNRT"]
#labels = ["2D-KDE-TaylorF2", "3D-KDE-PhenomNRT", "lalsim_nest-PhenomNRT"]
#labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT", "lalsim_nest-PhenomNRT"]

labels = ["3D-KDE-PhenomNRT"]

def plotMaxMass_parametrized(eosname="APR4_EPP"):

    m=1.4
    eos = lalsim.SimNeutronStarEOSByName(eosname)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    eosMaxMass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI

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

        #filename='data/BNS/massHists/{}_GW170817inference_MaxMasses_10000samp.txt'.format(labels[ii])
        #if os.path.isfile(filename) != True: filename='data/BNS/massHists/{}_GW170817inference_MaxMasses.txt'.format(labels[ii])
        #filename='data/BNS/massHists/{}_16simulationsInference_MaxMasses_10000samp.txt'.format(labels[ii])
        filename='data/NSBH/massHists/{}_18simulationsInference_MaxMasses_10000samp.txt'.format(labels[ii])
        MaxMasses = np.loadtxt(filename).T
        plt.hist(MaxMasses, label=Labels[ii], alpha=0.45, fill=True, density=True, color=Colors[ii], histtype='step')

    plt.axvline(x=eosMaxMass, label=eosname, color="black")

    plt.xlabel("Max NS Masses",fontsize=20)
    plt.yticks([])
    plt.legend()
    #plt.savefig("plots/BNS/massHists/GW170817_MaxMasses1_10000samp.png", bbox_inches='tight')
    #plt.savefig("plots/BNS/massHists/GW170817_MaxMasses2_10000samp.png", bbox_inches='tight')
    #plt.savefig("plots/BNS/massHists/GW170817_MaxMasses3_10000samp.png", bbox_inches='tight')
    #plt.savefig("plots/BNS/massHists/GW170817_MaxMasses4_10000samp.png", bbox_inches='tight')

    #plt.savefig("plots/BNS/massHists/16simulations_MaxMasses_10000samp.png", bbox_inches='tight')
    plt.savefig("plots/NSBH/massHists/18simulations_MaxMasses_10000samp.png", bbox_inches='tight')

