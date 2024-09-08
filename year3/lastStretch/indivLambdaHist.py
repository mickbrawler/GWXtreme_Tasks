import numpy as np
import lal
import os.path
import lalsimulation as lalsim
import matplotlib.pyplot as plt

# Currently just equipped to produce the individual constraints of NSBH sims

label = "3D-KDE-PhenomNRT"
# specific injection (NSBH) name list for individual constraint computations
injections = ['103_8.82_1.23','116_9.83_1.15','131_2.2_1.53','177_9.59_1.97','196_3.34_2.13',
             '227_4.19_2.05','236_7.03_1.96','261_4.16_2.08','267_4.47_1.66','321_3.11_2.08',
             '327_3.11_1.22','380_7.95_2.1','386_2.64_1.81','432_3.51_1.94','452_3.55_1.47',
             '455_2.33_1.97','467_5.58_2.06','756_7.0_1.58']

def calcLambda_parametrized():
    # Recycled code from GWXtreme.

    for injection in injections:
        print(injection)
        filename='./data/NSBH/constraints/{}_{}simulationsInference_10000samp_gammas.txt'.format(label,injection)
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

        np.savetxt("data/NSBH/lambdaHists/{}_{}simulationsInference_Lambdas_10000samp.txt".format(label,injection),np.array(Lambdas).T)



def plotLambda_parametrized(eosname="APR4_EPP"):

    m=1.4
    eos = lalsim.SimNeutronStarEOSByName(eosname)
    fam = lalsim.CreateSimNeutronStarFamily(eos)

    rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
    kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
    cc = m*lal.MRSUN_SI/rr
    eosLambda = (2/3)*kk/(cc**5)

    Color = '#fb8072'

    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    #plt.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    Label = "3D KDE IMRPhenomPv2_NRTidal"

    for injection in injections:
        print(injection)
        filename='data/NSBH/lambdaHists/{}_{}simulationsInference_Lambdas_10000samp.txt'.format(label,injection)
        Lambdas = np.loadtxt(filename).T
        plt.clf()
        plt.hist(Lambdas, label=Label, alpha=0.45, fill=True, density=True, color=Color, histtype='step')

        plt.axvline(x=eosLambda, label=eosname, color="black")

        plt.xlabel("$\Lambda$(1.4)",fontsize=20)
        plt.yticks([])
        plt.legend()
        plt.savefig("plots/NSBH/lambdaHists/{}simulations_Lambdas_10000samp.png".format(injection), bbox_inches='tight')

