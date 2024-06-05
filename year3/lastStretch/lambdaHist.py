import numpy as np
import lal
import lalsimulation as lalsim
import matplotlib.pyplot as plt

def getLambda_parametrized():
    # Recycled code from GWXtreme.
    
    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
    Labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    for ii in range(len(labels)):

        filename='data/constraints/{}_16simulationsInference1000samp_gammas.txt'.format(labels[ii])
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

        plt.hist(Lambdas, label=Labels[ii])

    plt.xlabel("r'$\Lambda$(1.4)",fontsize=20)
    plt.legend()

    plt.savefig("plots/lambdaHists/16simulations1000samp_Lambdas.png", bbox_inches='tight')

#np.savetxt("data/constraints/{}_GW170817inference_gammas.txt".format(label),samples)
