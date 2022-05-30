from GWXtreme import eos_model_selection as ems
from matplotlib import pyplot as plt
import numpy as np

def eos_compare(trialn):
    
    # Saves plots of histograms of bayes factors of repeated trials for
    # multiple target eos with SLY and H4 as the reference eos.

    choose_eos = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2","SLY230A",
                  "SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6",
                  "SK272","SKI3","SKI5","MPA1","MS1B_PP","MS1_PP"]

    modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

    for eos in choose_eos:
        comparison1 = modsel.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=trialn)
        comparison2 = modsel.computeEvidenceRatio(EoS1=eos,EoS2="H4",trials=trialn)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)

        ax1.hist(comparison1[1], bins=10, edgecolor="black", label="Trials' Bayes Factors")
        ax1.axvline(comparison1[0], linestyle="--", label="True Bayes Factor")


        ax2.hist(comparison2[1], bins=10, edgecolor="black", label="Trials' Bayes Factors")
        ax2.axvline(comparison2[0], linestyle="--", label="True Bayes Factor")

        ax1.legend(prop={"size":8})
        ax1.set_title("EOS_Target:{}, EOS_Reference:{}".format(eos,"SLY"), fontsize=8)
        ax1.set_xlabel("Bayes Factor")
        ax1.set_ylabel("Frequency")

        ax2.legend(prop={"size":8})
        ax2.set_title("EOS_Target:{}, EOS_Reference:{}".format(eos,"H4"), fontsize=8)
        ax2.set_xlabel("Bayes Factor")

        plt.tight_layout()

        plt.savefig("plots/compare_plot_{}_{}.png".format(eos,choose_eos.index(eos)))
        plt.clf()
