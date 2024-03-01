from GWXtreme import eos_model_selection as ems
import numpy as np
import matplotlib.pyplot as plt
import json

# Colors Ghosh used for BF Bar plots.
# #8fbbd9 #ff7f0f
# #1f77b4 #ffd7b6 #bfe2bf

def singleEventBFs():
    # Plots barplot of BFs using GW170817's uP(LTs) posterior and
    #                                       uP(Ls) posterior (with errorbars)!

    uLTs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/outdir/real/uniformP_LTs/GW170817/data_labelsAdjusted.json" 
    uLs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year3/GW170817_prior_L1L2/CIT_attempt_successful/outdir/GW170817_result_simplified.json"

    modsel_uLTs = ems.Model_selection(uLTs_File,Ns=4000,kdedim=2)
    modsel_uLs = ems.Model_selection(uLs_File,Ns=4000,kdedim=3)

    labels = ["2D KDE", "3D KDE","Actual"]
    #colors = ["#1b9e77","#d95f02","#7570b3"] # Colors we initially used
    colors = ["#1f77b4", "#ffd7b6", "#bfe2bf"]
    methods = [modsel_uLTs, modsel_uLs]
    eosList = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
    
    with open("/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/outdir/nested_sampling_results.json","r") as f:
        nestSamp = json.load(f)
    nest_BFs = []
    nest_stds = []
    for eos in eosList:
        nest_BFs.append(nestSamp[eos][0])
        nest_stds.append(nestSamp[eos][1])

    methods_BFs = []
    methods_uncerts = []
    for method in methods:
        print(method)
        BFs = []
        uncerts = []
        for eos in eosList:
            print(eos)
            bf, bf_trials = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=100)
            #bf = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=0)
            uncert = np.std(bf_trials) * 2
            BFs.append(bf)
            uncerts.append(uncert)
        methods_BFs.append(BFs)
        methods_uncerts.append(uncerts)

    x_axis = np.arange(len(eosList))
    plt.clf()
    spacing = [-.15,0,.15]
    plt.rcParams.update({"font.size":18})
    plt.figure(figsize=(15, 10))
    for index in range(len(methods)):
        #plt.bar(x_axis+spacing[index],methods_BFs[index],.15,yerr=methods_uncerts[index],label=labels[index],color=colors[index])
        plt.bar(x_axis+spacing[index],methods_BFs[index],.15,label=labels[index],color=colors[index])

        plt.errorbar(x_axis+spacing[index],methods_BFs[index],yerr=methods_uncerts[index],ls="none",ecolor="black",capsize=3.0)
    
    plt.bar(x_axis+spacing[2],nest_BFs,.15,yerr=nest_stds,label=labels[2],color=colors[2])

    plt.yscale("log")
    plt.xticks(x_axis,eosList,rotation=45,ha="right")
    plt.ylim(1.0e-4,(max(BFs)+max(uncerts))*10.)
    #plt.axhline(1.0,color="k",linestyle="--")
    #plt.title("EoS Bayes Factors w.r.t. SLY")
    plt.ylabel("Bayes-factor w.r.t SLY")
    plt.legend()
    label = "GW170817_CITrun"
    plt.savefig("plots/2D_3D/{}_barplot_2D_3D_BFs.png".format(label),bbox_inches="tight")

    Dictionary = {labels[Index]:{eosList[eIndex]:[methods_BFs[Index][eIndex],methods_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(methods))}
    with open("plots/2D_3D/data/{}_2D_3D_BFs.json".format(label),"w") as f:
        json.dump(Dictionary, f, indent=2, sort_keys=True)

