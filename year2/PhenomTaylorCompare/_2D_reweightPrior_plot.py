# Reweighting prior task (using a uP(Ls) sourced (L~,q) posterior, can I get the
# same BF as a uP(LTs) sourced (L~,q) posterior if I "account" for it in the 
# computation of the Bayes Factor). This requires use of the my PriorInverseWeight
# branch of my fork of GWXtreme.

import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import fsolve
from GWXtreme import eos_model_selection as ems
import glob

def singleEventBFs(priorFile,log=False):
    # Plot bars of BFS of same injected event, differing in their prior.
    # UniformP (dL~, L~), UniformP (L1, L2) used as priors for the runs.
    # For the UniformP (L1, L2) one, we'll produce BFs with and without 
    # inverse weighting.

    uLTs_Dir = "../bilby_runs/3dkde_studies/Anarya_uniformLTs/phenom-injections/TaylorF2"
    uLs_Dir = "../bilby_runs/3dkde_studies/outdir/Phenom_Taylor/IMRPhenomPv2_NRTidal/APR4_EPP"
    injections = ["282_1.58_1.37", "202_1.35_1.14", "179_1.35_1.23", "71_1.37_1.33", "122_1.77_1.19",
                  "241_1.31_1.28", "220_1.36_1.24", "282_1.35_1.32", "149_1.35_1.23", "237_1.36_1.26",
                  "138_1.5_1.21", "235_1.4_1.3", "219_1.3_1.28", "260_1.48_1.33", "164_1.34_1.19",
                  "55_1.38_1.33", "78_1.35_1.32"]
    filenameEnd = "bns_example_result.json"
    for injection in injections:
        print(injection)
        uLTs_File = "{}/{}/{}".format(uLTs_Dir,injection,filenameEnd)
        uLs_File = "{}/{}/{}".format(uLs_Dir,injection,filenameEnd)

        uLTs_modsel = ems.Model_selection(uLTs_File,UpriorLTs=True,Ns=4000)
        uLs_modsel = ems.Model_selection(uLs_File,UpriorLTs=True,Ns=4000)
        uLs_reweightM_modsel = ems.Model_selection(uLs_File,UpriorLTs=False,Ns=4000)
        uLs_reweightB_modsel = ems.Model_selection(uLs_File,UpriorLTs=priorFile,Ns=4000)

        labels = ["UniformP (dL~,L~)", "UniformP (L1,L2)", "UniformP (L1,L2), MOCK reweight", "UniformP (L1,L2), BILBY reweight"]
        colors = ["#d7191c", "#fdae61", "#abdda4", "#2b83ba"]

        methods = [uLTs_modsel, uLs_modsel, uLs_reweightM_modsel, uLs_reweightB_modsel]
        eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP","MPA1","MS1_PP","MS1B_PP"]
        methods_BFs = []
        methods_gs = []
        methods_uncerts = []
        for method in methods:
            print(method)
            BFs = []
            gs = []
            uncerts = []
            for eos in eosList:
                print(eos)
                bf, g, bf_trials = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=1000)
                #bf, g = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=0)
                uncert = np.std(bf_trials) * 2
                BFs.append(bf)
                gs.append(g)
                uncerts.append(uncert)
            methods_BFs.append(BFs)
            methods_gs.append(gs)
            methods_uncerts.append(uncerts)

        x_axis = np.arange(len(eosList))
        spacing = [-.3,-.1,.1,.3]
        plt.clf()
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(15, 10))
        for index in range(len(methods)):
            plt.bar(x_axis+spacing[index],methods_BFs[index],.15,yerr=methods_uncerts[index],label=labels[index],color=colors[index])
            #plt.bar(x_axis+spacing[index],methods_BFs[index],.15,label=labels[index],color=colors[index])

        if log == False: plt.ylim(top=1.2)
        plt.xticks(x_axis,eosList,rotation=45,ha="right")
        ax = plt.gca()
        if log == True: ax.set_yscale("log")
        plt.legend()
        plt.xlabel("EoSs")
        plt.ylabel("Bayes Factor")
        plt.title("EoS Bayes Factors w.r.t. SLY")
        label = uLTs_File.split('/')[-2]
        plt.savefig("plots/difPriors/{}_barplot_difPriors_BFs.png".format(label))

        Dictionary = {labels[Index]:{eosList[eIndex]:[methods_BFs[Index][eIndex],methods_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(labels))}
        with open("plots/difPriors/data/{}_difPriors_BFs.json".format(label),"w") as f:
            json.dump(Dictionary, f, indent=2, sort_keys=True)


def multipleEventBFs(priorFile, log=False):
    uLTs_Dir = "../bilby_runs/3dkde_studies/Anarya_uniformLTs/phenom-injections/TaylorF2/"
    uLs_Dir = "../bilby_runs/3dkde_studies/outdir/Phenom_Taylor/IMRPhenomPv2_NRTidal/APR4_EPP/"
    uLTs_Files = glob.glob("{}/*/*.json".format(uLTs_Dir))
    uLs_Files = glob.glob("{}/*/*.json".format(uLs_Dir))

    uLTs_stack = ems.Stacking(uLTs_Files,UpriorLTs=True,Ns=4000)
    uLs_stack = ems.Stacking(uLs_Files,UpriorLTs=True,Ns=4000)
    uLs_reweightM_stack = ems.Stacking(uLs_Files,UpriorLTs=False,Ns=4000)
    uLs_reweightB_stack = ems.Stacking(uLs_Files,UpriorLTs=priorFile,Ns=4000)

    labels = ["UniformP (dL~,L~)", "UniformP (L1,L2)", "UniformP (L1,L2), MOCK reweight", "UniformP (L1,L2), BILBY reweight"]
    colors = ["#d7191c", "#fdae61", "#abdda4", "#2b83ba"]
    stacks = [uLTs_stack, uLs_stack, uLs_reweightM_stack, uLs_reweightB_stack]
    eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP","MPA1","MS1_PP","MS1B_PP"]
    stacks_BFs = []
    stacks_uncerts = []
    for stack in stacks:
        print(stack)
        BFs = []
        uncerts = []
        for eos in eosList:
            print(eos)
            bf, bf_trials = stack.stack_events(EoS1=eos,EoS2="SLY",trials=1000)
            #bf = stack.stack_events(EoS1=eos,EoS2="SLY",trials=0)
            uncert = np.std(bf_trials) * 2
            BFs.append(bf)
            uncerts.append(uncert)
        stacks_BFs.append(BFs)
        stacks_uncerts.append(uncerts)

    x_axis = np.arange(len(eosList))
    spacing = [-.3,-.1,.1,.3]
    plt.clf()
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(15, 10))
    for index in range(len(stacks)):
        plt.bar(x_axis+spacing[index],stacks_BFs[index],.15,yerr=stacks_uncerts[index],label=labels[index],color=colors[index])
        #plt.bar(x_axis+spacing[index],stacks_BFs[index],.15,label=labels[index],color=colors[index])

    if log == False: plt.ylim(top=1.2)
    plt.xticks(x_axis,eosList,rotation=45,ha="right")
    ax = plt.gca()
    if log == True: ax.set_yscale("log")
    plt.legend()
    plt.xlabel("EoSs")
    plt.ylabel("Joint Bayes Factor")
    plt.title("EoS Joint Bayes Factors w.r.t. SLY")
    plt.savefig("plots/difPriors/allJoint_barplot_difPriors_BFs.png")

    Dictionary = {labels[Index]:{eosList[eIndex]:[stacks_BFs[Index][eIndex],stacks_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(labels))}
    with open("plots/difPriors/data/allJoint_difPriors_BFs.json","w") as f:
        json.dump(Dictionary, f, indent=2, sort_keys=True)

