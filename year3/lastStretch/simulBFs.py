from GWXtreme import eos_model_selection as ems
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os.path

def singleEventBFs(Trials=1000):

    labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    uLTs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_LTs/phenom-injections/TaylorF2"
    uLs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_Ls/IMRPhenomPv2_NRTidal/APR4_EPP"
    phenomPhenom_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Phenom/IMRPhenomPv2_NRTidal/APR4_EPP"

    injections = ["282_1.58_1.37", "202_1.35_1.14", "179_1.35_1.23", "122_1.77_1.19", 
                  "71_1.37_1.33", "55_1.38_1.33", "78_1.35_1.32",
                  "241_1.31_1.28", "220_1.36_1.24", "282_1.35_1.32", "149_1.35_1.23", "237_1.36_1.26", 
                  "138_1.5_1.21", "235_1.4_1.3", "219_1.3_1.28", "260_1.48_1.33", "164_1.34_1.19"]

    #filenameEnd = "bns_example_result.json"
    filenameEnd = "bns_example_result_simplified.json"

    index = 0
    for injection in injections:
        print(injection)

        try:
            uLTs_File = "{}/{}/{}".format(uLTs_Dir,injection,filenameEnd)
            uLs_File = "{}/{}/{}".format(uLs_Dir,injection,filenameEnd)
            phenomPhenom_File = "{}/{}/{}".format(phenomPhenom_Dir,injection,filenameEnd)

            modsel_uLTs = ems.Model_selection(uLTs_File,Ns=4000,kdedim=2)
            modsel_uLs = ems.Model_selection(uLs_File,Ns=4000,kdedim=3)
            modsel_phenomPhenom = ems.Model_selection(phenomPhenom_File,Ns=4000,kdedim=3)

        except FileNotFoundError:
            uLTs_File = "{}/troublesome/{}/{}".format(uLTs_Dir,injection,filenameEnd)
            uLs_File = "{}/troublesome/{}/{}".format(uLs_Dir,injection,filenameEnd)
            phenomPhenom_File = "{}/troublesome/{}/{}".format(phenomPhenom_Dir,injection,filenameEnd)

            modsel_uLTs = ems.Model_selection(uLTs_File,Ns=4000,kdedim=2)
            modsel_uLs = ems.Model_selection(uLs_File,Ns=4000,kdedim=3)
            modsel_phenomPhenom = ems.Model_selection(phenomPhenom_File,Ns=4000,kdedim=3)

        methods = [modsel_uLTs, modsel_uLs,modsel_phenomPhenom]
        eosList = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
        methods_BFs = []
        methods_trials = []
        for method in methods:
            print(method)
            BFs = []
            trials = []
            for eos in eosList:
                print(eos)
                bf, bf_trials = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=Trials)
                #bf = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=0)
                BFs.append(bf)
                trials.append(bf_trials.tolist())
            methods_BFs.append(BFs)
            methods_trials.append(trials)
        
        output = "data/BFs/{}_2D_3D_BFs.json".format(injection)

        if os.path.isfile(output) == True:
            with open(output,"r") as f:
                Dictionary = json.load(f)

            for Index in range(len(labels)):
                dictionary = {}
                for eIndex in range(len(eosList)):
                    dictionary[eosList[eIndex]] = [methods_BFs[Index][eIndex],methods_trials[Index][eIndex]]
                Dictionary[labels[Index]] = dictionary

            with open(output,"w") as f:
                json.dump(Dictionary, f, indent=2, sort_keys=True)

        else: # First time doing this sort of run so new file is made@
            Dictionary = {labels[Index]:{eosList[eIndex]:[methods_BFs[Index][eIndex],methods_trials[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(labels))}
            with open(output,"w") as f:
                json.dump(Dictionary, f, indent=2, sort_keys=True)

def singleEventPlots():

    Dir = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/data/BFs"
    injections = ["282_1.58_1.37", "202_1.35_1.14", "179_1.35_1.23", "122_1.77_1.19", 
                  "71_1.37_1.33", "55_1.38_1.33", "78_1.35_1.32",
                  "241_1.31_1.28", "220_1.36_1.24", "282_1.35_1.32", "149_1.35_1.23", "237_1.36_1.26", 
                  "138_1.5_1.21", "235_1.4_1.3", "219_1.3_1.28", "260_1.48_1.33", "164_1.34_1.19"]

    for injection in injections:

        File = "{}/{}_2D_3D_BFs.json".format(Dir,injection)
        with open(File,"r") as f:
            data = json.load(f)

        labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
        eosList = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
        colors = ["#d7191c","#fdae61","#abdda4"]
        x_axis = np.arange(len(eosList))
        spacing = [-.20,0.,.20]
        plt.clf()
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(15, 10))

        counter = 0
        for label in labels:

            BFs = []
            uncerts = []
            for eos in eosList:
                BFs.append(data[label][eos][0]) 
                trials = np.array(data[label][eos][1])
                uncert = np.std(trials) * 2
                uncerts.append(uncert)

            plt.bar(x_axis+spacing[counter],BFs,.20,label=labels[counter],color=colors[counter])
            plt.errorbar(x_axis+spacing[counter],BFs,yerr=uncerts,ls="none",ecolor="black")
            counter += 1

        plt.yscale("log")
        plt.xticks(x_axis,eosList,rotation=90,ha="right")
        plt.ylim(1.0e-3,(max(BFs)+max(uncerts))*10.)
        plt.axhline(1.0,color="k",linestyle="--",alpha=0.2)
        plt.ylabel("Bayes-factor w.r.t SLY")
        plt.legend()
        plt.savefig("plots/BFs/{}_2D_3D_BFs.png".format(injection), bbox_inches="tight")

def multipleEventBFs(Trials=1000):

    uLTs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_LTs/phenom-injections/TaylorF2"
    uLs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_Ls/IMRPhenomPv2_NRTidal/APR4_EPP"
    phenomPhenom_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Phenom/IMRPhenomPv2_NRTidal/APR4_EPP"

    # Seems that only 13 of the events aren't called troublesome. Ig those caused extremely small BFs.
    #uLTs_Files = glob.glob("{}/*/*simplified.json".format(uLTs_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(uLTs_Dir))
    #uLs_Files = glob.glob("{}/*/*simplified.json".format(uLs_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(uLs_Dir))
    #phenomPhenom_Files = glob.glob("{}/*/*simplified.json".format(phenomPhenom_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(phenomPhenom_Dir))
    uLTs_Files = glob.glob("{}/*/*simplified.json".format(uLTs_Dir))[:2] 
    uLs_Files = glob.glob("{}/*/*simplified.json".format(uLs_Dir))[:2]
    phenomPhenom_Files = glob.glob("{}/*/*simplified.json".format(phenomPhenom_Dir))[:2]

    stack_uLTs = ems.Stacking(uLTs_Files,kdedim=2)
    stack_uLs = ems.Stacking(uLs_Files,kdedim=3)
    stack_phenomPhenom = ems.Stacking(phenomPhenom_Files,kdedim=3)

    output = "data/BFs/3simulations_2D_3D_BFs_100trial.json"

    labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    stacks = [stack_uLTs, stack_uLs, stack_phenomPhenom]
    eosList = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
    eosList = ["BHF_BBB2","KDE0V"]
    stacks_BFs = []
    stacks_uncerts = []
    for stack in stacks:
        print(stack)
        BFs = []
        uncerts = []
        for eos in eosList:
            print(eos)
            bf, bf_trials = stack.stack_events(EoS1=eos,EoS2="SLY",trials=Trials)
            #bf = stack.stack_events(EoS1=eos,EoS2="SLY",trials=0)
            uncert = np.std(bf_trials) * 2
            BFs.append(bf)
            uncerts.append(uncert)
        stacks_BFs.append(BFs)
        stacks_uncerts.append(uncerts)

    #TEST THIS SOON PLEASE BEFORE USING
    if os.path.isfile(output) == True:
        with open(output,"r") as f:
            Dictionary = json.load(f)

        for Index in range(len(labels)):
            dictionary = {}
            for eIndex in range(len(eosList)):
                dictionary[eosList[eIndex]] = [stacks_BFs[Index][eIndex],stacks_uncerts[Index][eIndex]]
            Dictionary[labels[Index]] = dictionary

        with open(output,"w") as f:
            json.dump(Dictionary, f, indent=2, sort_keys=True)

    else: # First time doing this sort of run so new file is made

        Dictionary = {labels[Index]:{eosList[eIndex]:[stacks_BFs[Index][eIndex],stacks_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(labels))}
        with open(output,"w") as f:
            json.dump(Dictionary, f, indent=2, sort_keys=True)


def multipleEventPlots():
     
    File = "data/BFs/13simulations_2D_3D_BFs_100trial.json"
    with open(File,"r") as f:
        data = json.load(f)
 
    labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    eosList = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
    colors = ["#d7191c","#fdae61","#abdda4"]
    x_axis = np.arange(len(eosList))
    spacing = [-.20,0.,.20]

    plt.clf()
    plt.rcParams.update({"font.size":18})
    plt.figure(figsize=(15, 10))
 
    counter = 0
    for label in labels:
 
        BFs = []
        uncerts = []
        for eos in eosList:
            BFs.append(data[label][eos][0])
            if len(data[label][eos][1]) != 1:
                trials = np.array(data[label][eos][1])
                uncert = np.std(trials) * 2
                uncerts.append(uncert)
            else:
                uncert = data[label][eos][1][0]
                uncerts.append(uncert)
 
        plt.bar(x_axis+spacing[counter],BFs,.20,label=labels[counter],color=colors[counter])
        plt.errorbar(x_axis+spacing[counter],BFs,yerr=uncerts,ls="none",ecolor="black")
        counter += 1

    plt.yscale("log")
    plt.xticks(x_axis,eosList,rotation=90,ha="right")
    plt.ylim(1.0e-3,(max(BFs)+max(uncerts))*10.)
    plt.axhline(1.0,color="k",linestyle="--",alpha=0.2)
    plt.ylabel("Bayes-factor w.r.t SLY")
    plt.legend()
    plt.savefig("plots/BFs/13simulations_2D_3D_BFs_100trial.png",bbox_inches="tight")

