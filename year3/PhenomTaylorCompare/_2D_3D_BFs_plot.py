from GWXtreme import eos_model_selection as ems
import numpy as np
import matplotlib.pyplot as plt
import json
import glob

def singleEventBFs(log=False):
    # Makes barplot of BFs for a single simulation comparing that of uP(LTs) and
    # uP(Ls) (with errorbars!).

    uLTs_Dir = "../../year2/bilby_runs/simulations/outdir/2nd_Phenom_Taylor/uniformP_LTs/IMRPhenomPv2_NRTidal/APR4_EPP"
    uLs_Dir = "../../year2/bilby_runs/simulations/outdir/2nd_Phenom_Taylor/uniformP_Ls/IMRPhenomPv2_NRTidal/APR4_EPP"
    #injections = ["282_1.58_1.37", "202_1.35_1.14", "179_1.35_1.23", "71_1.37_1.33", "122_1.77_1.19", 
    #              "241_1.31_1.28", "220_1.36_1.24", "282_1.35_1.32", "149_1.35_1.23", "237_1.36_1.26", 
    #              "138_1.5_1.21", "235_1.4_1.3", "219_1.3_1.28", "260_1.48_1.33", "164_1.34_1.19", 
    #              "55_1.38_1.33", "78_1.35_1.32"]
    #injections = ["55_1.54_1.41", "65_1.36_1.17"]
    injections = ["55_1.54_1.41"] # 01/30/24 wanted to test out Anarya's dev-3d-prod branch with my changes
    filenameEnd = "bns_example_result.json"
    for injection in injections:
        print(injection)
        uLTs_File = "{}/{}/{}".format(uLTs_Dir,injection,filenameEnd)
        uLs_File = "{}/{}/{}".format(uLs_Dir,injection,filenameEnd)

        modsel_uLTs = ems.Model_selection(uLTs_File,Ns=4000,kdedim=2)
        modsel_uLs = ems.Model_selection(uLs_File,Ns=4000,kdedim=3)

        labels = ["UniformP (dL~,L~)", "UniformP (L1,L2)"]
        colors = ["#66c2a5","#fc8d62"]
        methods = [modsel_uLTs, modsel_uLs]
        #eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP","MPA1","MS1_PP","MS1B_PP"]
        eosList = ["BHF_BBB2","H4","APR4_EPP","MPA1","MS1_PP"]
        methods_BFs = []
        methods_uncerts = []
        for method in methods:
            print(method)
            BFs = []
            uncerts = []
            for eos in eosList:
                print(eos)
                bf, bf_trials = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=1000)
                #bf = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=0)
                uncert = np.std(bf_trials) * 2
                BFs.append(bf)
                uncerts.append(uncert)
            methods_BFs.append(BFs)
            methods_uncerts.append(uncerts)

        x_axis = np.arange(len(eosList))
        spacing = [-.1,.1]
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
        plt.ylabel("Joint Bayes Factor")
        plt.title("EoS Joint Bayes Factors w.r.t. SLY")
        label = uLTs_File.split('/')[-2]
        #plt.savefig("plots/2D_3D/{}_barplot_2D_3D_BFs.png".format(label))
        plt.savefig("./{}_barplot_2D_3D_BFs.png".format(label)) # 01/30/24 wanted to test out Anarya's dev-3d-prod branch with my changes

    Dictionary = {labels[Index]:{eosList[eIndex]:[methods_BFs[Index][eIndex],methods_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(labels))}
    #with open("plots/2D_3D/data/{}_2D_3D_BFs.json".format(label),"w") as f:
    with open("./{}_2D_3D_BFs.json".format(label),"w") as f: # 01/30/24 wanted to test out Anarya's dev-3d-prod branch with my changes
        json.dump(Dictionary, f, indent=2, sort_keys=True)


def multipleEventBFs(log=False):
    # Makes barplot of jointBFs using all simulations comparing that of uP(LTs) 
    # and uP(Ls) (with errorbars!).

    uLTs_Dir = "../bilby_runs/3dkde_studies/Anarya_uniformLTs/phenom-injections/TaylorF2/"
    uLs_Dir = "../bilby_runs/3dkde_studies/outdir/Phenom_Taylor/IMRPhenomPv2_NRTidal/APR4_EPP/"
    uLTs_Files = glob.glob("{}/*/*.json".format(uLTs_Dir))
    uLs_Files = glob.glob("{}/*/*.json".format(uLs_Dir))

    stack_uLTs = ems.Stacking(uLTs_Files,kdedim=2)
    stack_uLs = ems.Stacking(uLs_Files,kdedim=3)

    labels = ["UniformP (dL~,L~)", "UniformP (L1,L2)"]
    colors = ["#66c2a5","#fc8d62"]
    stacks = [stack_uLTs, stack_uLs]
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
    spacing = [-.1,.1]
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
    plt.savefig("plots/2D_3D/allJoint_barplot_2D_3D_BFs.png")

    Dictionary = {labels[Index]:{eosList[eIndex]:[stacks_BFs[Index][eIndex],stacks_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(labels))}
    with open("plots/2D_3D/data/allJoint_2D_3D_BFs.json","w") as f:
        json.dump(Dictionary, f, indent=2, sort_keys=True)

