from GWXtreme import eos_model_selection as ems
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os.path

def singleEventBFs(Trials=1000):

    #labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    labels = ["3D KDE PhenomPv2"]
    #uLTs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_LTs/phenom-injections/TaylorF2"
    #uLs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_Ls/IMRPhenomPv2_NRTidal/APR4_EPP"
    #phenomPhenom_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Phenom/IMRPhenomPv2_NRTidal/APR4_EPP"
    nsbhPhenom_Dir = './files/NSBH/IMRPhenomPv2_NRTidal/APR4_EPP'

#    injections = ["282_1.58_1.37", "202_1.35_1.14", "179_1.35_1.23", "122_1.77_1.19", 
#                  "71_1.37_1.33", "55_1.38_1.33", "78_1.35_1.32",
#                  "241_1.31_1.28", "220_1.36_1.24", "282_1.35_1.32", "149_1.35_1.23", "237_1.36_1.26", 
#                  "138_1.5_1.21", "235_1.4_1.3", "219_1.3_1.28", "260_1.48_1.33", "164_1.34_1.19"]

    injections = ['103_8.82_1.23','116_9.83_1.15','131_2.2_1.53','177_9.59_1.97','196_3.34_2.13',
                  '227_4.19_2.05','236_7.03_1.96','261_4.16_2.08','267_4.47_1.66','321_3.11_2.08',
                  '327_3.11_1.22','380_7.95_2.1','386_2.64_1.81','432_3.51_1.94','452_3.55_1.47',
                  '455_2.33_1.97','467_5.58_2.06','756_7.0_1.58']
    #filenameEnd = "bns_example_result.json"
    filenameEnd = "bns_example_result_simplified.json"

    index = 0
    for injection in injections:
        print(injection)

        try:
            #uLTs_File = "{}/{}/{}".format(uLTs_Dir,injection,filenameEnd)
            #uLs_File = "{}/{}/{}".format(uLs_Dir,injection,filenameEnd)
            #phenomPhenom_File = "{}/{}/{}".format(phenomPhenom_Dir,injection,filenameEnd)
            nsbhPhenom_File = "{}/{}/{}".format(nsbhPhenom_Dir,injection,filenameEnd)

            #modsel_uLTs = ems.Model_selection(uLTs_File,Ns=4000,kdedim=2)
            #modsel_uLs = ems.Model_selection(uLs_File,Ns=4000,kdedim=3)
            #modsel_phenomPhenom = ems.Model_selection(phenomPhenom_File,Ns=4000,kdedim=3)
            modsel_nsbhPhenom = ems.Model_selection(nsbhPhenom_File,Ns=4000,kdedim=3)

        except FileNotFoundError:
            uLTs_File = "{}/troublesome/{}/{}".format(uLTs_Dir,injection,filenameEnd)
            uLs_File = "{}/troublesome/{}/{}".format(uLs_Dir,injection,filenameEnd)
            phenomPhenom_File = "{}/troublesome/{}/{}".format(phenomPhenom_Dir,injection,filenameEnd)

            modsel_uLTs = ems.Model_selection(uLTs_File,Ns=4000,kdedim=2)
            modsel_uLs = ems.Model_selection(uLs_File,Ns=4000,kdedim=3)
            modsel_phenomPhenom = ems.Model_selection(phenomPhenom_File,Ns=4000,kdedim=3)

        #methods = [modsel_uLTs, modsel_uLs,modsel_phenomPhenom]
        methods = [modsel_nsbhPhenom]
        eosList = ["SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
        methods_BFs = []
        methods_trials = []
        for method in methods:
            print(method)
            BFs = []
            trials = []
            for eos in eosList:
                print(eos)
                bf, bf_trials = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=Trials)
                print(bf)
                #bf = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=0)
                BFs.append(bf)
                trials.append(bf_trials.tolist())
            methods_BFs.append(BFs)
            methods_trials.append(trials)
        
        output = "data/NSBH/BFs/{}_BFs_100samp.json".format(injection)

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

    Dir = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/data/NSBH/BFs"
#    injections = ["282_1.58_1.37", "202_1.35_1.14", "179_1.35_1.23", "122_1.77_1.19", 
#                  "71_1.37_1.33", "55_1.38_1.33", "78_1.35_1.32",
#                  "241_1.31_1.28", "220_1.36_1.24", "282_1.35_1.32", "149_1.35_1.23", "237_1.36_1.26", 
#                  "138_1.5_1.21", "235_1.4_1.3", "219_1.3_1.28", "260_1.48_1.33", "164_1.34_1.19"]

    injections = ['103_8.82_1.23','116_9.83_1.15','131_2.2_1.53','177_9.59_1.97','196_3.34_2.13',
                  '227_4.19_2.05','236_7.03_1.96','261_4.16_2.08','267_4.47_1.66','321_3.11_2.08',
                  '327_3.11_1.22','380_7.95_2.1','386_2.64_1.81','432_3.51_1.94','452_3.55_1.47',
                  '455_2.33_1.97','467_5.58_2.06','756_7.0_1.58']

    for injection in injections:

        File = "{}/{}_BFs_100samp.json".format(Dir,injection)
        with open(File,"r") as f:
            data = json.load(f)

        #labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
        labels = ["3D KDE PhenomPv2"]
        eosList = ["SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
        colors = ["#d7191c"]
        x_axis = np.arange(len(eosList))
        #spacing = [-.20,0.,.20]
        spacing = [0]
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
        plt.savefig("plots/NSBH/BFs/{}_2D_3D_BFs_100samp.png".format(injection), bbox_inches="tight")

def multipleEventBFs(Trials=1000):

    #uLTs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_LTs/phenom-injections/TaylorF2"
    #uLs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_Ls/IMRPhenomPv2_NRTidal/APR4_EPP"
    #phenomPhenom_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Phenom/IMRPhenomPv2_NRTidal/APR4_EPP"
    nsbhPhenom_Dir = '/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/files/NSBH/IMRPhenomPv2_NRTidal/APR4_EPP'

    # Seems that only 13 of the events aren't called troublesome. Ig those caused extremely small BFs.
    #uLTs_Files = glob.glob("{}/*/*simplified.json".format(uLTs_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(uLTs_Dir))
    #uLs_Files = glob.glob("{}/*/*simplified.json".format(uLs_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(uLs_Dir))
    #phenomPhenom_Files = glob.glob("{}/*/*simplified.json".format(phenomPhenom_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(phenomPhenom_Dir))
    nsbhPhenom_Files = glob.glob("{}/*/*simplified.json".format(nsbhPhenom_Dir))

    #stack_uLTs = ems.Stacking(uLTs_Files,kdedim=2)
    #stack_uLs = ems.Stacking(uLs_Files,kdedim=3)
    #stack_phenomPhenom = ems.Stacking(phenomPhenom_Files,kdedim=3)
    stack_nsbhPhenom = ems.Stacking(nsbhPhenom_Files,kdedim=3, Ns=4000)

    #output = "data/BNS/BFs/16simulations_2D_3D_BFs_1000trial.json"
    output = "data/NSBH/BFs/18simulations_BFs_100trial.json"

    #labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    labels = ["3D KDE PhenomPv2"]
    #stacks = [stack_uLTs, stack_uLs, stack_phenomPhenom]
    stacks = [stack_nsbhPhenom]
    eosList = ["SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
    stacks_BFs = []
    stacks_uncerts = []
    stacks_uncerts2 = []
    for stack in stacks:
        print(stack)
        BFs = []
        uncerts = []
        uncerts2 = []
        for eos in eosList:
            print(eos)
            bf, bf_trials = stack.stack_events(EoS1=eos,EoS2="SLY",trials=Trials) # bf_trials here is the joint bf array
            events_errors = stack.all_bayes_factors_errors # each events separate error
            #bf = stack.stack_events(EoS1=eos,EoS2="SLY",trials=0)
            uncert = np.std(bf_trials) * 2 # np.std of joint BF array (traditional method)
            uncert2 = np.std(events_errors) * 2 # double np.std
            BFs.append(bf)
            #uncerts.append(np.nan) # 0 TRIAL CASE
            uncerts.append(uncert)
            uncerts2.append(uncert)
        stacks_BFs.append(BFs)
        stacks_uncerts.append(uncerts)
        stacks_uncerts2.append(uncerts2)

    #TEST THIS SOON PLEASE BEFORE USING
    if os.path.isfile(output) == True:
        with open(output,"r") as f:
            Dictionary = json.load(f)

        for Index in range(len(labels)):
            dictionary = {}
            for eIndex in range(len(eosList)):
                dictionary[eosList[eIndex]] = [stacks_BFs[Index][eIndex],stacks_uncerts[Index][eIndex],stacks_uncerts2[Index][eIndex]]
            Dictionary[labels[Index]] = dictionary

        with open(output,"w") as f:
            json.dump(Dictionary, f, indent=2, sort_keys=True)

    else: # First time doing this sort of run so new file is made

        Dictionary = {labels[Index]:{eosList[eIndex]:[stacks_BFs[Index][eIndex],stacks_uncerts[Index][eIndex],stacks_uncerts2[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(labels))}
        with open(output,"w") as f:
            json.dump(Dictionary, f, indent=2, sort_keys=True)


def multipleEventPlots():
     
    #File = "data/BNS/BFs/16simulations_2D_3D_BFs_1000trial.json"
    File = "data/NSBH/BFs/18simulations_BFs_100trial.json"
    with open(File,"r") as f:
        data = json.load(f)
 
    #labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    labels = ["3D KDE PhenomPv2"]
    eosList = ["SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1B_PP","MS1_PP"]
    #colors = ['#ffffb3','#bebada','#fb8072']
    colors = ["#d7191c"]
    x_axis = np.arange(len(eosList))
    #spacing = [-.20,0.,.20]
    spacing = [.0]

    plt.clf()
    plt.rcParams.update({"font.size":18})
    plt.figure(figsize=(15, 10))
 
    counter = 0
    for label in labels:
 
        BFs = []
        uncerts = []
        for eos in eosList:
            BFs.append(data[label][eos][0])
            uncert = data[label][eos][-1]
            uncerts.append(uncert)
 
        plt.bar(x_axis+spacing[counter],BFs,.20,label=labels[counter],color=colors[counter])
        plt.errorbar(x_axis+spacing[counter],BFs,yerr=uncerts,ls="none",ecolor="black")
        counter += 1

    plt.yscale("log")
    plt.xticks(x_axis,eosList,rotation=90,ha="right")
    plt.ylim(1.0e-3,(max(BFs)+max(uncerts))*10.)
    plt.axhline(1.0,color="k",linestyle="--",alpha=0.2)
    plt.ylabel("Joint Bayes-factor w.r.t SLY")
    plt.legend()
    #plt.savefig("plots/BNS/BFs/16simulations_2D_3D_BFs_1000trial.png",bbox_inches="tight")
    plt.savefig("plots/NSBH/BFs/18simulations_BFs_100trial.png",bbox_inches="tight")

