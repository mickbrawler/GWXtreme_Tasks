from GWXtreme import eos_model_selection as ems
import numpy as np
import matplotlib.pyplot as plt
import json
import glob

def singleEventBFs(Trials=1000):

    labels = ["TaylorF2 Prior 2D KDE", "TaylorF2 Prior 3D KDE", "PhenomNRT Prior 3D KDE"]
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
        
        output = "plots/data/{}_2D_3D_BFs.json".format(injection)

        # STILL UNTESTED # STILL UNTESTED # STILL UNTESTED # STILL UNTESTED # STILL UNTESTED
        # If you've already done this run, likely for different waveforms/priors,
        # it will append the data to the current file under its label.
        # If same labels are used though, overwriting of that field will occur.
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

    Dir = "/home/michael/projects/eos/GWXtreme_Tasks/year3/PhenomTaylorCompare/plots/data/"
#    injections = ["282_1.58_1.37", "202_1.35_1.14", "179_1.35_1.23", "122_1.77_1.19", 
#                  "71_1.37_1.33", "55_1.38_1.33", "78_1.35_1.32",
#                  "241_1.31_1.28", "220_1.36_1.24", "282_1.35_1.32", "149_1.35_1.23", "237_1.36_1.26", 
#                  "138_1.5_1.21", "235_1.4_1.3", "219_1.3_1.28", "260_1.48_1.33", "164_1.34_1.19"]
    injections = ["282_1.58_1.37", "122_1.77_1.19", "55_1.38_1.33"] 

    for injection in injections:

        File = "{}/{}_2D_3D_BFs.json".format(Dir,injection)
        with open(File,"r") as f:
            data = json.load(f)


#        labels = ["Phenom-Taylor (dL~,L~) Uniform Prior", "(Phenom-Taylor (L1,L2) Uniform Prior", "(Phenom-Phenom (L1,L2) Uniform Prior"]
        labels = ["Phenom-Taylor (dL~,L~) Uniform Prior", "(Phenom-Taylor (L1,L2) Uniform Prior"]
        Labels = ["2D KDE (GWXtreme)", "3D KDE (GWXtreme)"]
        eosList = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
        #colors = ["#66c2a5","#fc8d62"] # Colors we initially used
        #colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        #colors = ["#e41a1c", "#377eb8"]
        colors = ["#66c2a5", "#fc8d62"]
        x_axis = np.arange(len(eosList))
#        spacing = [-.3,-.1,.1,.3]
        spacing = [-.10,.10]
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

            plt.bar(x_axis+spacing[counter],BFs,.20,label=Labels[counter],color=colors[counter])
            plt.errorbar(x_axis+spacing[counter],BFs,yerr=uncerts,ls="none",ecolor="black")
            counter += 1

        plt.yscale("log")
        plt.xticks(x_axis,eosList,rotation=90,ha="right")
        plt.ylim(1.0e-4,(max(BFs)+max(uncerts))*10.)
        plt.axhline(1.0,color="k",linestyle="--",alpha=0.2)
        plt.ylabel("Bayes-factor w.r.t SLY")
        plt.legend()
        plt.savefig("plots/{}_barplot_2D_3D_BFs.png".format(injection), bbox_inches="tight")


# Still have to adopt above logic for a joint BF plot. Test above first before proceeding.
def multipleEventBFs():

    uLTs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_LTs/phenom-injections/TaylorF2"
    uLs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_Ls/IMRPhenomPv2_NRTidal/APR4_EPP"

    uLTs_Files = glob.glob("{}/*/*simplified.json".format(uLTs_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(uLTs_Dir))
    uLs_Files = glob.glob("{}/*/*simplified.json".format(uLs_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(uLs_Dir))

    stack_uLTs = ems.Stacking(uLTs_Files,kdedim=2)
    stack_uLs = ems.Stacking(uLs_Files,kdedim=3)

    labels = ["(dL~,L~) Uniform Prior", "(L1,L2) Uniform Prior"]
    #colors = ["#66c2a5","#fc8d62"] # Colors we used to use
    colors = ["#1f78b4", "#b2df8a"]
    stacks = [stack_uLTs, stack_uLs]
    eosList = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
    stacks_BFs = []
    stacks_uncerts = []
    for stack in stacks:
        print(stack)
        BFs = []
        uncerts = []
        for eos in eosList:
            print(eos)
            bf, bf_trials = stack.stack_events(EoS1=eos,EoS2="SLY",trials=100)
            #bf = stack.stack_events(EoS1=eos,EoS2="SLY",trials=0)
            uncert = np.std(bf_trials) * 2
            BFs.append(bf)
            uncerts.append(uncert)
        stacks_BFs.append(BFs)
        stacks_uncerts.append(uncerts)

    x_axis = np.arange(len(eosList))
    spacing = [-.075,.075]
    plt.clf()
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(15, 10))
    for index in range(len(stacks)):
        #plt.bar(x_axis+spacing[index],stacks_BFs[index],.15,yerr=stacks_uncerts[index],label=labels[index],color=colors[index])
        plt.bar(x_axis+spacing[index],stacks_BFs[index],.15,label=labels[index],color=colors[index])

        plt.errorbar(x_axis+spacing[index],stacks_BFs[index],yerr=stacks_uncerts[index],ls="none",ecolor="black")

    plt.yscale("log")
    plt.xticks(x_axis,eosList,rotation=45,ha="right")
    #plt.axhline(1.0,color="k",linestyle="--")
    plt.ylabel("Bayes-factor w.r.t SLY")
    plt.legend()
    plt.savefig("plots/2D_3D/allJoint_barplot_2D_3D_BFs.png",bbox_inches="tight")

    Dictionary = {labels[Index]:{eosList[eIndex]:[stacks_BFs[Index][eIndex],stacks_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(labels))}
    with open("plots/2D_3D/data/allJoint_2D_3D_BFs.json","w") as f:
        json.dump(Dictionary, f, indent=2, sort_keys=True)

