from GWXtreme import eos_model_selection as ems
from GWXtreme.parametrized_eos_sampler import mcmc_sampler
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# DO NOT DELETE ANYTHING THAT IS COMMENTED OUT. I'M A LAZY CODER AND JUST COMMENT
# STUFF IN AND OUT INSTEAD OF OVERENGINEERING A SCRIPT.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def singleEventBFs(Trials=1000):

    #uLTs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/outdir/real/uniformP_LTs/GW170817/simplified_result.json" 
    #uLs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year3/GW170817_prior_L1L2/CIT_attempt_successful/outdir/simplified_result.json"
    uLs_phenom_File = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/files/GW170817phenom.json"
    output = "plots/data/GW170817_2D_3D_BFs.json"

    #modsel_uLTs = ems.Model_selection(uLTs_File,Ns=4000,kdedim=2)
    #modsel_uLs = ems.Model_selection(uLs_File,Ns=4000,kdedim=3)
    modsel_phenom_uLs = ems.Model_selection(uLs_phenom_File,Ns=4000,kdedim=3)

    #labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT", "LALInference_Nest"]
    labels = ["3D KDE PhenomNRT"]
    #methods = [modsel_uLTs, modsel_uLs]
    methods = [modsel_phenom_uLs]
    eosList = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]

    with open("/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/outdir/nested_sampling_results.json","r") as f:
        nestSamp = json.load(f)
    nest_BFs = []
    nest_stds = []
    for eos in eosList:
        nest_BFs.append(nestSamp[eos][0])
        nest_stds.append([nestSamp[eos][1]])

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

    methods_BFs.append(nest_BFs)
    methods_trials.append(nest_stds)


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

    else: # First time doing this sort of run so new file is made
        Dictionary = {labels[Index]:{eosList[eIndex]:[methods_BFs[Index][eIndex],methods_trials[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(labels))}
        with open(output,"w") as f:
            json.dump(Dictionary, f, indent=2, sort_keys=True)

def singleEventPlots():

    File = "plots/data/GW170817_2D_3D_BFs.json"
    with open(File,"r") as f:
        data = json.load(f)


    labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT", "LALInference_Nest"]
    eosList = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1_PP","MS1B_PP"]
    colors = ["#d7191c","#fdae61","#abdda4","#2b83ba"]
#    colors = ["#1f77b4", "#ffd7b6", "#bfe2bf"]
    x_axis = np.arange(len(eosList))
    spacing = [-.30,-.10,.10,.30]
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
    plt.ylim(1.0e-4,(max(BFs)+max(uncerts))*10.)
    plt.axhline(1.0,color="k",linestyle="--",alpha=0.2)
    #plt.title("EoS Bayes Factors w.r.t. SLY")
    plt.ylabel("Bayes-factor w.r.t SLY")
    plt.legend()
    plt.savefig("plots/GW170817_barplot_2D_3D_BFs.png",bbox_inches="tight")

def constraint():
    # Adopted from the driver provided in GWXtreme's git. That said it may be restrictive.
    # I don't think I can alter the "look" of the plot, and overlaying different constraints
    # may be harder than the simple use of fig.show(). Old script used to this.

    #Array Containing list of paths to the .dat files  containing the posterior samples for the events:
    uLTs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/outdir/real/uniformP_LTs/GW170817/simplified_result.json" 
    uLs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year3/GW170817_prior_L1L2/CIT_attempt_successful/outdir/simplified_result.json"
    uLs_phenom_File = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/files/GW170817phenom.json"
    filesToCompare = [uLTs_File,uLs_File,uLs_phenom_File]
    labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    Labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
    dims = [2,3,3]

    for ii in range(len(filesToCompare)):

        fnames=[filesToCompare[ii]]
        #Name of/ Path to file in which EoS parameter posterior samples will be saved:
        outname='plots/constraints/data/{}_GW170817inference'.format(Labels[ii])

        #Initialize Sampler Object:
        """For SPectral"""
        sampler=mcmc_sampler(fnames, {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}, outname, nwalkers=10, Nsamples=1000, ndim=4, spectral=True,npool=100,kdedim=dims[ii])

        #Run, Save , Plot
        sampler.initialize_walkers()
        sampler.run_sampler()
        sampler.save_data()

        fig=sampler.plot(cornerplot={'plot':True,'true vals':None},p_vs_rho={'plot':True,'true_eos':'AP4'})
        fig['corner'].savefig('plots/constraints/corners/{}_GW170817_corner.png'.format(Labels[ii]))
        fig['p_vs_rho'][0].savefig('plots/constraints/{}_GW170817_constraint.png'.format(Labels[ii]))
        fig['p_vs_rho'][0].show()

    plt.savefig('plots/constraints/GW170817_comparison.png')

