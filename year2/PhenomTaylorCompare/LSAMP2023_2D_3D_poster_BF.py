from GWXtreme import eos_model_selection as ems
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import h5py
import convertlambdas

# Collected these useful functions and altered them slightly for the LSAMP poster.
# Thought we'd use the 2D 3D study, but that still need some more time in the oven.
# Still though, might find a plotting alteration useful from here.

def uniform_LTs_Ls_GW170817_data():
    # Uses uP(LTs) produced GW170817 (L~,q) posterior, turns it into a (L1,L2,q) 
    # posterior. That way we have the required posteriors to code around of, and
    # make the 2D/3D kde code that will use the legit uP(LTs) and uP(Ls) posteriors.

    # This is wrong to reiterate. We just did it to test our code's logic. 
    # We NEED the real uP(Ls) GW170817 posterior to proceed!

    Dir = "../bilby_runs/3dkde_studies/outdir/real"

    _data = np.recfromtxt("../compare_GWXtreme/posterior_samples/posterior_samples_narrow_spin_prior.dat",names=True)
    (q,mc,LambdaT)=(np.array(_data['q']),np.array(_data['mc_source']),
                    np.array(_data['lambdat']))
    m1, m2 = ems.getMasses(q, mc)
    sm = (m1*m2)/((m1+m2)**2)
    DLambdaT = np.array(_data['dlambdat'])

    converter = convertlambdas.LambdasInversion(sm,LambdaT,DLambdaT)
    Lambda1, Lambda2 = converter.solve_system()

    LTs_dict = {'posterior':{'content':{'m1_source':m1.tolist(),'m2_source':m2.tolist(),'mass_ratio':q.tolist(),'chirp_mass':mc.tolist(),'delta_lambda_tilde':DLambdaT.tolist(),'lambda_tilde':LambdaT.tolist()}}}
    fname = Dir+"/uniformP_LTs/GW170817/data.json"
    with open(fname,"w") as f:
        json.dump(LTs_dict,f,indent=2,sort_keys=True)

    Ls_dict = {'posterior':{'content':{'m1_source':m1.tolist(),'m2_source':m2.tolist(),'mass_ratio':q.tolist(),'chirp_mass':mc.tolist(),'lambda_1':Lambda1,'lambda_2':Lambda2,'delta_lambda_tilde':DLambdaT.tolist(),'lambda_tilde':LambdaT.tolist()}}}
    fname = Dir+"/uniformP_Ls/GW170817/data.json"
    with open(fname,"w") as f:
        json.dump(Ls_dict,f,indent=2,sort_keys=True)

def singleEventBFs():
    # Plots barplot of BFs using GW170817's uP(LTs) posterior and
    #                                       uP(Ls) posterior (with errorbars)!

    uLTs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/outdir/real/uniformP_LTs/GW170817/data.json" 
    uLs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/GW170817_prior_L1L2/outdir/GW170817_result.json"

    modsel_uLTs = ems.Model_selection(uLTs_File,Ns=4000,kdedim=2)
    modsel_uLs = ems.Model_selection(uLs_File,Ns=4000,kdedim=3)

    labels = ["2D KDE", "3D KDE","Actual"]
    colors = ["#1b9e77","#d95f02","#7570b3"]
    methods = [modsel_uLTs, modsel_uLs]
    #eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP","MPA1","MS1_PP","MS1B_PP"]
    eosList = ["BHF_BBB2","KDE0V","SKOP"]
    
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
    plt.rcParams.update({"font.size":20})
    plt.figure(figsize=(15, 10))
    for index in range(len(methods)):
        plt.bar(x_axis+spacing[index],methods_BFs[index],.15,yerr=methods_uncerts[index],label=labels[index],color=colors[index])
        #plt.bar(x_axis+spacing[index],methods_BFs[index],.15,label=labels[index],color=colors[index])
    
    plt.bar(x_axis+spacing[2],nest_BFs,.15,yerr=nest_stds,label=labels[2],color=colors[2])

    plt.yscale("log")
    plt.xticks(x_axis,eosList,rotation=45,ha="right")
    plt.ylim(1.0e-4,(max(BFs)+max(uncerts))*10.)
    plt.axhline(1.)
    plt.title("EoS Bayes Factors w.r.t. SLY")
    plt.ylabel("Bayes Factor")
    plt.legend()
    label = "GW170817"
    plt.savefig("plots/2D_3D/{}_barplot_2D_3D_BFs.png".format(label))

    Dictionary = {labels[Index]:{eosList[eIndex]:[methods_BFs[Index][eIndex],methods_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(methods))}
    with open("plots/2D_3D/data/{}_2D_3D_BFs.json".format(label),"w") as f:
        json.dump(Dictionary, f, indent=2, sort_keys=True)

