# Make bar plot comparing bayes factors from uniformP (dLT,LT) (2D KDE)
#                                        to uniformP (L1,L2) (3D KDE)

from GWXtreme import eos_model_selection as ems
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import h5py
import convertlambdas

def uniform_LTs_Ls_GW170817_data():

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

def singleEventBFs(log=False):

    Dir = "../bilby_runs/3dkde_studies/outdir/real"
    injections = ["GW170817"]
    filenameEnd = "data.json"
    for injection in injections:
        print(injection)
        uLTs_File = "{}/uniformP_LTs/{}/{}".format(Dir,injection,filenameEnd)
        uLs_File = "{}/uniformP_Ls/{}/{}".format(Dir,injection,filenameEnd)

        modsel_uLTs = ems.Model_selection(uLTs_File,Ns=4000,kdedim=2)
        modsel_uLs = ems.Model_selection(uLs_File,Ns=4000,kdedim=3)

        labels = ["2D Approximation", "3D Approximation","Actual"]
        colors = ["#1b9e77","#d95f02","#7570b3"]
        methods = [modsel_uLTs, modsel_uLs]
        eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP","MPA1","MS1_PP","MS1B_PP"]
        

        with open("nested_sampling_results.json","r") as f:
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
                bf, bf_trials = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=1000)
                #bf = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=0)
                uncert = np.std(bf_trials) * 2
                BFs.append(bf)
                uncerts.append(uncert)
            methods_BFs.append(BFs)
            methods_uncerts.append(uncerts)

        x_axis = np.arange(len(eosList))
        spacing = [-.15,0,.15]
        plt.clf()
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(15, 10))
        for index in range(len(methods)):
            plt.bar(x_axis+spacing[index],methods_BFs[index],.15,yerr=methods_uncerts[index],label=labels[index],color=colors[index])
            #plt.bar(x_axis+spacing[index],methods_BFs[index],.15,label=labels[index],color=colors[index])
        
        plt.bar(x_axis+spacing[2],nest_BFs,.15,yerr=nest_stds,label=labels[2],color=colors[2])

        if log == False: plt.ylim(top=1.2)
        plt.xticks(x_axis,eosList,rotation=45,ha="right")
        ax = plt.gca()
        if log == True: ax.set_yscale("log")
        plt.legend()
        plt.xlabel("EoSs")
        plt.ylabel("Bayes Factor")
        plt.title("EoS Bayes Factors w.r.t. SLY")
        label = uLTs_File.split('/')[-2]
        plt.savefig("plots/2D_3D/{}_barplot_2D_3D_BFs.png".format(label))

    Dictionary = {labels[Index]:{eosList[eIndex]:[methods_BFs[Index][eIndex],methods_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(methods))}
    with open("plots/2D_3D/data/{}_2D_3D_BFs.json".format(label),"w") as f:
        json.dump(Dictionary, f, indent=2, sort_keys=True)


def multipleEventBFs(log=False):

    uLTs_Dir = "../bilby_runs/3dkde_studies/outdir/1st_Phenom_Taylor/uniformP_LTs/phenom-injections/TaylorF2"
    uLs_Dir = "../bilby_runs/3dkde_studies/outdir/1st_Phenom_Taylor/uniformP_Ls/IMRPhenomPv2_NRTidal/APR4_EPP"
    uLTs_Files = glob.glob("{}/*/*.json".format(uLTs_Dir))
    uLs_Files = glob.glob("{}/*/*.json".format(uLs_Dir))

    stack_uLTs = ems.Stacking(uLTs_Files,kdedim=2)
    stack_uLs = ems.Stacking(uLs_Files,kdedim=3)

    labels = ["2D Approximation", "3D Approximation","Actual"]
    colors = ["#1b9e77","#d95f02","#7570b3"]
    stacks = [stack_uLTs, stack_uLs]
    eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP","MPA1","MS1_PP","MS1B_PP"]


    with open("nested_sampling_results.json","r") as f:
        nestSamp = json.load(f)
    nest_BFs = []
    nest_stds = []
    for eos in eosList:
        nest_BFs.append(nestSamp[eos][0])
        nest_stds.append(nestSamp[eos][1])


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
    spacing = [-.15,0,.15]
    plt.clf()
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(15, 10))
    for index in range(len(stacks)):
        plt.bar(x_axis+spacing[index],stacks_BFs[index],.15,yerr=stacks_uncerts[index],label=labels[index],color=colors[index])
        #plt.bar(x_axis+spacing[index],stacks_BFs[index],.15,label=labels[index],color=colors[index])

    plt.bar(x_axis+spacing[2],nest_BFs,.15,yerr=nest_stds,label=labels[2],color=colors[2])

    if log == False: plt.ylim(top=1.2)
    plt.xticks(x_axis,eosList,rotation=45,ha="right")
    ax = plt.gca()
    if log == True: ax.set_yscale("log")
    plt.legend()
    plt.xlabel("EoSs")
    plt.ylabel("Joint Bayes Factor")
    plt.title("EoS Joint Bayes Factors w.r.t. SLY")
    plt.savefig("plots/2D_3D/allJoint_barplot_2D_3D_BFs_wth_errorProne_rmvd.png")

    Dictionary = {labels[Index]:{eosList[eIndex]:[stacks_BFs[Index][eIndex],stacks_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(stacks))}
    with open("plots/2D_3D/data/allJoint_2D_3D_BFs_wth_errorProne_rmvd.json","w") as f:
        json.dump(Dictionary, f, indent=2, sort_keys=True)

