from GWXtreme import eos_model_selection as ems
import numpy as np
import matplotlib.pyplot as plt

def combine_events():

    GW_events = ["anaryaShare/Files/posterior_samples_narrow_spin_prior_170817.dat"]
    EM_events = ["anaryaShare/Files/J0030_3spot_RM.txt","anaryaShare/Files/NICER+XMM_J0740_RM.txt"]
    modsels = ems.Stacking(GW_events,em_event_list=EM_events,spectral=True)
    #eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP","MPA1","MS1_PP","MS1B_PP"]
    #eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP"]
    eosList = ["BHF_BBB2","KDE0V"]

    BFs = []
    uncerts = []
    for eos in eosList:
        bf, bf_trials = modsels.stack_events(EoS1=eos,EoS2="SLY",trials=100)
        #bf = modsels.stack_events(eos,"SLY",trials=0)
        uncert = np.std(bf_trials) * 2
        BFs.append(bf)
        uncerts.append(uncert)

    x_axis = np.arange(len(eosList))
    plt.clf()
    plt.rcParams.update({"font.size":20})
    fig = plt.figure(figsize=(15,10))
    plt.bar(x_axis,BFs,.5,yerr=uncerts,color="#fc8d62")
    #plt.bar(x_axis,BFs,.5,color="#fc8d62")

    plt.yscale('log')
    plt.xticks(x_axis,eosList)
    plt.ylim(1.0e-4,(max(BFs)+max(uncerts))*10.)
    plt.axhline(1.)
    plt.title("EoS Joint Bayes Factors w.r.t. SLY")
    plt.ylabel("Joint Bayes Factor")
    plt.savefig("joint_EM_GW_BFs.png", bbox_inches='tight')

def individual_events():

    modselGW170817 = ems.Model_selection("anaryaShare/Files/posterior_samples_narrow_spin_prior_170817.dat", spectral = True)
    modselJ0030 = ems.Model_selection_em("anaryaShare/Files/J0030_3spot_RM.txt", inverse_mr_prior = ems.inverse_prior_func, spectral = True)
    modselJ0740 = ems.Model_selection_em("anaryaShare/Files/NICER+XMM_J0740_RM.txt", inverse_mr_prior = ems.inverse_prior_func_gaussian_mass, spectral = True)

    methods = [modselGW170817, modselJ0030, modselJ0740]
    labels = ["GW170817","J0030","J0740"]
    #eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP","MPA1","MS1_PP","MS1B_PP"]
    #eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP"]
    eosList = ["BHF_BBB2","KDE0V"]

    methods_BFs = []
    methods_uncerts = []
    for method in methods:
        BFs = []
        uncerts = []
        for eos in eosList:
            bf, bf_trials = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=100)
            #bf = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=0)
            uncert = np.std(bf_trials) * 2
            BFs.append(bf)
            uncerts.append(uncert)
        methods_BFs.append(BFs)
        methods_uncerts.append(uncerts)

    x_axis = np.arange(len(eosList))
    plt.clf()
    plt.rcParams.update({"font.size":20})
    fig = plt.figure(figsize=(15,10))
    plt.bar(x_axis-.25,methods_BFs[0],.2,yerr=methods_uncerts[0],label=labels[0],color="#b2df8a")
    plt.bar(x_axis,methods_BFs[1],.2,yerr=methods_uncerts[1],label=labels[1],color="#a6cee3")
    plt.bar(x_axis+.25,methods_BFs[2],.2,yerr=methods_uncerts[2],label=labels[2],color="#1f78b4")
    #plt.bar(x_axis-.25,methods_BFs[0],.2,label=labels[0],color="#b2df8a")
    #plt.bar(x_axis,methods_BFs[1],.2,label=labels[1],color="#a6cee3")
    #plt.bar(x_axis+.25,methods_BFs[2],.2,label=labels[2],color="#1f78b4")

    plt.yscale('log')
    plt.xticks(x_axis,eosList)
    plt.ylim(1.0e-4,(max(methods_BFs[0]+methods_BFs[1]+methods_BFs[2])+max(methods_uncerts[0]+methods_uncerts[1]+methods_uncerts[2]))*10.)
    plt.axhline(1.)
    plt.legend()
    plt.title("EoS Bayes Factors w.r.t. SLY")
    plt.ylabel("Bayes Factor")
    plt.savefig("EM_GW_BFs.png", bbox_inches='tight')

