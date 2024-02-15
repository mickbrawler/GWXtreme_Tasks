from GWXtreme import eos_model_selection as ems
import numpy as np
import matplotlib.pyplot as plt
import json
import convertlambdas

# Create 3D KDE BF comparison plot using uniform (L1,L2) prior GW170817 PE 
# results from Taylor and Phenom waveforms (latter linked by Anarya).

def convert_Taylor_dat_to_json(dat_file):
    # Convert Taylor (uniformPLs) .dat PE files on GW170817 to jsons. 
    # Particularly the uniformP(dLT,LT).

    _data = np.recfromtxt(dat_file, names=True)
    (m1,m2,lambdaT,dlambdaT,luminosity_distance)=(np.array(_data['m1_source']),
                                            np.array(_data['m2_source']),
                                            np.array(_data['lambdat']),
                                            np.array(_data['dlambdat']),
                                            np.array(_data['distance']))
    q = m2 / m1
    mc = ((m1*m2)**(3/5)) / ((m1+m2)**(1/5))
    mr = (m1*m2)/((m1+m2)**2) # symmetric mass ratio

    convert = convertlambdas.LambdasInversion(mr,lambdaT,dlambdaT)
    lambda1, lambda2 = convert.solve_system()

    data={'mass_1_source':list(m1),'mass_2_source':list(m2),'mass_ratio':list(q),'chirp_mass':list(mc),'lambda_1':list(lambda1),'lambda_2':list(lambda2),'lambda_tilde':list(lambdaT),'delta_lambda_tilde':list(dlambdaT),'luminosity_distance':list(luminosity_distance)}
    total = {"posterior":{"content":data}}

    with open(dat_file[0:-3]+"json","w") as f:
        json.dump(total,f,indent=2,sort_keys=True)


def convert_Phenom_dat_to_json(dat_file):
    # Convert Phenom .dat PE files on GW170817 to jsons.

    _data = np.recfromtxt(dat_file, names=True)
    (m1,m2,lambda1,lambda2,luminosity_distance)=(np.array(_data['m1_detector_frame_Msun']),
                                            np.array(_data['m2_detector_frame_Msun']),
                                            np.array(_data['lambda1']),
                                            np.array(_data['lambda2']),
                                            np.array(_data['luminosity_distance_Mpc']))
    q = m2 / m1
    mc = ((m1*m2)**(3/5)) / ((m1+m2)**(1/5))
    mr = (m1*m2)/((m1+m2)**2) # symmetric mass ratio

    lambdaT = (8/13)*((1+7*mr-31*mr**2)*(lambda1+lambda2)+((1-4*mr)**.5)*(1+9*mr-11*mr**2)*(lambda1-lambda2))
    dlambdaT = .5*(((1-4*mr)**.5)*(1-(13272/1319)*mr+(8944/1319)*mr**2)*(lambda1+lambda2)+(1-(15910/1319)*mr+(32850/1319)*mr**2+(3380/1319)*mr**3)*(lambda1-lambda2))

    data={'mass_ratio':list(q),'chirp_mass':list(mc),'lambda_1':list(lambda1),'lambda_2':list(lambda2),'lambda_tilde':list(lambdaT),'delta_lambda_tilde':list(dlambdaT),'luminosity_distance':list(luminosity_distance)}
    total = {"posterior":{"content":data}}

    with open(dat_file[0:-3]+"json","w") as f:
        json.dump(total,f,indent=2,sort_keys=True)


def plot():
    # Makes barplot of BFs for GW170817 with Taylor and Phenom waveform.

    Taylor_LTs_result = "Files/GW170817_LTs_result.json" # LTs prior Taylor PE run in json form
    Taylor_Ls_result = "Files/GW170817_Ls_result.json"
    high_Phenom_result = "Files/high_spin_PhenomPNRT_posterior_samples.json"
    low_Phenom_result = "Files/low_spin_PhenomPNRT_posterior_samples.json"

    modsel_Taylor_LTs = ems.Model_selection(Taylor_LTs_result,Ns=4000,kdedim=2)
    modsel_Taylor_Ls = ems.Model_selection(Taylor_Ls_result,Ns=4000,kdedim=3)
    modsel_high_Phenom = ems.Model_selection(high_Phenom_result,Ns=4000,kdedim=3)
    modsel_low_Phenom = ems.Model_selection(low_Phenom_result,Ns=4000,kdedim=3)

    labels = ["TaylorLs", "TaylorLTs", "high-spin-Phenom","low-spin-Phenom"]
    colors = ["#1b9e77","#d95f02","#7570b3","#e7298a"]
    modsels = [modsel_Taylor_LTs,modsel_Taylor_Ls,modsel_high_Phenom,modsel_low_Phenom]
    eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP","MPA1","MS1_PP","MS1B_PP"]
    modsels_BFs = []
    modsels_uncerts = []
    for modsel in modsels:
        print(modsel)
        BFs = []
        uncerts = []
        for eos in eosList:
            print(eos)
            bf, bf_trials = modsel.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=100)
            #bf = modsel.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=0)
            uncert = np.std(bf_trials) * 2
            BFs.append(bf)
            uncerts.append(uncert)
        modsels_BFs.append(BFs)
        modsels_uncerts.append(uncerts)

    x_axis = np.arange(len(eosList))
    spacing = [-.3,-.1,.1,.3]
    plt.clf()
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(15, 10))
    for index in range(len(modsels)):
        plt.bar(x_axis+spacing[index],modsels_BFs[index],.175,yerr=modsels_uncerts[index],label=labels[index],color=colors[index])
        #plt.bar(x_axis+spacing[index],modsels_BFs[index],.15,label=labels[index],color=colors[index])

    plt.xticks(x_axis,eosList,rotation=45,ha="right")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("EoSs")
    plt.ylabel("Bayes Factor")
    plt.title("EoS Bayes Factors w.r.t. SLY")
    plt.savefig("outdir/BFs/3D_BF_Taylor_Phenom.png")

    Dictionary = {labels[Index]:{eosList[eIndex]:[modsels_BFs[Index][eIndex],modsels_uncerts[Index][eIndex]] for eIndex in range(len(eosList))} for Index in range(len(labels))}
    #Dictionary = {labels[Index]:{eosList[eIndex]:modsels_BFs[Index][eIndex] for eIndex in range(len(eosList))} for Index in range(len(labels))}
    with open("outdir/BFs/3D_BF_Taylor_Phenom.json","w") as f:
        json.dump(Dictionary, f, indent=2, sort_keys=True)

