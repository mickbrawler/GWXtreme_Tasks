import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from scipy.optimize import fsolve
from GWXtreme import eos_model_selection as ems
import lambdaInversion

def Dist_plot():
    #M1 distribution, M2 distribution, M1-M2 distribution
    #L1 dist, L2 dist, L1-L2 dist
    #dL~ dist, L~ dist, dL~-L~ dist 

    anaryaFile = "./anarya_bns_example_result.json"
    myFile = "./mine_bns_example_result.json"

    with open(anaryaFile,"r") as f:
        A_data = json.load(f)["posterior"]["content"]

    with open(myFile,"r") as f:
        M_data = json.load(f)["posterior"]["content"]

    # UNIFORM (DLT, LT) Prior
    A_mc = A_data["chirp_mass"]
    A_q = A_data["mass_ratio"]
    # Anarya's files don't have Mass1
    # Anarya's files don't have Mass2
    Amass1, Amass2 = ems.getMasses(np.array(A_q),np.array(A_mc))
    A_sq = (Amass1*Amass2)/((Amass1+Amass2)**2)
    A_dLT = A_data["delta_lambda"]
    A_LT = A_data["lambda_tilde"]
    invert = lambdaInversion.TildeInvertLambda(A_sq,A_dLT,A_LT)
    A_L1, A_L2 = invert.solve_system()
    # Anarya's files don't have L1
    # Anarya's files don't have L2

    # UNIFORM (L1, L2) Prior
    M_mc = M_data["chirp_mass"]
    M_q = M_data["mass_ratio"]
    #Mmass1 = M_data["mass_1"]
    #Mmass2 = M_data["mass_2"]
    Mmass1, Mmass2 = ems.getMasses(np.array(M_q),np.array(M_mc))
    M_sq = (Mmass1*Mmass2)/((Mmass1+Mmass2)**2)
    M_dLT = M_data["delta_lambda_tilde"]
    M_LT = M_data["lambda_tilde"]
    invert = lambdaInversion.TildeInvertLambda(M_sq,M_dLT,M_LT)
    M_L1, M_L2 = invert.solve_system()
    #M_L1 = M_data["lambda_1"]
    #M_L2 = M_data["lambda_2"]

    Avars1 = [Amass1, A_dLT, A_L1]
    Avars2 = [Amass2, A_LT, A_L2]
    Mvars1 = [Mmass1, M_dLT, M_L1]
    Mvars2 = [Mmass2, M_LT, M_L2]
    labels1 = ["mass1", "delta_lambda", "lambda1"]
    labels2 = ["mass2", "lambda_tilde", "lambda2"]
    legend1, legend2 = "Uniform (dL~,L~) Prior", "Uniform (L1,L2) Prior"
    Acolor, Mcolor = "red", "blue"
    for Avar1,Avar2,Mvar1,Mvar2,label1,label2 in zip(Avars1,Avars2,Mvars1,Mvars2,labels1,labels2):

        plt.clf()
        sns.kdeplot(data=Avar1,color=Acolor)
        sns.kdeplot(data=Mvar1,color=Mcolor)
        plt.legend(labels=[legend1,legend2])
        plt.title(label1)
        plt.savefig("plots/{}.png".format(label1))

        plt.clf()
        sns.kdeplot(data=Avar2,color=Acolor)
        sns.kdeplot(data=Mvar2,color=Mcolor)
        plt.legend(labels=[legend1,legend2])
        plt.title(label2)
        plt.savefig("plots/{}.png".format(label2))

        plt.clf()
        plt.rcParams['lines.linewidth'] = 1
        AAvar1, AAvar2 = np.mgrid[min(Avar1):max(Avar1):100j, min(Avar2):max(Avar2):100j]
        positions = np.vstack([AAvar1.ravel(), AAvar2.ravel()])
        values = np.vstack([Avar1, Avar2])
        kernel = st.gaussian_kde(values)
        A_f = np.reshape(kernel(positions).T, AAvar1.shape)

        MMvar1, MMvar2 = np.mgrid[min(Mvar1):max(Mvar1):100j, min(Mvar2):max(Mvar2):100j]
        positions = np.vstack([MMvar1.ravel(), MMvar2.ravel()])
        values = np.vstack([Mvar1, Mvar2])
        kernel = st.gaussian_kde(values)
        M_f = np.reshape(kernel(positions).T, MMvar1.shape)

        A = plt.contour(AAvar1, AAvar2, A_f, colors="red", alpha=.25)
        M = plt.contour(MMvar1, MMvar2, M_f, colors="blue", alpha=.25)
        h1,l1 = A.legend_elements()
        h2,l1 = M.legend_elements()
        plt.legend([h1[0], h2[0]], [legend1,legend2])
        plt.title("{}, {}".format(label1,label2))
        plt.savefig("plots/{}_{}.png".format(label1,label2))

def BF_barplot(inverseWeight=False):
    #Bayes Factor comparison Task:
    #Plot bars of BFS: 
    #Uniform (dL~, L~) produced (L~, dL~), Uniform (L1, L2) produced (dL~, L~)

    anaryaFile = "./anarya_bns_example_result.json"
    myFile = "./mine_bns_example_result.json"
    Amodsel = ems.Model_selection(anaryaFile,UpriorLTs=True,Ns=4000)
    Mmodsel = ems.Model_selection(myFile,UpriorLTs=inverseWeight,Ns=4000)

    labels = ["Uniform (dL~,L~) Prior", "Uniform (L1,L2) Prior"]

    methods = [Amodsel,Mmodsel]
    eosList = ["BHF_BBB2","KDE0V","SKOP","H4","HQC18","SKMP","APR4_EPP","MPA1","MS1_PP","MS1B_PP"]
    #eosList = ["BHF_BBB2","KDE0V","SKOP"]
    methods_BFs = []
    #methods_uncerts = []
    for method in methods:
        BFs = []
        #uncerts = []
        for eos in eosList:
            #bf, bf_trials = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=10)
            bf = method.computeEvidenceRatio(EoS1=eos,EoS2="SLY",trials=0)
            #uncert = np.std(bf_trials) * 2
            BFs.append(bf)
            #uncerts.append(uncert)
        methods_BFs.append(BFs)
        #methods_uncerts.append(uncerts)

    x_axis = np.arange(len(eosList))
    #plt.bar(x_axis-.25,methods_BFs[0],.5,yerr=methods_uncerts[0],label=labels[0],color="red")
    #plt.bar(x_axis+.25,mehtods_BFs[1],.5,yerr=methods_uncerts[1],label=labels[1],color="blue")
    plt.bar(x_axis-.25,methods_BFs[0],.4,label=labels[0],color="red")
    plt.bar(x_axis+.25,methods_BFs[1],.4,label=labels[1],color="blue")

    plt.xticks(x_axis,eosList,rotation=45,ha="right")
    plt.legend()
    plt.title("EoS Bayes Factors w.r.t. SLY")
    plt.xlabel("EoSs")
    plt.ylabel("Bayes Factor")
    #plt.savefig("plots/barplot_inverseWeight_difPriors_BFs.png")
    if inverseWeight == True: saveLabel = "_inverseWeight_"
    else: saveLabel = "_"
    plt.savefig("plots/barplot{}difPriors_BFs.png".format(saveLabel))

