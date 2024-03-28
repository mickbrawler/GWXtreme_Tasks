import numpy as np
from scipy.optimize import fsolve
import json
import sys
from astropy.cosmology import z_at_value, Planck18
from astropy import units

# Need to use simulations env since its python=3.9 and therefore has astropy=6.0
# anarya_test env is python=3.6 to be compatible with ray, but then has old vers of astropy

def simplifySimulation():

    uLTs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_LTs/phenom-injections/TaylorF2"
    uLs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_Ls/IMRPhenomPv2_NRTidal/APR4_EPP"
    phenomPhenom_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Phenom/IMRPhenomPv2_NRTidal/APR4_EPP"

#    priors = [uLTs_Dir,uLs_Dir]
    priors = [phenomPhenom_Dir]

    injections = ["282_1.58_1.37", "202_1.35_1.14", "179_1.35_1.23", "71_1.37_1.33", "122_1.77_1.19",
                  "241_1.31_1.28", "220_1.36_1.24", "282_1.35_1.32", "149_1.35_1.23", "237_1.36_1.26",
                  "138_1.5_1.21", "235_1.4_1.3", "219_1.3_1.28", "260_1.48_1.33", "164_1.34_1.19",
                  "55_1.38_1.33", "78_1.35_1.32"]

    filenameEnd = "bns_example_result.json"
    #lambdasLabels = [["lambda_tilde","delta_lambda"],["lambda_1","lambda_2"]]
    lambdasLabels = [["lambda_1","lambda_2"]]
    outputEnd = "bns_example_result_simplified.json"

    for prior,labels in zip(priors,lambdasLabels):
        label_a, label_b = labels

        for injection in injections:

            D = injection.split('_')[0]
            print(D)
            z=z_at_value(Planck18.luminosity_distance,float(D)*units.Mpc).value

            # This first try except is due to my directory separation for "troublesome" injections
            try:
                with open("{}/{}/{}".format(prior,injection,filenameEnd),"r") as f:
                    data = json.load(f)['posterior']['content']
            except FileNotFoundError:
                with open("{}/troublesome/{}/{}".format(prior,injection,filenameEnd),"r") as f:
                    data = json.load(f)['posterior']['content']

            # This second try except is for when the file supplied has no m1 and m2; converter is then used
            try: m1, m2, q, mc, lambda_A, lambda_B = data['m1_source'],data['m2_source'],data['mass_ratio'],data['chirp_mass'],data[label_a],data[label_b]
            except KeyError: 
                q, mc, lambda_A, lambda_B = data['mass_ratio'],data['chirp_mass'],data[label_a],data[label_b]
                m1, m2 = MassesInversion(q,mc).solve_system()

            Dict = {'posterior':{'content':{'mass_1_source':m1,'mass_2_source':m2,'mass_ratio':q,'chirp_mass_source':mc,label_a:lambda_A,label_b:lambda_B}}}

            # This third try except is due to same reason as first try except
            try:
                with open("{}/{}/{}".format(prior,injection,outputEnd),"w") as f:
                    json.dump(Dict,f,indent=2,sort_keys=True)
            except FileNotFoundError:
                with open("{}/troublesome/{}/{}".format(prior,injection,outputEnd),"w") as f:
                    json.dump(Dict,f,indent=2,sort_keys=True)



def simplifyRealEvent():
    # Seems I did this in an ipython environment and then followed it for the 
    # simulation simplifying.

    uLTs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/outdir/real/uniformP_LTs/GW170817/data.json"
    uLTs_Dir = "/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/outdir/real/uniformP_LTs/GW170817/"
    uLTs_labels = ["lambda_tilde","delta_lambda_tilde"]

    uLs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year3/GW170817_prior_L1L2/CIT_attempt_successful/outdir/GW170817_result.json"
    uLs_Dir = "/home/michael/projects/eos/GWXtreme_Tasks/year3/GW170817_prior_L1L2/CIT_attempt_successful/outdir/"
    uLs_labels = ["lambda_1","lambda_2"]


    Files = [uLTs_File,uLs_File]
    Dirs = [uLTs_Dir,uLs_Dir]
    labels = [uLTs_labels,uLs_labels]
    for index in range(len(Files)):
        print(labels[index])
        label_a, label_b = labels[index]

        with open(Files[index],"r") as f: data = json.load(f)['posterior']['content']

        # In a perfect world this would first try would always work
        try: m1, m2, q, mc, lambda_A, lambda_B = data['m1_source'],data['m2_source'],data['mass_ratio'],data['chirp_mass_source'],data[label_a],data[label_b]

        except KeyError: 
            # This has to work for GW170817 (LT,dLT)
            # Case: It has source masses and the lambdas atleast
            try:
                m1, m2, lambda_A, lambda_B = np.array(data['m1_source']),np.array(data['m2_source']),data[label_a],data[label_b]
                q = m2 / m1
                mc = ( (m1*m2)**(3/5) ) / ( (m1+m2)**(1/5) )

            except KeyError:
                # This has to work for GW170817 (L1,L2)
                # Case: It only has detector frame masses
                m1, m2, lambda_A, lambda_B = np.array(data['mass_1']),np.array(data['mass_2']),data[label_a],data[label_b]
                D = 40.4 # Mpc
                z=z_at_value(Planck18.luminosity_distance,float(D)*units.Mpc).value
                m1 *= (1+z)
                m2 *= (1+z)
                q = m2 / m1
                mc = ( (m1*m2)**(3/5) ) / ( (m1+m2)**(1/5) )


        if label_a == "lambda_tilde":
            label_a = "lambdat"
            label_b = "dlambdat"
        Dict = {'posterior':{'content':{'m1_source':m1.tolist(),'m2_source':m2.tolist(),'q':q.tolist(),'mc_source':mc.tolist(),label_a:lambda_A,label_b:lambda_B}}}

        with open(Dirs[index]+"/simplified_result.json","w") as f:
            json.dump(Dict,f,indent=2,sort_keys=True)



class MassesInversion:

    def __init__(self, q, mc):

        self.q = q
        self.mc = mc

    def MassSolving(self, Ms):
        # Expression providing function for fsolve call

        Mass1 = Ms[0]
        Mass2 = Ms[1]
        expressions = np.zeros(2)
        expressions[0] = (Mass2/Mass1) - self.q
        expressions[1] = ( ((Mass1*Mass2)**(3/5)) / ((Mass1+Mass2)**(1/5)) ) - self.mc
        return expressions

    def solve_system(self):

        Masses1 = []
        Masses2 = []
        for self.q, self.mc in zip(self.q, self.mc):

            Mass1, Mass2 = fsolve(self.MassSolving,[1.0,1.0])
            Masses1.append(Mass1)
            Masses2.append(Mass2)

        return(Masses1,Masses2)
