from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import json
import pylab as pl

def get_eos_BF():
    # Produces file of a dictionary of each EoS' Bayes factor.
    # Different Bayes factor for each version of an EoS (named, MRK, ML,
    # piecewise, spectral)

    env = "base"
    ref_EoS = "SLY"
    EoS_list = ["APR4_EPP", "H4", "MS1"]
    posterior_files = ["posterior_samples/posterior_samples_narrow_spin_prior.dat",
                       "posterior_samples/posterior_samples_broad_spin_prior.dat"]

    # Parametrized bestfits
    with open("../parametric_tests/files/basic_runs/1_piecewise_EoS_bestfits.json", "r") as f:
        piecewise_EoS = json.load(f)
    with open("../parametric_tests/files/basic_runs/1_spectral_EoS_bestfits.json", "r") as f:
        spectral_EoS = json.load(f)

    increment = 0
    Type = ["narrow", "broad"]
    for posterior_file in posterior_files:

        modsel = ems.Model_selection(posteriorFile=posterior_file, spectral=False)
        s_modsel = ems.Model_selection(posteriorFile=posterior_file, spectral=True)
        
        named_BFs = {}
        MRK_BFs = {}
        ML_BFs = {}
        piecewise_BFs = {}
        spectral_BFs = {}
        for EoS in EoS_list:

            BF = modsel.computeEvidenceRatio(EoS1=EoS, EoS2=ref_EoS)
            named_BFs.update({EoS:BF})
            BF = modsel.computeEvidenceRatio(EoS1="comparison_files/MRK/{}.txt".format(EoS), EoS2=ref_EoS)
            MRK_BFs.update({EoS:BF})
            BF = modsel.computeEvidenceRatio(EoS1="comparison_files/ML/{}.txt".format(EoS), EoS2=ref_EoS)
            ML_BFs.update({EoS:BF})

            print(piecewise_EoS[EoS])
            BF = modsel.computeEvidenceRatio(EoS1=piecewise_EoS[EoS], EoS2=ref_EoS)
            piecewise_BFs.update({EoS:BF})
            print(spectral_EoS[EoS])
            BF = s_modsel.computeEvidenceRatio(EoS1=spectral_EoS[EoS], EoS2=ref_EoS)
            spectral_BFs.update({EoS:BF})

