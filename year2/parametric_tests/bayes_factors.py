from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import json
import pylab as pl

def get_EoS_error(og_file, piecewise_file, spectral_file, posterior_file):
    
    with open(og_file, "r") as f:
        og_bestfit = json.load(f)
    with open(piecewise_file, "r") as f:
        piecewise_bestfit = json.load(f)
    with open(spectral_file, "r") as f:
        spectral_bestfit = json.load(f)
        
    BF_dict = {}
    error_BF_dict = {}
    #EoS_names = ["APR4_EPP", "MS1", "H4"]
    EoS_names = ["APR4_EPP"]
    for EoS_name in EoS_names:
        modsel = ems.Model_selection(posteriorFile=posterior_file, spectral=False)
        named_BF = modsel.computeEvidenceRatio(EoS1=EoS_name, EoS2='SLY')
        og_BF = modsel.computeEvidenceRatio(EoS1=og_bestfit[EoS_name][0:4], EoS2='SLY')
        og_error = (abs(named_BF - og_BF) / named_BF) * 100
        piecewise_BF = modsel.computeEvidenceRatio(EoS1=piecewise_bestfit[EoS_name], EoS2='SLY')
        piecewise_error = (abs(named_BF - piecewise_BF) / named_BF) * 100
        modsel = ems.Model_selection(posteriorFile=posterior_file, spectral=True)
        spectral_BF = modsel.computeEvidenceRatio(EoS1=spectral_bestfit[EoS_name], EoS2='SLY')
        spectral_error = (abs(named_BF - spectral_BF) / named_BF) * 100
        BF_dict.update({EoS_name:{"named":named_BF, "og":og_BF, "piecewise":piecewise_BF, "spectral":spectral_BF}})
        error_BF_dict.update({EoS_name:{"og":og_error, "piecewise":piecewise_error, "spectral":spectral_error}})

    with open("files/parametrized_BF/bayes_factors.json", "w") as f:
        json.dump(BF_dict, f, indent=2, sort_keys=True)

    with open("files/parametrized_BF/error_bayes_factors.json", "w") as f:
        json.dump(error_BF_dict, f, indent=2, sort_keys=True)
