from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import json
import pylab as pl

def get_dict_eos_BF(posterior_file, outfile):
    # Produce dictionary of each eos' bayes factor.
    # posterior_file    : Posterior file name string 
    # outfile   : Name of output file

    modsel = ems.Model_selection(posteriorFile=posterior_file, spectral=False)
    eos_names = lalsim.SimNeutronStarEOSNames
    eos_BFs={}
    for eos_name in eos_names:
        BF = modsel.computeEvidenceRatio(EoS1=eos_name, EoS2='SLY')
        eos_BFs.update({eos_name:BF})

    with open(outfile, "w") as f:
        json.dump(eos_BFs, f, indent=2, sort_keys=True)

def get_dict_eos_stack_BF(posterior_files, outfile):
    # Produce dictionary of each eos' stacked bayes factor.
    # posterior_files   : List of posterior file name strings
    # outfile   : Name of output file
    
    stackobj = ems.Stacking(posterior_files, spectral=False)
    eos_names = lalsim.SimNeutronStarEOSNames
    eos_stack_BFs={}
    for eos_name in eos_names:
        stack_BF = stackobj.stack_events(eos_name, 'SLY')
        eos_stack_BFs.update({eos_name:stack_BF})

    with open(outfile, "w") as f:
        json.dump(eos_stack_BFs, f, indent=2, sort_keys=True)

def get_perc_error(GWXtreme_file, Anarya_file, outfile):
    # Produce dictionary of each eos' percect error (between GWXtreme & Anarya's build)
    # GWXtreme_file : File with bayes factors produced by default GWXtreme
    # Anarya_file   : File with bayes factors produced by Anarya's build
    # outfile   : Name of output file

    eos_names = lalsim.SimNeutronStarEOSNames

    with open(GWXtreme_file,"r") as f:
        default_dict = json.load(f)

    with open(Anarya_file,"r") as f:
        anarya_dict = json.load(f)

    default_vals = np.array(list(default_dict.values()))
    anarya_vals = np.array(list(anarya_dict.values()))
    perc_error = (np.abs(default_vals - anarya_vals) / default_vals) * 100

    pl.clf()
    pl.figure(figsize=(25, 10))
    pl.bar(eos_names, perc_error)
    pl.title("Percent Error of Anarya's Build")
    pl.xlabel("EoS Names")
    pl.ylabel("Percent Error")
    pl.xticks(rotation=45, ha='right', fontsize=5)
    pl.tight_layout()
    pl.savefig(outfile)

