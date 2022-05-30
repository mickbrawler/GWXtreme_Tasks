from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import json

# Prepare json with all values needed for table we can send to Jocelyn Read

#           Narrow                  Broad
# EOS Tabulated MCMC Paper  Tabulated MCMC Paper

p_eos_val = {"AP4":[33.269,2.830,3.445,3.348],
             "MPA1":[33.495,3.446,3.572,2.887],
             "MS1":[33.858,3.224,3.033,1.325]}

modsel1 = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

modsel2 = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_broad_spin_prior.dat")

with open("Analysis/Refined_bestof_1_8_eos_global_values.json","r") as f:
    data = json.load(f)

table_values = {}
for eos in data:
    
    bfs_holder = [] # hold bf for each eos

    n_lal_bf = modsel1.computeEvidenceRatio(eos,"SLY")
    b_lal_bf = modsel2.computeEvidenceRatio(eos,"SLY")

    p1,g1,g2,g3,_ = data[eos]
    n_mc_bf = modsel1.computeEvidenceRatio([p1,g1,g2,g3],"SLY")
    b_mc_bf = modsel2.computeEvidenceRatio([p1,g1,g2,g3],"SLY")

    if eos in p_eos_val:

        p1,g1,g2,g3 = p_eos_val[eos]
        n_p_bf = modsel1.computeEvidenceRatio([p1,g1,g2,g3],"SLY")
        b_p_bf = modsel2.computeEvidenceRatio([p1,g1,g2,g3],"SLY")
        
    else:

        n_p_bf = "nah"
        b_p_bf = "nah"

    table_values.update({eos:[n_lal_bf,n_mc_bf,n_p_bf,b_lal_bf,b_mc_bf,b_p_bf]})

with open("Plots/Casabona_plots/data/table_values_1.json","w") as f:
    json.dump(table_values, f, indent=2, sort_keys=True)

