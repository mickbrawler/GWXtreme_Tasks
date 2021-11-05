from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json

p_eos_val = {"AP4":[33.269,2.830,3.445,3.348],
             "MPA1":[33.495,3.446,3.572,2.887],
             "MS1":[33.858,3.224,3.033,1.325]}

def get_data(mkn, trials=1000):
    # Gets all the data we need in an easy to reuse fashion
    # Each eos' : name, lal bayes factor, lal deviation, piecewise polytrope, and
    # piecewise deviation are stored in their own lists

    with open("Analysis/Refined_bestof_1_8_eos_global_values.json","r") as f:
        data = json.load(f)

    lal_bfs = [] # Need bayes factors from lal source in list
    lal_sds = []

    pp_bfs = [] # Need bayes factors from piecewise polytrope values in list
    pp_sds = []

    og_pp_bfs = [] # Need bayes factors from paper
    og_pp_sds = []

    modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

    for eos in data:

        print(eos)

        if eos in p_eos_val:

            p1,g1,g2,g3 = p_eos_val[eos]
            og_pp_bf, og_pp_trials = modsel.computeEvidenceRatio([p1,g1,g2,g3],"SLY",trials=trials)
            og_pp_bfs.append(og_pp_bf)
            og_pp_sds.append(np.std(og_pp_trials) * 2)

        else:

            og_pp_bfs.append(0)
            og_pp_sds.append(0)

        lal_bf, lal_trials = modsel.computeEvidenceRatio(eos,"SLY",trials=trials)
        lal_bfs.append(lal_bf)
        lal_sds.append(np.std(lal_trials) * 2)

        p1,g1,g2,g3,_ = data[eos]
        pp_bf, pp_trials = modsel.computeEvidenceRatio([p1,g1,g2,g3],"SLY",trials=trials)
        pp_bfs.append(pp_bf)
        pp_sds.append(np.std(pp_trials) * 2)

    output = np.vstack((og_pp_bfs,og_pp_sds,lal_bfs,lal_sds,pp_bfs,pp_sds)).T
    np.savetxt("Plots/Casabona_plots/data/bar_plot_data_mk{}.txt".format(mkn),output,fmt="%f\t%f\t%f\t%f\t%f\t%f")

def plotter(mkn):
    # Makes bar plot

    data = np.loadtxt("Plots/Casabona_plots/data/bar_plot_data_mk2.txt")
    
    og_pp_bfs,og_pp_sds,lal_bfs,lal_sds,pp_bfs,pp_sds = data.T

    with open("Analysis/Refined_bestof_1_8_eos_global_values.json","r") as f:
        data = json.load(f)

    eos_list = []
    for eos in data: eos_list.append(eos)

    x_axis = np.arange(len(eos_list))

    pl.rcParams.update({"font.size":18})
    pl.figure(figsize=(20,10))
    pl.bar(x_axis-.30,lal_bfs,.3,yerr=lal_sds,label="LAL Simulation Method")
    pl.bar(x_axis+.00,pp_bfs,.3,yerr=pp_sds,label="Piecewise Polytrope Method")
    pl.bar(x_axis+.30,og_pp_bfs,.3,yerr=og_pp_sds,label="Original Piecewise Polytrope Method")

    pl.xticks(x_axis,eos_list,rotation=45,ha="right")
    pl.ylabel("Bayes-factor w.r.t SLY")
    pl.title("Likelihood Comparison")
    pl.legend()
    pl.savefig("Plots/Casabona_plots/bar_plot_mk{}.png".format(mkn))
