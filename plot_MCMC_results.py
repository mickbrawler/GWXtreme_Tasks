import numpy as np
import pylab as pl
import seaborn as sns
import json

def plotter(filename,p0file,g1file,g2file,g3file,eos_ind):
    #file names must include directory and type.

    with open(filename,"r") as f:
        data = json.load(f)

    sns.kdeplot(data["p0"][eos_ind])
    pl.axvline(x=data["p0"][eos_ind][np.argmax(data["r2"][eos_ind])])
    pl.savefig(p0file)

    pl.clf()
    sns.kdeplot(data["g1"][eos_ind])
    pl.axvline(x=data["g1"][eos_ind][np.argmax(data["r2"][eos_ind])])
    pl.savefig(g1file)

    pl.clf()
    sns.kdeplot(data["g2"][eos_ind])
    pl.axvline(x=data["g2"][eos_ind][np.argmax(data["r2"][eos_ind])])
    pl.savefig(g2file)

    pl.clf()
    sns.kdeplot(data["g3"][eos_ind])
    pl.axvline(x=data["g3"][eos_ind][np.argmax(data["r2"][eos_ind])])
    pl.savefig(g3file)
