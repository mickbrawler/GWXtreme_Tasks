import numpy as np
import json
import corner
import pandas as pd
import pylab as pl

def combine(cores, new=False):
    # cores ::  Number of cores used in latest run
    # new   ::  True-(no files present) : False-(files already present)

    filetypes = ["seg_faults.txt","no_errors.txt","errors.txt"]
    if new:
        seg_fault_samples = []
        no_error_samples = []
        error_samples = []
    else:
        with open("files/combined/seg_fault_samples.json", "r") as f: seg_fault_samples = json.load(f)
        with open("files/combined/no_errors_samples.json", "r") as f: no_error_samples = json.load(f)
        with open("files/combined/errors_samples.json", "r") as f: error_samples = json.load(f)

    for core in range(1,cores+1):
        for filetype in filetypes:
            File = "./files/runs/core{}/{}".format(core,filetype)
            try: samples = np.loadtxt(File).tolist()
            except FileNotFoundError: continue # Error popped up when ./core50 didn't have anything meaning it must have gotten interrupted...
            if filetype == "seg_faults.txt": seg_fault_samples += samples
            elif filetype == "no_errors.txt": no_error_samples += samples
            elif filetype == "errors.txt": error_samples += samples

    with open("files/combined/seg_fault_samples.json", "w") as f: json.dump(seg_fault_samples, f, indent=2)
    with open("files/combined/no_errors_samples.json", "w") as f: json.dump(no_error_samples, f, indent=2)
    with open("files/combined/errors_samples.json", "w") as f: json.dump(error_samples, f, indent=2)

def parameter_slice_plots():
    # 6 2d plots of the 4d parameter space.

    pl.clf()
    pl.rcParams['figure.figsize'] = [10, 7]
    filenames = ["files/combined/seg_fault_samples.json", "files/combined/no_errors_samples.json", "files/combined/errors_samples.json"]
    colors = ["red", "blue", "yellow"]

    increment = 0
    s = 1
    for filename in filenames:

        with open(filename, "r") as f:
            samples = np.array(json.load(f))

        g1_p1 = samples[:,0]
        g2_g1 = samples[:,1]
        g3_g2 = samples[:,2]
        g4_g3 = samples[:,3]
        
        pl.clf()
        pl.scatter(g1_p1,g2_g1,s=s)
        pl.xlabel("g1_p1")
        pl.ylabel("g2_g1")
        pl.title("g1_p1  g2_g1")
        pl.title("")
        pl.savefig("files/combined/plots/g1_p1__g2_g1.png")

        pl.clf()
        pl.scatter(g1_p1,g3_g2,s=s)
        pl.xlabel("g1_p1")
        pl.ylabel("g3_g2")
        pl.title("g1_p1  g3_g2")
        pl.savefig("files/combined/plots/g1_p1__g3_g2.png")

        pl.clf()
        pl.scatter(g1_p1,g4_g3,s=s)
        pl.xlabel("g1_p1")
        pl.ylabel("g4_g3")
        pl.title("g1_p1  g4_g3")
        pl.savefig("files/combined/plots/g1_p1__g4_g3.png")

        pl.clf()
        pl.scatter(g2_g1,g3_g2,s=s)
        pl.xlabel("g2_g1")
        pl.ylabel("g3_g2")
        pl.title("g2_g1  g3_g2")
        pl.savefig("files/combined/plots/g2_g1__g3_g2.png")

        pl.clf()
        pl.scatter(g2_g1,g4_g3,s=s)
        pl.xlabel("g2_g1")
        pl.ylabel("g4_g3")
        pl.title("g2_g1  g4_g3")
        pl.savefig("files/combined/plots/g2_g1__g4_g3.png")

        pl.clf()
        pl.scatter(g3_g2,g4_g3,s=s)
        pl.xlabel("g3_g2")
        pl.ylabel("g4_g3")
        pl.title("g3_g2  g4_g3")
        pl.savefig("files/combined/plots/g3_g2__g4_g3.png")

        increment += 1

