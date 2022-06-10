import numpy as np
import json
import corner
import pandas as pd
import pylab as pl
from GWXtreme import eos_prior as ep

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
            except FileNotFoundError: 
                print("interrupted core")
                continue # Error popped up when ./core50 didn't have anything meaning it must have gotten interrupted...
            if filetype == "seg_faults.txt": seg_fault_samples += samples
            elif filetype == "no_errors.txt": no_error_samples += samples
            elif filetype == "errors.txt": error_samples += samples

    with open("files/combined/seg_fault_samples.json", "w") as f: json.dump(seg_fault_samples, f, indent=2)
    with open("files/combined/no_errors_samples.json", "w") as f: json.dump(no_error_samples, f, indent=2)
    with open("files/combined/errors_samples.json", "w") as f: json.dump(error_samples, f, indent=2)

# Two options are either include errors or don't

def running_isvalid(include_errors=False):
    # Runs isvalid on samples from multiple files and save the ones that were true

    if include_errors:
        filenames = ["files/combined/seg_fault_samples.json", "files/combined/no_errors_samples.json", "files/combined/errors_samples.json"]
        label = "with_errors"
    else:
        filenames = ["files/combined/seg_fault_samples.json", "files/combined/no_errors_samples.json"]
        label = "no_errors"

    samples = []
    for filename in filenames:
        with open(filename, "r") as f: samples += json.load(f)

    samples = np.array(samples)
    params = {"gamma1":samples[:,0],"gamma2":samples[:,1],"gamma3":samples[:,2],"gamma4":samples[:,3]}
    priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}
    valid_indices = ep.is_valid_eos(params,priorbounds,spectral=True)
    samples = samples[valid_indices].tolist()
    with open("files/combined/{}_valid_samples.json".format(label), "w") as f: json.dump(samples, f, indent=2)


def parameter_slice_plots(include_errors=False):
    # 6 2d plots of the 4d parameter space.

    pl.clf()

    if include_errors:
        filenames = ["files/combined/seg_fault_samples.json", "files/combined/no_errors_samples.json", "files/combined/errors_samples.json", "files/combined/with_errors_valid_samples.json"]
        Dir = "with_errors/"
        colors = ["red", "blue", "yellow", "black"]
    else:
        filenames = ["files/combined/seg_fault_samples.json", "files/combined/no_errors_samples.json", "files/combined/no_errors_valid_samples.json"]
        Dir = "no_errors/"
        colors = ["red", "blue", "black"]

    increment = 0
    s = .01
    for filename in filenames:

        with open(filename, "r") as f:
            samples = np.array(json.load(f))

        g1_p1 = samples[:,0]
        g2_g1 = samples[:,1]
        g3_g2 = samples[:,2]
        g4_g3 = samples[:,3]
        
        print("1")
        pl.figure(1)
        pl.scatter(g1_p1,g2_g1,s=s,color=colors[increment])
        pl.xlabel("g1_p1")
        pl.ylabel("g2_g1")
        pl.title("g1_p1  g2_g1")

        print("2")
        pl.figure(2)
        pl.scatter(g1_p1,g3_g2,s=s,color=colors[increment])
        pl.xlabel("g1_p1")
        pl.ylabel("g3_g2")
        pl.title("g1_p1  g3_g2")

        print("3")
        pl.figure(3)
        pl.scatter(g1_p1,g4_g3,s=s,color=colors[increment])
        pl.xlabel("g1_p1")
        pl.ylabel("g4_g3")
        pl.title("g1_p1  g4_g3")

        print("4")
        pl.figure(4)
        pl.scatter(g2_g1,g3_g2,s=s,color=colors[increment])
        pl.xlabel("g2_g1")
        pl.ylabel("g3_g2")
        pl.title("g2_g1  g3_g2")

        print("5")
        pl.figure(5)
        pl.scatter(g2_g1,g4_g3,s=s,color=colors[increment])
        pl.xlabel("g2_g1")
        pl.ylabel("g4_g3")
        pl.title("g2_g1  g4_g3")

        print("6")
        pl.figure(6)
        pl.scatter(g3_g2,g4_g3,s=s,color=colors[increment])
        pl.xlabel("g3_g2")
        pl.ylabel("g4_g3")
        pl.title("g3_g2  g4_g3")

        increment += 1

    pl.figure(1)
    pl.savefig("files/plots/{}g1_p1__g2_g1.png".format(Dir))
    
    pl.figure(2)
    pl.savefig("files/plots/{}g1_p1__g3_g2.png".format(Dir))

    pl.figure(3)
    pl.savefig("files/plots/{}g1_p1__g4_g3.png".format(Dir))

    pl.figure(4)
    pl.savefig("files/plots/{}g2_g1__g3_g2.png".format(Dir))

    pl.figure(5)
    pl.savefig("files/plots/{}g2_g1__g4_g3.png".format(Dir))

    pl.figure(6)
    pl.savefig("files/plots/{}g3_g2__g4_g3.png".format(Dir))
