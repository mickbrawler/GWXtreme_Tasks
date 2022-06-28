import numpy as np
import json
import corner
import pandas as pd
import pylab as pl
from GWXtreme import eos_prior as ep

def combine(cores, new=False, spectral=True):
    # cores ::  Number of cores used in latest run
    # new   ::  True-(no files present) : False-(files already present)

    if spectral: Dir = "spectral/"
    else: Dir = "piecewise/"

    filetypes = ["seg_faults.txt","no_errors.txt","runtime_errors.txt","value_errors.txt"]
    if new:
        seg_fault_samples = []
        no_error_samples = []
        runtime_error_samples = []
        value_error_samples = []
    else:
        with open("files/combined/{}seg_fault_samples.json".format(Dir), "r") as f: seg_fault_samples = json.load(f)
        with open("files/combined/{}no_error_samples.json".format(Dir), "r") as f: no_error_samples = json.load(f)
        with open("files/combined/{}runtime_error_samples.json".format(Dir), "r") as f: runtime_error_samples = json.load(f)
        with open("files/combined/{}value_error_samples.json".format(Dir), "r") as f: value_error_samples = json.load(f)

    for core in range(1,cores+1):
        for filetype in filetypes:
            File = "./files/runs/core{}/{}".format(core,filetype)
            try: samples = np.loadtxt(File).tolist()
            except FileNotFoundError: 
                print("interrupted core")
                continue # Cores may be accidently interrupted causing a lack of files
            if filetype == "seg_faults.txt": seg_fault_samples += samples
            elif filetype == "no_errors.txt": no_error_samples += samples
            elif filetype == "runtime_errors.txt": runtime_error_samples += samples
            elif filetype == "value_errors.txt": value_error_samples += samples

    with open("files/combined/{}seg_fault_samples.json".format(Dir), "w") as f: json.dump(seg_fault_samples, f, indent=2)
    with open("files/combined/{}no_error_samples.json".format(Dir), "w") as f: json.dump(no_error_samples, f, indent=2)
    with open("files/combined/{}runtime_error_samples.json".format(Dir), "w") as f: json.dump(runtime_error_samples, f, indent=2)
    with open("files/combined/{}value_error_samples.json".format(Dir), "w") as f: json.dump(value_error_samples, f, indent=2)

# Two options are either include runtime & value errors or don't
# BOTTOM FUNCTIONS STILL NEED DOUBLE PARAMETRIZATION FUNCTIONALITY

def running_isvalid(include_seg_faults=False, include_errors=False, spectral=True):
    # Runs isvalid on samples from multiple files and save the ones that were true

    if spectral: Dir = "spectral/"
    else: Dir = "piecewise/"

    if (include_seg_faults == True) & (include_errors == True):
        filenames = ["files/combined/{}seg_fault_samples.json".format(Dir), "files/combined/{}no_error_samples.json".format(Dir), "files/combined/{}runtime_error_samples.json".format(Dir), "files/combined/{}value_error_samples.json".format(Dir)]
        label = "seg_error"
    elif (include_seg_faults == True) & (include_errors == False):
        filenames = ["files/combined/{}seg_fault_samples.json".format(Dir), "files/combined/{}no_error_samples.json".format(Dir)]
        label = "seg"
    elif (include_seg_faults == False) & (include_errors == False):
        filenames = ["files/combined/{}no_error_samples.json".format(Dir)]
        label = "no_error"

    samples = []
    for filename in filenames:
        with open(filename, "r") as f: samples += json.load(f)

    samples = np.array(samples)
    params = {"gamma1":samples[:,0],"gamma2":samples[:,1],"gamma3":samples[:,2],"gamma4":samples[:,3]}
    if spectral:
        #priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}
        priorbounds = {'gamma1':{'params':{"min":0.0,"max":2.5}},'gamma2':{'params':{"min":-2.0,"max":2.0}},'gamma3':{'params':{"min":-1.0,"max":1.0}},'gamma4':{'params':{"min":-0.1,"max":0.1}}}
    else:
        #priorbounds = {'logP':{'params':{"min":33.6,"max":34.5}},'gamma1':{'params':{"min":2.0,"max":4.5}},'gamma2':{'params':{"min":1.1,"max":4.5}},'gamma3':{'params':{"min":1.1,"max":4.5}}}
        priorbounds = {'logP':{'params':{"min":33.6,"max":34.5}},'gamma1':{'params':{"min":2.0,"max":4.5}},'gamma2':{'params':{"min":1.1,"max":4.5}},'gamma3':{'params':{"min":1.1,"max":4.5}}}
    valid_indices = ep.is_valid_eos(params,priorbounds,spectral=True)
    samples = samples[valid_indices].tolist()
    with open("files/combined/{}{}_valid_samples.json".format(Dir,label), "w") as f: json.dump(samples, f, indent=2)

def parameter_slice_plots(include_seg_faults=False, include_errors=False, s=0.01, spectral=True):
    # 6 2d plots of the 4d parameter space.

    pl.clf()

    if spectral: Dir = "spectral/"
    else: Dir = "piecewise/"

    if include_seg_faults & include_errors:
        filenames = ["files/combined/{}seg_fault_samples.json".format(Dir), "files/combined/{}no_error_samples.json".format(Dir), "files/combined/{}runtime_error_samples.json".format(Dir), "files/combined/{}value_error_samples.json".format(Dir), "files/combined/{}seg_error_valid_samples.json".format(Dir)]
        sub_Dir = "seg_error/"
        colors = ["red", "blue", "yellow", "green", "black"]
    elif (include_seg_faults == True) & (include_errors == False):
        filenames = ["files/combined/{}seg_fault_samples.json".format(Dir), "files/combined/{}no_error_samples.json".format(Dir), "files/combined/{}seg_valid_samples.json".format(Dir)]
        sub_Dir = "seg/"
        colors = ["red", "blue", "black"]
    # Added recently to make requested plot
    elif (include_seg_faults == False) & (include_errors == True):
        filenames = ["files/combined/{}no_error_samples.json".format(Dir), "files/combined/{}runtime_error_samples.json".format(Dir)] 
        sub_Dir = "error/"
        colors = ["black", "red"]
    elif (include_seg_faults == False) & (include_errors == False):
        filenames = ["files/combined/{}no_error_samples.json".format(Dir), "files/combined/{}no_error_valid_samples.json".format(Dir)]
        sub_Dir = "no_error/"
        colors = ["red", "black"]

    increment = 0
    for filename in filenames:
        print(filename)

        with open(filename, "r") as f:
            samples = np.array(json.load(f))

        g1_p1 = samples[:,0]
        g2_g1 = samples[:,1]
        g3_g2 = samples[:,2]
        g4_g3 = samples[:,3]
        
        pl.figure(1)
        pl.scatter(g1_p1,g2_g1,s=s,color=colors[increment])
        pl.xlabel("g1_p1")
        pl.ylabel("g2_g1")
        pl.title("g1_p1  g2_g1")

        pl.figure(2)
        pl.scatter(g1_p1,g3_g2,s=s,color=colors[increment])
        pl.xlabel("g1_p1")
        pl.ylabel("g3_g2")
        pl.title("g1_p1  g3_g2")

        pl.figure(3)
        pl.scatter(g1_p1,g4_g3,s=s,color=colors[increment])
        pl.xlabel("g1_p1")
        pl.ylabel("g4_g3")
        pl.title("g1_p1  g4_g3")

        pl.figure(4)
        pl.scatter(g2_g1,g3_g2,s=s,color=colors[increment])
        pl.xlabel("g2_g1")
        pl.ylabel("g3_g2")
        pl.title("g2_g1  g3_g2")

        pl.figure(5)
        pl.scatter(g2_g1,g4_g3,s=s,color=colors[increment])
        pl.xlabel("g2_g1")
        pl.ylabel("g4_g3")
        pl.title("g2_g1  g4_g3")

        pl.figure(6)
        pl.scatter(g3_g2,g4_g3,s=s,color=colors[increment])
        pl.xlabel("g3_g2")
        pl.ylabel("g4_g3")
        pl.title("g3_g2  g4_g3")

        increment += 1

    pl.figure(1)
    pl.savefig("files/plots/{}{}g1_p1__g2_g1.png".format(Dir,sub_Dir))
    
    pl.figure(2)
    pl.savefig("files/plots/{}{}g1_p1__g3_g2.png".format(Dir,sub_Dir))

    pl.figure(3)
    pl.savefig("files/plots/{}{}g1_p1__g4_g3.png".format(Dir,sub_Dir))

    pl.figure(4)
    pl.savefig("files/plots/{}{}g2_g1__g3_g2.png".format(Dir,sub_Dir))

    pl.figure(5)
    pl.savefig("files/plots/{}{}g2_g1__g4_g3.png".format(Dir,sub_Dir))

    pl.figure(6)
    pl.savefig("files/plots/{}{}g3_g2__g4_g3.png".format(Dir,sub_Dir))
