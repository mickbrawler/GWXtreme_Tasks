import numpy as np
import json

def combine(cores, new):

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
            samples = np.loadtxt(File).tolist()
            if filetype == "seg_faults.txt": seg_fault_samples += samples
            elif filetype == "no_errors.txt": no_error_samples += samples
            elif filetype == "errors.txt": error_samples += samples

    with open("files/combined/seg_fault_samples.json", "w") as f: json.dump(seg_fault_samples, f, indent=2)
    with open("files/combined/no_errors_samples.json", "w") as f: json.dump(no_error_samples, f, indent=2)
    with open("files/combined/errors_samples.json", "w") as f: json.dump(error_samples, f, indent=2)

