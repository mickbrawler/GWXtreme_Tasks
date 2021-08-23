# Python file meant to compute percent error for specific files

import numpy as np
import json
from glob import glob

#sample_size = len(glob("results/*.json")) - 1 # how many jsons to compare with control

with open("results/bayes_factors_SLY.json", "r") as f:
    control = json.load(f)["bayes factors"] # just want list of bayes factors

sample_holder = []
count = 0
for sample in glob("results/samples/*.json"):
    
    print("Column {}: {}".format(count,sample))
    count += 1
    with open(sample, "r") as f:
        sample_holder.append(json.load(f)["bayes factors"])

percent_errors = []
for sample in sample_holder:
    percent_errors.append(((np.array(control) - np.array(sample)) * 100) / np.array(control))

output = np.vstack((percent_errors[0],percent_errors[1])).T
outputfile = "results/percent_error.txt"
np.savetxt(outputfile, output, fmt="%f\t%f")


