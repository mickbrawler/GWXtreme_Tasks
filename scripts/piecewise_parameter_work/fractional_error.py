import numpy as np
import glob

# Find fractional error in actual evidence and interpolated evidence

# Since the length of the actual and interpolated evidence are different (due 
# to the errors I get in calculating the actual evidence) I need to produce an 
# array with only the interpolated evidences with corresponding actual evidences

# needs to add up all the evidence files first, then calculate the error

def connect_txt_files(txts_path, outputfile):
    # txts_path     : Directory for txt files.
    # outputfile    : Filename holding combined data of multiple txt files.

    filenames = glob.glob("{}*.txt".format(txts_path))

    with open(outputfile,"w") as f:
        for filename in filenames:
            with open(filename) as f2:
                contents = f2.read()
                f.write(contents)

def calculate_it(actual_filename, interp_filename, N_type, label):
    # actual_filename   : Actual evidences file.
    # interp_filename   : Interpolated evidences file.
    # N_type            : N used for interpolent is a necessary sub-directory.
    # label             : Naming for file.

    actual_data = np.loadtxt(actual_filename)
    actual_parameters = actual_data[:,0]
    actual_evidences = actual_data[:,1]

    interp_data = np.loadtxt(interp_filename)
    interp_parameters = interp_data[:,0]
    interp_evidences = interp_data[:,1]

    indice = 0
    matching_actual_evidences = []
    matching_interp_evidences = []
    matching_parameters = []

    while indice < len(actual_evidences):
        
        if actual_parameters[indice] in interp_parameters:

            if np.isnan(interp_evidences[np.where(interp_parameters == actual_parameters[indice])]): 

                indice += 1

            else:

                matching_parameters.append(actual_parameters[indice])
                matching_actual_evidences.append(actual_evidences[indice])
                matching_interp_evidences.append(interp_evidences[np.where(interp_parameters == actual_parameters[indice])][0])
                indice += 1
        
        else: indice += 1

    matching_parameters = np.array(matching_parameters)
    matching_actual_evidences = np.array(matching_actual_evidences)
    matching_interp_evidences = np.array(matching_interp_evidences)
    
    # If there are 0s present in the evidences, get them out of there!
    mask = matching_actual_evidences != 0.0
    matching_parameters = matching_parameters[mask]
    matching_actual_evidences = matching_actual_evidences[mask]
    matching_interp_evidences = matching_interp_evidences[mask]

    error = np.absolute(matching_interp_evidences - matching_actual_evidences) / matching_actual_evidences

    output =  output = np.vstack((matching_parameters,matching_actual_evidences,matching_interp_evidences,error)).T
    outputfile = "parameter_files/data/error_analysis/{}/error_{}.txt".format(N_type,label)
    np.savetxt(outputfile, output, fmt="%f\t%f\t%f\t%f")

