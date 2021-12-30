import numpy as np
import glob

# Find fractional error in actual evidence and interpolated evidence

# Since the length of the actual and interpolated evidence are different (due 
# to the errors I get in calculating the actual evidence) I need to produce an 
# array with only the interpolated evidences with corresponding actual evidences

# needs to add up all the evidence files first, then calculate the error

def connect_txt_files(txts_path,outputfile):

    filenames = glob.glob("{}*.txt".format(txts_path))

    with open(outputfile,"w") as f:
        for filename in filenames:
            with open(filename) as f2:
                contents = f2.read()
                f.write(contents)

def calculate_it(actual_filename, interp_filename, parameter):

    actual_data = np.loadtxt(actual_filename)
    actual_parameters = actual_data[:,0]
    actual_evidences = actual_data[:,1]

    interp_data = np.loadtxt(interp_filename)
    interp_parameters = interp_data[:,0]
    interp_evidences = interp_data[:,1]

    indice = 0
    matching_actual_evidences = []
    matching_interp_evidences = []

    while indice < len(actual_evidences):
        
        if actual_parameters[indice] in interp_parameters:

            if np.isnan(interp_evidences[np.where(interp_parameters == actual_parameters[indice])]): 

                indice += 1

            else:

                matching_actual_evidences.append(actual_evidences[indice])
                matching_interp_evidences.append(interp_evidences[np.where(interp_parameters == actual_parameters[indice])])
                indice += 1
        
        else: indice +=1
    
    error = np.absolute(np.array(matching_interp_evidences) - (np.array(matching_actual_evidences) + 1.0)) / (np.array(matching_actual_evidences) + 1.0)

    print(error)

