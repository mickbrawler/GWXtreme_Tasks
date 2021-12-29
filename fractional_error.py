import numpy as np

# Find fractional error in actual evidence and interpolated evidence

# Since the length of the actual and interpolated evidence are different (due 
# to the errors I get in calculating the actual evidence) I need to produce an 
# array with only the interpolated evidences with corresponding actual evidences

# needs to add up all the evidence files first, then calculate the error

def connect_txt_files(outputfile):

    #filenames = ["parameter_test_data/p1_N_100.txt","parameter_test_data/g1_N_100.txt","parameter_test_data/g2_N_100.txt","parameter_test_data/g3_N_100.txt"]
    filenames = ["parameter_test_data/interp_p1_N_50_to_100.txt","parameter_test_data/interp_g1_N_50_to_100.txt","parameter_test_data/interp_g2_N_50_to_100.txt","parameter_test_data/interp_g3_N_50_to_100.txt"]

    with open(outputfile,"w") as f:
        for filename in filenames:
            with open(filename) as f2:
                contents = f2.read()
                f.write(contents)

def calculate_it(actual_filename, interp_filename, parameter, label):

    actual_data = np.loadtxt(actual_filename)
    actual_parameters = actual_data[:,0]
    actual_evidences = actual_data[:,1]

    interp_data = np.loadtxt(interp_filename)
    interp_parameters = interp_data[:,0]
    interp_evidences = interp_data[:,1]

    indice = 0
    matching_interp_parameters = []
    matching_interp_evidences = []
    for parameter in interp_parameters:
        
        if actual_parameters[indice] == parameter:
            matching_interp_parameters.append(interp_parameters[indice])
            matching_interp_evidences.append(interp_evidences[indice])

        elif actual_parameters[indice] != parameter: 
            continue

        indice += 1
    
    # add by .01 to avoid division by 0 error
    error = np.sum(abs(matching_interp_evidences - (actual_evidences+.01)) / (actual_evidences+.01))

    return(error)

