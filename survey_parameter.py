from GWXtreme import eos_model_selection as ems
import numpy as np
import pylab as pl
from scipy import interpolate

#p1 [32.8805-33.9805]   middle=33.4305
#g1 [1.8430-4.4430]     middle=3.143
#g2 [1.3315-3.9315]     middle=2.6315
#g3 [1.4315-4.0315]     middle=2.7315

def survey(parameter_choice, label, N=1000, fixed_p1=33.4305, fixed_g1=3.143,
           fixed_g2=2.6315, fixed_g3=2.7315):
    
    # Survey piecewise polytropic parameters. Fix three parameters. Loop over one 
    # of the parameters. See how fast the value of the evidence changes. This is 
    # called Profiling. This way we get a good idea of what are resolution should 
    # be for each parameter.

    # parameter_choice : what parameter to loop over.
    # label : label for files.
    # N : Length of parameter space within feasible boundaries.

    modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat", spectral=False)

    counter = 1
    evidences = []
    parameters_tested1 = [] # parameter values tested
    parameters_tested2 = []

    if parameter_choice == "p1":

        p1_range = np.linspace(32.8805,33.9805,N)
        for trial_p1 in p1_range:
            try:
                evidence = modsel.eos_evidence([trial_p1,fixed_g1,fixed_g2,fixed_g3])
                evidences.append(evidence)
                print(str(counter)+":p1_value:"+str(trial_p1))
                print(str(counter)+":p1_evidence:"+str(evidence))
                counter += 1
                parameters_tested1.append(trial_p1)
            except ValueError: continue
            except RuntimeError: continue

    elif parameter_choice == "g1":

        g1_range = np.linspace(1.8430,4.4430,N)
        for trial_g1 in g1_range:
            try:
                evidence = modsel.eos_evidence([fixed_p1,trial_g1,fixed_g2,fixed_g3])
                evidences.append(evidence)
                print(str(counter)+":g1_value:"+str(trial_g1))
                print(str(counter)+":g1_evidence:"+str(evidence))
                counter += 1
                parameters_tested1.append(trial_g1)
            except ValueError: continue
            except RuntimeError: continue

    elif parameter_choice == "g2":

        g2_range = np.linspace(1.3315,3.9315,N)
        for trial_g2 in g2_range:
            try:
                evidence = modsel.eos_evidence([fixed_p1,fixed_g1,trial_g2,fixed_g3])
                evidences.append(evidence)
                print(str(counter)+":g2_value:"+str(trial_g2))
                print(str(counter)+":g2_evidence:"+str(evidence))
                counter += 1
                parameters_tested1.append(trial_g2)
            except ValueError: continue
            except RuntimeError: continue

    elif parameter_choice == "g3":

        g3_range = np.linspace(1.4315,4.0315,N)
        for trial_g3 in g3_range:
            try:
                evidence = modsel.eos_evidence([fixed_p1,fixed_g1,fixed_g2,trial_g3])
                evidences.append(evidence)
                print(str(counter)+":g3_value:"+str(trial_g3))
                print(str(counter)+":g3_evidence:"+str(evidence))
                counter += 1
                parameters_tested1.append(trial_g3)
            except ValueError: continue
            except RuntimeError: continue
    
    elif parameter_choice == "p1_g1":

        p1_range = np.linspace(32.8805,33.9805,N)
        g1_range = np.linspace(1.8430,4.4430,N)

        for trial_p1 in p1_range:
            for trial_g1 in g1_range:
                try:
                    evidence = modsel.eos_evidence([trial_p1,trial_g1,fixed_g2,fixed_g3])
                    evidences.append(evidence)
                    print(str(counter)+":p1,g1_value:{},{}".format(trial_p1,trial_g1))
                    print(str(counter)+":p1,g1_evidence:"+str(evidence))
                    counter += 1
                    parameters_tested1.append(trial_p1)
                    parameters_tested2.append(trial_g1)
                except ValueError: continue
                except RuntimeError: continue
    
    elif parameter_choice == "p1_g2":

        p1_range = np.linspace(32.8805,33.9805,N)
        g2_range = np.linspace(1.3315,3.9315,N)

        for trial_p1 in p1_range:
            for trial_g2 in g2_range:
                try:
                    evidence = modsel.eos_evidence([trial_p1,fixed_g1,trial_g2,fixed_g3])
                    evidences.append(evidence)
                    print(str(counter)+":p1,g2_value:{},{}".format(trial_p1,trial_g2))
                    print(str(counter)+":p1,g2_evidence:"+str(evidence))
                    counter += 1
                    parameters_tested1.append(trial_p1)
                    parameters_tested2.append(trial_g2)
                except ValueError: continue
                except RuntimeError: continue

    elif parameter_choice == "p1_g3":

        p1_range = np.linspace(32.8805,33.9805,N)
        g3_range = np.linspace(1.4315,4.0315,N)

        for trial_p1 in p1_range:
            for trial_g3 in g3_range:
                try:
                    evidence = modsel.eos_evidence([trial_p1,fixed_g1,fixed_g2,trial_g3])
                    evidences.append(evidence)
                    print(str(counter)+":p1,g3_value:{},{}".format(trial_p1,trial_g3))
                    print(str(counter)+":p1,g3_evidence:"+str(evidence))
                    counter += 1
                    parameters_tested1.append(trial_p1)
                    parameters_tested2.append(trial_g3)
                except ValueError: continue
                except RuntimeError: continue

    elif parameter_choice == "g1_g2":

        g1_range = np.linspace(1.8430,4.4430,N)
        g2_range = np.linspace(1.3315,3.9315,N)

        for trial_g1 in g1_range:
            for trial_g2 in g2_range:
                try:
                    evidence = modsel.eos_evidence([fixed_p1,trial_g1,trial_g2,fixed_g3])
                    evidences.append(evidence)
                    print(str(counter)+":g1,g2_value:{},{}".format(trial_g1,trial_g2))
                    print(str(counter)+":g1,g2_evidence:"+str(evidence))
                    counter += 1
                    parameters_tested1.append(trial_g1)
                    parameters_tested2.append(trial_g2)
                except ValueError: continue
                except RuntimeError: continue

    elif parameter_choice == "g1_g3":

        g1_range = np.linspace(1.8430,4.4430,N)
        g3_range = np.linspace(1.4315,4.0315,N)

        for trial_g1 in g1_range:
            for trial_g3 in g3_range:
                try:
                    evidence = modsel.eos_evidence([fixed_p1,trial_g1,fixed_g2,trial_g3])
                    evidences.append(evidence)
                    print(str(counter)+":g1,g3_value:{},{}".format(trial_g1,trial_g3))
                    print(str(counter)+":g1,g3_evidence:"+str(evidence))
                    counter += 1
                    parameters_tested1.append(trial_g1)
                    parameters_tested2.append(trial_g3)
                except ValueError: continue
                except RuntimeError: continue

    elif parameter_choice == "g2_g3":

        g2_range = np.linspace(1.3315,3.9315,N)
        g3_range = np.linspace(1.4315,4.0315,N)

        for trial_g2 in g2_range:
            for trial_g3 in g3_range:
                try:
                    evidence = modsel.eos_evidence([fixed_p1,fixed_g1,trial_g2,trial_g3])
                    evidences.append(evidence)
                    print(str(counter)+":g2,g3_value:{},{}".format(trial_g2,trial_g3))
                    print(str(counter)+":g2,g3_evidence:"+str(evidence))
                    counter += 1
                    parameters_tested1.append(trial_g2)
                    parameters_tested2.append(trial_g3)
                except ValueError: continue
                except RuntimeError: continue

    # method of saving file
    
    if len(parameter_choice) == 2:

        output = np.vstack((parameters_tested1,evidences)).T
        outputfile = "parameter_test_data/{}_{}.txt".format(parameter_choice,label)
        np.savetxt(outputfile, output, fmt="%f\t%f")

    elif len(parameter_choice) == 4:

        output = np.vstack((parameters_tested1,parameters_tested2,evidences)).T
        outputfile = "parameter_test_data/{}_{}.txt".format(parameter_choice,label)
        np.savetxt(outputfile, output, fmt="%f\t%f\t%f")

def plot_evidence(filename, parameter, label, fixed_p1=33.4305, fixed_g1=3.143,
                  fixed_g2=2.6315, fixed_g3=2.7315):

    # Plot evidences of the given parameter space to see how it changes with small changes in the parameter.

    # filename: (string) name of txt file holding evidences for a set of varying parameters
    # parameter: (string) name of the parameter that is being varied
    # label : (string) label for files.

    data = np.loadtxt(filename)

    parameters = data[:,0]
    evidences = data[:,1]

    pl.rcParams.update({'font.size':18})
    pl.figure(figsize=(20,15))
    pl.plot(parameters,evidences)
    pl.xlabel(parameter)
    pl.ylabel("Evidences")
    
    if parameter == "p1": pl.title("g1:{},g2:{},g3:{}".format(fixed_g1,fixed_g2,fixed_g3))
    elif parameter == "g1": pl.title("p1:{},g2:{},g3:{}".format(fixed_p1,fixed_g2,fixed_g3))
    elif parameter == "g2": pl.title("p1:{},g1:{},g3:{}".format(fixed_p1,fixed_g1,fixed_g3))
    elif parameter == "g3": pl.title("p1:{},g1:{},g2:{}".format(fixed_p1,fixed_g1,fixed_g2))

    pl.savefig("parameter_test_plots/{}_{}.png".format(parameter,label))

def evidence_interpolation(filename, N, parameter_choice, label):

    # Calculate the evidences through interpolation using already calculated evidences.

    # filename: (string) name of txt file holding evidences for a set of varying parameters
    # N : Length of parameter space within feasible boundaries.
    # parameter_choice : parameter that is being varied.

    data = np.loadtxt(filename)

    parameters = data[:,0]
    evidences = data[:,1]

    f = interpolate.interp1d(parameters, evidences)
    if parameter_choice=="p1": parameter_range = np.linspace(32.8805,33.9805,N)
    elif parameter_choice=="g1": parameter_range = np.linspace(1.8430,4.4430,N)
    elif parameter_choice=="g2": parameter_range = np.linspace(1.3315,3.9315,N)
    elif parameter_choice=="g3": parameter_range = np.linspace(1.4315,4.0315,N)

    evidences = f(parameter_range)   # use interpolation function returned by `interp1d`

    output = np.vstack((parameter_range,evidences)).T
    outputfile = "parameter_test_data/interp_{}_{}.txt".format(parameter_choice,label)
    np.savetxt(outputfile, output, fmt="%f\t%f")

def plot_interp_actual_evidences(actual_filename, interp_filename, parameter,
                                 label, fixed_p1=33.4305, fixed_g1=3.143,
                                 fixed_g2=2.6315, fixed_g3=2.7315):

    # Plot the actual evidences and the interpolated ones to show the tool's accuracy.

    # filename: (string) name of txt file holding evidences for a set of varying parameters
    # parameter: (string) name of the parameter that is being varied
    # label : (string) label for files.

    actual_data = np.loadtxt(actual_filename)
    actual_parameters = actual_data[:,0]
    actual_evidences = actual_data[:,1]

    interp_data = np.loadtxt(interp_filename)
    interp_parameters = interp_data[:,0]
    interp_evidences = interp_data[:,1]

    pl.rcParams.update({'font.size':18})
    pl.figure(figsize=(20,15))
    pl.plot(actual_parameters,actual_evidences, label="actual")
    pl.plot(interp_parameters,interp_evidences, label="interpolated")
    pl.xlabel(parameter)
    pl.ylabel("Evidences")
    pl.legend()

    if parameter == "p1": pl.title("g1:{},g2:{},g3:{}".format(fixed_g1,fixed_g2,fixed_g3))
    elif parameter == "g1": pl.title("p1:{},g2:{},g3:{}".format(fixed_p1,fixed_g2,fixed_g3))
    elif parameter == "g2": pl.title("p1:{},g1:{},g3:{}".format(fixed_p1,fixed_g1,fixed_g3))
    elif parameter == "g3": pl.title("p1:{},g1:{},g2:{}".format(fixed_p1,fixed_g1,fixed_g2))

    pl.savefig("parameter_test_plots/comparison_{}_{}.png".format(parameter,label))

def plot_interp_actual_evidences_2d(actual_filename, interp_filename, parameter,
                                    label, fixed_p1=33.4305, fixed_g1=3.143,
                                    fixed_g2=2.6315, fixed_g3=2.7315):

    # Plot the actual evidences and the interpolated ones as two parameters are
    # varied to show the interpolation tool's accuracy. Should produce a heat
    # plot for the actual and inerpolation runs of a 2 parameter combination. 

    # filename: (string) name of txt file holding evidences for a set of varying parameters
    # parameter: (string) name of the parameter that is being varied
    # label : (string) label for files.

    actual_data = np.loadtxt(actual_filename)
    actual_parameters1 = actual_data[:,0]
    actual_parameters2 = actual_data[:,1]
    actual_evidences = actual_data[:,2]

    interp_data = np.loadtxt(interp_filename)
    interp_parameters1 = interp_data[:,0]
    interp_parameters2 = interp_data[:,1]
    interp_evidences = interp_data[:,2]

    A1, A2 = np.meshgrid(actual_parameters1,actual_parameters2)
    I1, I2 = np.meshgrid(interp_parameters1,interp_parameters2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    ax1.set_aspect('equal')
    ax1.set_title("Actual Evidence", fontsize=8)
    actual_plot = ax1.contourf(A1,A2,actual_evidences, cmap="hot")

    fig.colorbar(actual_plot, ax=ax1)

    ax2.set_aspect('equal')
    ax2.set_title("Interp Evidence", fontsize=8)
    interp_plot = ax2.contourf(I1,I2,interp_evidences, cmap="hot")

    fig.colorbar(interp_plot, ax=ax2)

    pl.savefig("parameter_test_plots/2d_comparison_{}_{}.png".format(parameter,label))
