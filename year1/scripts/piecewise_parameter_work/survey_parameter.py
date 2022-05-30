from GWXtreme import eos_model_selection as ems
import numpy as np
import pylab as pl
from scipy import interpolate
import glob

#p1 [32.8805-33.9805]   middle=33.4305
#g1 [1.8430-4.4430]     middle=3.143
#g2 [1.3315-3.9315]     middle=2.6315
#g3 [1.4315-4.0315]     middle=2.7315

def vary_fixed_parameters(parameter_choice, task, fix_N, N):
    # parameter_choice  : Choice of piecewise polytropic parameters (p1,g1,g2,g3).
    # task              : Sub-directory name. Either 1d_runs or varying_fixed_parameters.
    # fix_N             : Number of fixed parameter we'll use.
    # N                 : Length of parameter space within feasible boundaries.

    fixed_p1_values = np.round(np.linspace(32.8805,33.9805,fix_N),decimals=4)
    fixed_g1_values = np.round(np.linspace(1.8430,4.4430,fix_N),decimals=4)
    fixed_g2_values = np.round(np.linspace(1.3315,3.9315,fix_N),decimals=4)
    fixed_g3_values = np.round(np.linspace(1.4315,4.0315,fix_N),decimals=4)

    if parameter_choice == "p1":
        for g1 in fixed_g1_values: 
            label = "variance_g1_{}".format(g1)
            survey(task,parameter_choice,label,N,fixed_p1=33.4305,fixed_g1=g1,fixed_g2=2.6315,fixed_g3=2.7315)
        for g2 in fixed_g2_values: 
            label = "variance_g2_{}".format(g2)
            survey(task,parameter_choice,label,N,fixed_p1=33.4305,fixed_g1=3.143,fixed_g2=g2,fixed_g3=2.7315)
        for g3 in fixed_g3_values: 
            label = "variance_g3_{}".format(g3)
            survey(task,parameter_choice,label,N,fixed_p1=33.4305,fixed_g1=3.143,fixed_g2=2.6315,fixed_g3=g3)

    if parameter_choice == "g1":
        for p1 in fixed_p1_values: 
            label = "variance_p1_{}".format(p1)
            survey(task,parameter_choice,label,N,fixed_p1=p1,fixed_g1=3.143,fixed_g2=2.6315,fixed_g3=2.7315)
        for g2 in fixed_g2_values: 
            label = "variance_g2_{}".format(g2)
            survey(task,parameter_choice,label,N,fixed_p1=33.4305,fixed_g1=3.143,fixed_g2=g2,fixed_g3=2.7315)
        for g3 in fixed_g3_values: 
            label = "variance_g3_{}".format(g3)
            survey(task,parameter_choice,label,N,fixed_p1=33.4305,fixed_g1=3.143,fixed_g2=2.6315,fixed_g3=g3)

    if parameter_choice == "g2":
        for p1 in fixed_p1_values: 
            label = "variance_p1_{}".format(p1)
            survey(task,parameter_choice,label,N,fixed_p1=p1,fixed_g1=3.143,fixed_g2=2.6315,fixed_g3=2.7315)
        for g1 in fixed_g1_values: 
            label = "variance_g1_{}".format(g1)
            survey(task,parameter_choice,label,N,fixed_p1=33.4305,fixed_g1=g1,fixed_g2=2.6315,fixed_g3=2.7315)
        for g3 in fixed_g3_values: 
            label = "variance_g3_{}".format(g3)
            survey(task,parameter_choice,label,N,fixed_p1=33.4305,fixed_g1=3.143,fixed_g2=2.6315,fixed_g3=g3)

    if parameter_choice == "g3":
        for p1 in fixed_p1_values: 
            label = "variance_p1_{}".format(p1)
            survey(task,parameter_choice,label,N,fixed_p1=p1,fixed_g1=3.143,fixed_g2=2.6315,fixed_g3=2.7315)
        for g1 in fixed_g1_values: 
            label = "variance_g1_{}".format(g1)
            survey(task,parameter_choice,label,N,fixed_p1=33.4305,fixed_g1=g1,fixed_g2=2.6315,fixed_g3=2.7315)
        for g2 in fixed_g2_values: 
            label = "variance_g2_{}".format(g2)
            survey(task,parameter_choice,label,N,fixed_p1=33.4305,fixed_g1=3.143,fixed_g2=g2,fixed_g3=2.7315)

def survey(task, parameter_choice, label, N, fixed_p1=33.4305, fixed_g1=3.143,
           fixed_g2=2.6315, fixed_g3=2.7315):
    
    # Survey piecewise polytropic parameters. Fix three parameters. Loop over one 
    # of the parameters. See how fast the value of the evidence changes. This is 
    # called Profiling. This way we get a good idea of what are resolution should 
    # be for each parameter.

    # task              : Sub-directory name. Either 1d_runs or varying_fixed_parameters.
    # parameter_choice  : What parameter to loop over.
    # label             : Label for files.
    # N                 : Length of parameter space within feasible boundaries.

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
    
    # method of saving file
    
    output = np.vstack((parameters_tested1,evidences)).T
    outputfile = "parameter_files/data/{}/{}_{}.txt".format(task,parameter_choice,label)
    np.savetxt(outputfile, output, fmt="%f\t%f")

def plot_evidence(filename, parameter, label, fixed_p1=33.4305, fixed_g1=3.143,
                  fixed_g2=2.6315, fixed_g3=2.7315):

    # Plot evidences of the given parameter space to see how it changes with small changes in the parameter.

    # filename: (string) Name of txt file holding evidences for a set of varying parameters
    # parameter: (string) Name of the parameter that is being varied
    # label : (string) Label for files.

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

    pl.savefig("parameter_files/plots/{}_{}.png".format(parameter,label))

def evidence_interpolation(filename, N, parameter_choice, label):

    # Calculate the evidences through interpolation using already calculated evidences.

    # filename: (string) Name of txt file holding evidences for a set of varying parameters
    # N : Length of parameter space within feasible boundaries.
    # parameter_choice : Parameter that is being varied.
    # label : (string) Label for files.

    data = np.loadtxt(filename)

    parameters = data[:,0]
    evidences = data[:,1]

    f = interpolate.interp1d(parameters, evidences, bounds_error=False)
    if parameter_choice=="p1": parameter_range = np.linspace(32.8805,33.9805,N)
    elif parameter_choice=="g1": parameter_range = np.linspace(1.8430,4.4430,N)
    elif parameter_choice=="g2": parameter_range = np.linspace(1.3315,3.9315,N)
    elif parameter_choice=="g3": parameter_range = np.linspace(1.4315,4.0315,N)

    evidences = f(parameter_range)   # use interpolation function returned by `interp1d`

    output = np.vstack((parameter_range,evidences)).T
    outputfile = "parameter_files/data/1d_runs/interp_{}_{}.txt".format(parameter_choice,label)
    np.savetxt(outputfile, output, fmt="%f\t%f")

def plot_interp_actual_evidences(actual_filename, interp_filename, N_start,
                                 parameter, label, fixed_p1=33.4305, 
                                 fixed_g1=3.143, fixed_g2=2.6315, fixed_g3=2.7315):

    # Plot the actual evidences and the interpolated ones to show the tool's accuracy.

    # filename: (string) Name of txt file holding evidences for a set of varying parameters
    # parameter: (string) Name of the parameter that is being varied
    # label : (string) Label for files.

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

    pl.savefig("parameter_files/plots/{}/comparison_{}_{}.png".format(N_start,parameter,label))

def plot_varying_fixed_parameter(fixed_p0,fixed_pf):
    # fixed_p0  : Iterated parameter. 
    # fixed_pf  : Varying fixed parameter.

    pl.clf()

    filenames = glob.glob("parameter_files/data/varying_fixed_parameters/{}_variance_{}*".format(fixed_p0,fixed_pf))
    
#    if fixed_pf == "p1": index_0 = 60
#    else: index_0 = 61
    index_0 = 61
    index_f = -4

    for filename in filenames:
        data = np.loadtxt(filename)
        try: parameters = data[:,0]
        except IndexError: continue
        evidences = data[:,1]
        pf_value = filename[index_0:index_f]
        print(pf_value)
        
        pl.plot(parameters,evidences,label="{}={}".format(fixed_pf,pf_value))

    pl.xlabel(fixed_p0)
    pl.ylabel("Evidences")
    pl.title("{}_variance_{}".format(fixed_p0,fixed_pf))
    pl.legend()
    pl.savefig("parameter_files/plots/varying_fixed_parameters/{}_variance_{}.png".format(fixed_p0,fixed_pf))
    pl.clf()

