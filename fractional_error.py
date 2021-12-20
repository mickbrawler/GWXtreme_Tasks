import numpy as np

# Find fractional error in actual evidence and interpolated evidence

# Since the length of the actual and interpolated evidence are different (due 
# to the errors I get in calculating the actual evidence) I need to produce an 
# array with only the interpolated evidences with corresponding actual evidences

def calculate_it(actual_filename, interp_filename, parameter, label):

    
