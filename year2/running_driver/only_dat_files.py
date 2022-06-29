import glob
import os.path

# Script that moves unnecessary .dat lacking event directorys across
# all snrbins. Wanted it to sort the lowest and highest 5 mass events
# but needs more thinking.

paths = ["dat_files/snrbin/APR4_EPP/13_to_15/",
         "dat_files/snrbin/APR4_EPP/23_to_25/",
         "dat_files/snrbin/APR4_EPP/33_to_35/"]

for path in paths:
    
    #low_masses = []
    Dirs = glob.glob(path + "*")
    for Dir in Dirs:

        test_file = Dir + "/bns_example_samples.dat"
        test_Dir = Dir + "/"
        if os.path.exists(test_file) == False:
            print("False")
            os.system("mv {} {}no_dat/".format(test_Dir, path))
        else:
            print("True")
            #sub_Dir = test_Dir[len(path):]
            #low_mass = int(sub_Dir[sub_Dir.find("_")+1:sub_Dir[::-1].find("_")-1])
            #low_masses.append(low_mass)
            
