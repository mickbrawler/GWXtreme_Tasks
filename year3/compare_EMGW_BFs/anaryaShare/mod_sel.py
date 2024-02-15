from GWXtreme.eos_model_selection import Stacking
import numpy as np
import h5py

import glob
import json
import sys

import matplotlib.pyplot as plt

EoS2 = 'APR4_EPP'
EoS1 = ["ALF1", "ALF2", "ALF3", "ALF4",
         "AP1", "AP2", "AP3", "AP4", "APR4_EPP",
         "BBB2", "BGN1H1", "BPAL12", 
         "BSK19", "BSK20", "BSK21",
         "ENG", "FPS", "GNH3",
         "GS1", "GS2",
         "H1", "H2", "H3", "H4", "H5", "H6", "H7",
         "MPA1", "MS1B", "MS1B_PP", "MS1_PP", "MS1", "MS2",
         "PAL6", "PCL2", "PS",
         "QMC700",
         "SLY4", "SLY",
         "SQM1", "SQM2", "SQM3",
         "WFF1", "WFF2", "WFF3",
         "APR", "BHF_BBB2",
         "KDE0V", "KDE0V1", "RS", "SK255", "SK272",
         "SKA", "SKB", "SKI2", "SKI3", "SKI4", "SKI5", "SKI6",
         "SKMP", "SKOP",
         "SLY2", "SLY230A", "SLY9",
         "HQC18"]


GW_events = ['Files/posterior_samples_narrow_spin_prior_170817.dat']

EM_events = ['Files/J0030_3spot_RM.txt','Files/NICER+XMM_J0740_RM.txt']


modsels = Stacking(GW_events,em_event_list=EM_events,spectral = True)
print(modsels.stack_events('SLY','APR4_EPP'))
EoS2 = 'SLY'

with open('outdir/Joint_BF_wrt_SLY_170817+0030+0740.txt','w') as out_file:
    out_file.write("Eos1       joint-BF")
    out_file.write("\n")
    
    for i,eos1 in enumerate(EoS1):
        jbf2d = modsels.stack_events(eos1,EoS2)
        out_file.write(eos1+"\t"+str(jbf2d))
        out_file.write("\n")
        print(i,len(EoS1),eos1,EoS2,jbf2d)



all_eos = np.array(['BHF_BBB2', 'KDE0V', 'KDE0V1', 'SKOP', 'H4', 'HQC18', 'SKMP', 'SLY9', 'APR4_EPP', 'SKI2', 'SKI4', 'SKI6', 'MPA1', 
                    'SLY2', 'SLY230A', 'MS1_PP', 'MS1B_PP'])

bf2d = np.loadtxt('outdir/Joint_BF_wrt_SLY_170817+0030+0740.txt',usecols = [1],dtype=float,unpack=True,skiprows=1)
EoS = np.loadtxt('outdir/Joint_BF_wrt_SLY_170817+0030+0740.txt',usecols = [0],dtype=str,unpack=True,skiprows=1)
arg = np.array([np.where(eos==EoS)[0] for eos in all_eos]) 
EoS = EoS[arg].reshape(len(arg))
bf2d=bf2d[arg].reshape(len(arg))
X_axis = np.arange(len(EoS))
plt.close()
fig = plt.figure(figsize=(15,10))
#plt.bar(X_axis - 0.2, bf3d, 0.4, label = '3d')
plt.bar(X_axis + 0.2, bf2d, 0.4, label = '2d')
plt.yscale('log')
plt.xticks(X_axis, EoS)
plt.ylim(1.0e-4,max(bf2d)*10.)
plt.axhline(1.)
plt.legend()
fig.savefig('outdir/17+J0300+J0740_modsel.png')
