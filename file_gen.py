#%% Add Delphes path env vars
os.environ['LD_LIBRARY_PATH'] += ':/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/Delphes'
os.environ['ROOT_INCLUDE_PATH']=':/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/:/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/Delphes/external'
# %% Import important packages
import ROOT

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
# %% Add Delphes library
ROOT.gSystem.Load(f'libDelphes')
# %% Load some samples
SAMPLEDIR='/global/cfs/cdirs/atlas/kkrizka/hbbvsgbb/samples'
DATA_DIR = "../data"
BGROUND = ["PROC_gbba"]
M_SIG = ["PROC_hbbwlnu"]
I_SIG = ["PROC_ja"]
FILE_EXT = "/Events/run_01/tag_1_delphes_events.root"

CURRFILE = BGROUND[0]
fh = ROOT.TFile.Open(f'{SAMPLEDIR}/{CURRFILE}{FILE_EXT}')
t = fh.Get('Delphes')

t.Show()
# %% Loop over all jets and create an average image

# Create an empty image corresponding to eta/phi
check = []
etamin=-1.5
etamax= 1.5
etabin=100
etawdt=(etamax-etamin)/etabin

phimin=-1.5
phimax=1.5
phibin=100
phiwdt=(phimax-phimin)/phibin


image0 = np.zeros(( (0) + 1,etabin,phibin), dtype=float)
image1 = np.zeros(( (6498) + 1,etabin,phibin), dtype=float)
image2 = np.zeros(( (304) + 1,etabin,phibin), dtype=float)
image3 = np.zeros(( (14857) + 1,etabin,phibin), dtype=float)
elem_num = [0, 0, 0, 0]
#%%
# Loop over all event
temp = time.localtime()
print(f"Start time: {temp.tm_hour}:{temp.tm_min}:{temp.tm_mon}")
n=0
for e in t:
    # Loop over all jets in the event
    for fj in e.GenFatJet:
        ## Homework 2
        # Add labelling information based on:
        # 0: contains a higgs boson and both b-quarks
        # 1: contains a gluon and both b-quarks
        # 2: contains a gluon and both light quarks
        # 3: others
        flag = [1, 2, 3, 4, 5, -5, 21, 25]
        flag = {a: 0 for a in flag}

        for p in t.Particle:
            #if p.Status != 23:
                #continue
            delt_r = np.sqrt(np.power(p.Phi - fj.Phi, 2) + np.power(p.Eta - fj.Eta, 2))
            if delt_r < 0.5:
                if p.PID in flag.keys():
                    flag[p.PID] += 1
        img_use = image3[elem_num[3]]
        if flag[5] and flag[-5] and flag[25]: 
            img_use = image0[elem_num[0]]
            elem_num[0] += 1
        elif flag[5] and flag[-5] and flag[21]:
            img_use = image1[elem_num[1]]
            elem_num[1] += 1
        elif flag[21] and ((flag[1] and flag[2]) or (flag[3] and flag[4])):
            img_use = image2[elem_num[2]]
            elem_num[2] += 1 
        else:
            elem_num[3] += 1 

        # Loop over all particles in the jet
        for c in fj.Constituents:
            myeta= int(np.floor((c.Eta - fj.Eta - etamin)/etawdt)) 
            myphi= int(np.floor((c.Phi - fj.Phi - phimin)/phiwdt)) 
            check.append([myeta, myphi, c.PT])
            # Bounds check
            if myeta < 0 or myeta >= img_use.shape[0]:
                continue
            if myphi < 0 or myphi >= img_use.shape[1]:
                continue

            # Add to image
            img_use[myeta, myphi] += c.PT
        n+=1
        
        if n % 1000 == 0:
            print(f"{n} Done: {elem_num}")
print("Done with Iteration")
print(image0.shape)
print(image1.shape)
print(image2.shape)
print(image3.shape)
# %% saving arrays into directory
np.save(f"{DATA_DIR}/{CURRFILE}/label_0", image0)
np.save(f"{DATA_DIR}/{CURRFILE}/label_1", image1)
np.save(f"{DATA_DIR}/{CURRFILE}/label_2", image2)
np.save(f"{DATA_DIR}/{CURRFILE}/label_3", image3)
