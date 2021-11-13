#%% Add Delphes path env vars
os.environ['LD_LIBRARY_PATH'] += ':/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/Delphes'
os.environ['ROOT_INCLUDE_PATH']=':/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/:/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/Delphes/external'
# %% Import important packages
import ROOT

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from file_support import *
# %% Add Delphes library
ROOT.gSystem.Load(f'libDelphes')
# %% Load some samples
SAMPLEDIR='/global/cfs/cdirs/atlas/kkrizka/hbbvsgbb/samples'
DATA_DIR = "../data"
BGROUND = ["PROC_gbba"]
M_SIG = ["PROC_hbbwlnu"]
I_SIG = ["PROC_ja"]
FILE_EXT = "/Events/run_01/tag_1_delphes_events.root"

CURRFILE = M_SIG[0]
fh = ROOT.TFile.Open(f'{SAMPLEDIR}/{CURRFILE}{FILE_EXT}')
t = fh.Get('Delphes')

t.Show()
# %% 
# Create an empty image corresponding to eta/phi
check = []
etamin=-1.5
etamax= 1.5
etabin=30
etawdt=(etamax-etamin)/etabin

phimin=-1.5
phimax=1.5
phibin=30
phiwdt=(phimax-phimin)/phibin

img_sizes = [9960, 21, 545, 7870]
image0 = np.zeros(( (img_sizes[0]) + 1,etabin,phibin), dtype=float)
image1 = np.zeros(( (img_sizes[1]) + 1,etabin,phibin), dtype=float)
image2 = np.zeros(( (img_sizes[2]) + 1,etabin,phibin), dtype=float)
elem_num = [0, 0, 0]
#%%
# Loop over all event
temp = time.localtime()
print(f"Start time: {temp.tm_hour}:{temp.tm_min}:{temp.tm_mon}")
n=0
for e in t:
    # Loop over all jets in the event
    p_obj = ParticleDict(e)
    label_0, label_1, label_2 = shortlist_particles(p_obj, False)
    for fj in e.GenFatJet:
        ## Homework 2
        # Add labelling information based on:
        # 0: contains a higgs boson and both b-quarks
        # 1: contains a gluon and both b-quarks
        # 2: contains a gluon and any non-b quark
        # 3: others
        
        label = filter_blind(p_obj, label_0, label_1, label_2, fj.Phi, fj.Eta)
        img_use = image2[elem_num[2]]
        if label == 0: 
            img_use = image0[elem_num[0]]
            elem_num[0] += 1
        elif label == 1:
            img_use = image1[elem_num[1]]
            elem_num[1] += 1
        else:
            elem_num[2] += 1 

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

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True)
im = ax1.imshow(log10(image0[0]),extent=(etamin,etamax,phimin,phimax))
ax1.set_title("0: Hbb")
im = ax2.imshow(log10(image1[1]),extent=(etamin,etamax,phimin,phimax))
ax2.set_title("1: gbb")
im = ax3.imshow(log10(image2[1]),extent=(etamin,etamax,phimin,phimax))
ax3.set_title("2: g*")
plt.suptitle(f"{CURRFILE} Example jet substructure")
fig.subplots_adjust(right=0.8)
fig.colorbar(im, ax = [ax1, ax2, ax3, ax4])
plt.savefig(f"{DATA_DIR}/{CURRFILE}/labels.png", facecolor = 'white', edgecolor = 'white')
plt.show()
# %% saving arrays into directory
np.save(f"{DATA_DIR}/{CURRFILE}/label_0", image0)
np.save(f"{DATA_DIR}/{CURRFILE}/label_1", image1)
np.save(f"{DATA_DIR}/{CURRFILE}/label_2", image2)

# %%
