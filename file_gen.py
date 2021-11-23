#%% Add Delphes path env vars
os.environ['LD_LIBRARY_PATH'] += ':/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/Delphes'
os.environ['ROOT_INCLUDE_PATH']=':/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/:/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/Delphes/external'
# %% Import important packages
import ROOT

import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from tensorflow.python.ops.ragged.ragged_array_ops import size
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

CURRFILE = I_SIG[0]
fh = ROOT.TFile.Open(f'{SAMPLEDIR}/{CURRFILE}{FILE_EXT}')
t = fh.Get('Delphes')

t.Show()
# %% 
#Obtain label sizes and conf..
get_leading = True
size_labels = [0, 0, 0]
print("Scan Start")
n = 0
for e in t:
    p_obj = ParticleDict(e)
    label_0, label_1 = biased_shortlist(p_obj, CURRFILE)
    num_jet = 0

    for fj in e.GenFatJet:
        label = filter_blind(p_obj, label_0, label_1, fj.Phi, fj.Eta)
        if label == 0: 
            size_labels[0] += 1
        elif label == 1:
            size_labels[1] += 1
        else:
            size_labels[2] += 1

        n += 1
        num_jet += 1

        if get_leading and num_jet == 2:
            break
print(size_labels)
#%%
# Create an empty image corresponding to eta/phi

etamin=-1.5
etamax= 1.5
etabin=30
etawdt=(etamax-etamin)/etabin

phimin=-1.5
phimax=1.5
phibin=30
phiwdt=(phimax-phimin)/phibin

img_sizes = size_labels
image0 = np.zeros(( (img_sizes[0]) + 1,etabin,phibin), dtype=float)
image1 = np.zeros(( (img_sizes[1]) + 1,etabin,phibin), dtype=float)
image2 = np.zeros(( (img_sizes[2]) + 1,etabin,phibin), dtype=float)
elem_num = [0, 0, 0]
#%%
# Loop over all event
temp = time.localtime()
print(f"Start time: {temp.tm_hour}:{temp.tm_min}:{temp.tm_mon}")
n=0
leading_jets = True
for e in t:
    # Loop over all jets in the event
    p_obj = ParticleDict(e)
    label_0, label_1 = biased_shortlist(p_obj, CURRFILE)

    jet_num = 0
    for fj in e.GenFatJet:
        ## Homework 2
        # Add labelling information based on:
        # 0: contains a higgs boson and both b-quarks
        # 1: contains a gluon and both b-quarks
        # 2: contains a gluon and any non-b quark
        # 3: others
        
        label = filter_blind(p_obj, label_0, label_1, fj.Phi, fj.Eta)
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
            myphi= int(angle_diff(phimin, angle_diff(fj.Phi, c.Phi))/phiwdt)
            #check.append([myeta, myphi, c.PT])
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

        jet_num += 1
        if leading_jets and jet_num == 2:
            break
print("Done with Iteration")
print(f"End count: {elem_num}")
print(f"Label 0 shape: {image0.shape}")
print(f"Label 1 shape: {image1.shape}")
print(f"Label 2 shape: {image2.shape}")
# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True)
im = ax1.imshow(log10(image0[0]),extent=(etamin,etamax,phimin,phimax))
ax1.set_title("0: Hbb")
im = ax2.imshow(log10(image1[0]),extent=(etamin,etamax,phimin,phimax))
ax2.set_title("1: gbb")
im = ax3.imshow(log10(image2[0]),extent=(etamin,etamax,phimin,phimax))
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
i = 0
labels = []
stop = False
for e in t:
    p_obj = ParticleDict(e)
    label_0, label_1 = shortlist_particles(p_obj, True)
    for fj in e.GenFatJet:
        label = filter_blind(p_obj, label_0, label_1, fj.Phi, fj.Eta)
        labels.append(label)
        if label == 1:
            stop = True
            break
    if stop:
        break
    i += 1
# %%
