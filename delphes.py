# %% Import important packages
import ROOT

import numpy as np
import matplotlib.pyplot as plt

# %% Add Delphes library
ROOT.gSystem.Load(f'libDelphes')

# %% Load some samples
SAMPLEDIR='/global/cfs/cdirs/atlas/kkrizka/hbbvsgbb/samples'
fh = ROOT.TFile.Open(f'{SAMPLEDIR}/PROC_gbba/Events/run_01/tag_1_delphes_events.root')
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

image = np.zeros((etabin,phibin), dtype=float)
image0 = np.zeros((etabin,phibin), dtype=float)
image1 = np.zeros((etabin,phibin), dtype=float)
image2 = np.zeros((etabin,phibin), dtype=float)
image3 = np.zeros((etabin,phibin), dtype=float)
#%%
# Loop over all event
a = 0
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
        #
        # Generated particles can be found in the `t.Particle` branch of the
        # input TTree.
        #
        # Do a delta R match.
        # delta R = sqtt((part.eta-jet.eta)**2+(part.phi-jet.phi)**2)
        # match is delta R < 0.5
        #
        # The PID contains the PDG ID of the particle (aka the type) using the
        # scheme at https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
        # The Status contains the status in the generator. Use status 23 for the
        # b-quarks. Use status 43(confirm?) for the gluon.
        flag = [1, 2, 5, -5, 21, 35]
        flag = {a: 0 for a in flag}

        for p in t.Particle:
            print(p.ClassName())
            print(dir(p))
            assert(0> 1), "Exit Point"
            if p.Status != 23:
                continue
            delt_r = np.sqrt(np.power(p.Phi - fj.Phi, 2) + np.power(p.Eta - fj.Eta, 2))
            if delt_r < 0.5:
                if p.PID in flag.keys():
                    flag[p.PID] += 1
        
        img_use = image3
        if flag[5] and flag[-5] and flag[35]: #Guessing the order takes precedence. This exclusivity of image objects would otherwise allow same jet to be classified in different ones
            img_use = image0
        if flag[5] and flag[-5] and flag[21]:
            img_use = image1
        if flag[21] and flag[1] and flag[2]:
            img_use = image2        
        a += 1

        # Loop over all particles in the jet
        for c in fj.Constituents:
            ## Homework 1
            # Change this to be the distance from the fat jet center.
            # ie: delta eta = c.Eta - fj.Eta
            myeta= int(np.floor((c.Eta - fj.Eta - etamin)/etawdt)) 
            myphi= int(np.floor((c.Phi - fj.Phi - phimin)/phiwdt)) 
            check.append([myeta, myphi, c.PT])
            # Bounds check
            if myeta < 0 or myeta >= image.shape[0]:
                continue
            if myphi < 0 or myphi >= image.shape[1]:
                continue

            # Add to image
            image[myeta,myphi] += c.PT
            img_use[myeta, myphi] += c.PT
        n+=1

# average the image
image/=n
# %% Show the image of the average jet shape
## Homework 3
# Make a separate plot for the three categories
plt.imshow(image,extent=(etamin,etamax,phimin,phimax))
plt.xlabel('$\eta$')
plt.ylabel('$\phi$')
plt.colorbar()
plt.show()
# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(image0,extent=(etamin,etamax,phimin,phimax))
ax2.imshow(image1,extent=(etamin,etamax,phimin,phimax))
ax3.imshow(image2,extent=(etamin,etamax,phimin,phimax))
ax4.imshow(image3,extent=(etamin,etamax,phimin,phimax))
plt.show()
# %%
