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
etamin=-2.5 
etamax= 2.5
etabin=100
etawdt=(etamax-etamin)/etabin

phimin=-3.14
phimax=3.14
phibin=100
phiwdt=(phimax-phimin)/phibin

image=np.zeros((etabin,phibin), dtype=float)

# Loop over all event
n=0
for e in t:
    # Loop over all jets in the event
    for fj in t.GenFatJet:
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
        #
        # The Status contains the status in the generator. Use status 23 for the
        # b-quarks to select ones from the Feynman diagram.
        #
        # Use different samples to determine what particle the b / light quarks came
        # came from.

        # Loop over all particles in the jet
        for c in fj.Constituents:
            ## Homework 1
            # Change this to be the distance from the fat jet center.
            # ie: delta eta = c.Eta - fj.Eta
            myeta=int(np.floor((c.Eta-etamin)/etawdt))
            myphi=int(np.floor((c.Phi-phimin)/phiwdt))

            # Bounds check
            if myeta < 0 or myeta >= image.shape[0]:
                continue
            if myphi < 0 or myphi >= image.shape[1]:
                continue

            # Add to image
            image[myeta,myphi] = c.PT
        n+=1

# average the image
image/=n

# %% Show the image of the average jet shape
## Homework 3
# Make a separate plot for the three categories
plt.imshow(np.log10(image),extent=(etamin,etamax,phimin,phimax))
plt.xlabel('$\eta$')
plt.ylabel('$\phi$')

# %%
