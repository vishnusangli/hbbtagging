#%% Add Delphes path env vars
os.environ['LD_LIBRARY_PATH'] += ':/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/Delphes'
os.environ['ROOT_INCLUDE_PATH']=':/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/:/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/Delphes/external'
# %% Import important packages
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from file_support import *
# %% Add Delphes library
ROOT.gSystem.Load(f'libDelphes')
print("checkpoint")
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

t.Show(0)
# Create an empty image corresponding to eta/phi
check = []
etamin=-1.5
etamax= 1.5
etabin=30
etawdt=(etamax-etamin)/etabin

phimin=-1.5
phimax=1.5
phibin= 30
phiwdt=(phimax-phimin)/phibin

image = np.zeros((etabin,phibin), dtype=float)
image0 = np.zeros((etabin,phibin), dtype=float)
image1 = np.zeros((etabin,phibin), dtype=float)
image2 = np.zeros((etabin,phibin), dtype=float)
image3 = np.zeros((etabin,phibin), dtype=float)
# %%
p_obj = ParticleDict(t)

# %% Track status of particles
status_dict = {}
def status_track(status):
    if status in status_dict.keys():
        status_dict[status] += 1
    else: 
        status_dict[status] = 1
#%%
# Loop over all events
n=0
num_vals = [0, 0, 0, 0]
quit = False
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
        # b-quarks to select ones from the Feynman diagram.
        #
        # Use different samples to determine what particle the b / light quarks came
        # came from.
        

        #PDGID - 1d, 2u, 3s, 4c, 5b, 6t
        #21 - g
        #25 - H
        flag = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 21, 25]
        flag = {a: 0 for a in flag}
        for p in t.Particle:
            delt_r = np.sqrt(np.power(p.Phi - fj.Phi, 2) + np.power(p.Eta - fj.Eta, 2))
            filter_func(p.PID, p.Status, delt_r, flag, CURRFILE)
            if quit:
                break
        
        img_use = image3
        if flag[5] and flag[-5] and flag[25]: #Guessing the order takes precedence. This exclusivity of image objects would otherwise allow same jet to be classified in different ones
            num_vals[0] += 1
            img_use = image0
        elif flag[5] and flag[-5] and flag[21]:
            num_vals[1] += 1
            img_use = image1
        elif flag[21] and sum([(flag[i] and flag[-i]) for i in [1, 2, 3, 4, 6]]):
            num_vals[2] += 1
            img_use = image2  
        else:
            num_vals[3] += 1    
        if quit:  
            break
        # Loop over all particles in the jet
        for c in fj.Constituents:
            ## Homework 1
            # Change this to be the distance from the fat jet center.
            # ie: delta eta = c.Eta - fj.Eta
            myeta= int(np.floor((c.Eta - fj.Eta - etamin)/etawdt)) 
            myphi= int(np.floor((c.Phi - fj.Phi - phimin)/phiwdt)) 
            #check.append([myeta, myphi, c.PT])
            # Bounds check
            if myeta < 0 or myeta >= image.shape[0]:
                continue
            if myphi < 0 or myphi >= image.shape[1]:
                continue

            # Add to image
            image[myeta,myphi] += c.PT
            img_use[myeta, myphi] += c.PT
        n+=1
        if n % 1000 == 0:
            print(f"{n} Done: {num_vals}")
    if quit:
        break

# average the image
image/=n
print("Done with Iteration")
## %% Average the plots
image0/=(1 or num_vals[0])
image1/=(1 or num_vals[1])
image2/=(1 or num_vals[2])
image3/=(1 or num_vals[3])
print(f"Label sizes: {num_vals}")
# %% Show the image of the average jet shape
plt.imshow(log10(image),extent=(etamin,etamax,phimin,phimax))
plt.xlabel('$\eta$')
plt.ylabel('$\phi$')
plt.colorbar()
plt.title(f"{CURRFILE} Average jet substructure")
plt.savefig(f"{DATA_DIR}/{CURRFILE}/total.png", facecolor = 'white', edgecolor = 'white')
plt.show()
# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True)
im = ax1.imshow(log10(image0),extent=(etamin,etamax,phimin,phimax))
ax1.set_title("0: Hbb")
im = ax2.imshow(log10(image1),extent=(etamin,etamax,phimin,phimax))
ax2.set_title("1: gbb")
im = ax3.imshow(log10(image2),extent=(etamin,etamax,phimin,phimax))
ax3.set_title("2: gsc, gtd")
im = ax4.imshow(log10(image3),extent=(etamin,etamax,phimin,phimax))
ax4.set_title("3: Other")
plt.suptitle(f"{CURRFILE} Average jet substructure")
fig.subplots_adjust(right=0.8)
fig.colorbar(im, ax = [ax1, ax2, ax3, ax4])
plt.savefig(f"{DATA_DIR}/{CURRFILE}/labels.png", facecolor = 'white', edgecolor = 'white')
plt.show()
