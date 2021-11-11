#%% Add Delphes path env vars
os.environ['LD_LIBRARY_PATH'] += ':/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/Delphes'
os.environ['ROOT_INCLUDE_PATH']=':/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/:/global/homes/v/vsangli/starters/MG5_aMC_v3_2_0/Delphes/external'
# %% Import important packages
import ROOT
import numpy as np
import matplotlib.pyplot as plt
#from file_support import *
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
class ParticleDict:
    def __init__(self, t) -> None:
        """
        Particle Dictionary to better interact with event particles
        """
        self.pdict = [] #PID, Status, Phi, Eta, D1, D2
        self.params = []
        i = 0
        for p in t.Particle:
            self.pdict.append([i, p.PID, p.Status, p.D1, p.D2])
            self.params.append([p.Phi, p.Eta])
            i += 1
    def track(self, e, pflag = True, limit = 10):
        """
        Outputs the different entries of particles and the final daughters. \n
        pflag: False removes verbose std output
        """
        #tracks given particle until it decays into two diff
        elems = [e]
        if pflag:
            print(self.pdict[e])
        d1 = self.pdict[e][3]
        d2 = self.pdict[e][4]
        lim = 0
        while d1 == d2:
            elems.append(d1)
            if pflag:
                print(self.pdict[d1])
            d2 = self.pdict[d1][4]
            d1 = self.pdict[d1][3]
            lim += 1
            if lim == limit:
                print(f"Larger than func limit {limit}")
                break
        if pflag:
            print(self.pdict[d1])
            print(self.pdict[d2])
        elems.append(d1)
        elems.append(d2)
        return elems
    def where(self, params, values):
        """
        Filter function with given parameters: \n
        "Elem" : pdict list element number \n
        "PID": Particle PID \n
        "Status": Particle Status \n
        "D" List containing the daughter reg particles (abs) \n
        """
        translate = ["Elem", "PID", "Status", "D"]
        filters = []
        output = []
        for i in params:
            if i in translate:
                if i == "D":
                    filters.append(i)
                else:
                    filters.append(translate.index(i))
            else:
                print(f"Incorrect filter {i}")
                return []
        for i in self.pdict:
                #flag = all([i[filters[e]] == values[e] for e in range(0, len(filters))])
                flag = True
                for f in range(0, len(filters)):
                    if filters[f] == "D":
                        d1, d2 = self.pdict[i[3]], self.pdict[i[4]] 
                        if not(d1[1] in values[f] or d2[1] in values[f]):
                            flag = flag and False
                    elif i[filters[f]] != values[f]:
                        flag = flag and False
                if flag:       
                    output.append(i)
        return output

    def give_ds(self, e, limit = 20):
        """
        Backend function that returns daughter particles
        """
        d1 = self.pdict[e][3]
        d2 = self.pdict[e][4]
        lim = 0
        while d1 == d2:
            d2 = self.pdict[d1][4]
            d1 = self.pdict[d1][3]
            lim += 1
            if lim == limit:
                return 0, d1, d2
        return 1, d1, d2
    
    def get_decays(self, PID, track = False):
        """
        Returns a dictionary containing PID and count of particles given arg PID decays into
        """
        vals = self.where(["PID"], [PID])
        output = {}
        for i in vals:
            success, d1, d2 = self.give_ds(i[0])
            ds = [i[3], i[4]]
            ds = [self.pdict[m][1] for m in ds]
            for m in ds:
                output[m] = output.get(m, 0) + 1
        return output
    
    def get_parents(self, elem):
        """
        Returns parent particle of given arg particle elem number
        """
        for i in self.dict:
            if i[3] == elem or i[4] == elem:
                return i

    def get_info(self):
        """
        Returns dictionary count of all particles in dict
        """
        output = {}
        for i in self.pdict:
            output[i[1]] = output.get(i[1], 0) + 1
        return output

    def in_jet(self, jPhi, jEta):
        """
        Returns list of particles that are within the jet bounds
        """
        output = []
        for elem in range(0, len(self.pdict)):
            delt_r = np.sqrt(np.power(self.params[elem][0] - jPhi, 2) + np.power(self.params[elem][1] - jEta, 2))
            if delt_r < 0.5:
                output.append(self.pdict[elem])
        return output

    def find_event(self, PID, d_ref, status = None):
        vals = self.where(["PID"])
        
def filter_func_blind(p_obj, jPhi, jEta):
    """
    File-blind filter that searches for all labels
    """
    H_parts = p_obj.where(["PID"], [25])
    pass
        

    
def filter_func(file, jPhi, jEta, p_obj):
    def in_jet(elem):
        delt_r = np.sqrt(np.power(p_obj.params[elem][0] - jPhi, 2) + np.power(p_obj.params[elem][1] - jEta, 2))
        return delt_r < 0.5
    def check_event(elem, PID, d1_ref, status = 0):
        if p_obj.pdict[elem][1] == PID and in_jet(elem):
            f, d1, d2 = p_obj.give_ds(elem)
            if f:
                if p_obj.pdict[d1][1] in d1_ref and  d1 == -d2 and in_jet(d1) and in_jet(d2):
                    return True, [PID, d1, d2]
        return False, []
    fail = 0
    for p in range(0, len(p_obj.pdict)):
        if True:
            if in_jet(p):
                print(f"In {p_obj.pdict[p]}")
            if check_event(p, 21, [5])[0]:

                return 1
            elif check_event(p, 25, [5])[0]:
                return 0
            else:
                success, vals = check_event(p, 21, [1, 2, 3, 4, 6])
                if success:
                    return 2
                else:
                    fail += 1
                return 3

        elif file == "PROC_hbbwlnu":
            if check_event(p, 25, [1, 2, 3, 4, 6]):
                flag[25] += 1
                flag[5] += 1
                flag[-5] += 1
        elif file == "PROC_ja":
            flag[21] += 1
            flag[1] += 1
            flag[-1] += 1
        else:
            raise SyntaxError("Incorrect filename")
        return 4
#%%
p_obj = ParticleDict(t)
# %%
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
