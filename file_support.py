# %%
import numpy as np

SAMPLEDIR='/global/cfs/cdirs/atlas/kkrizka/hbbvsgbb/samples'
DATA_DIR = "../data"
BGROUND = ["PROC_gbba"]
M_SIG = ["PROC_hbbwlnu"]
I_SIG = ["PROC_ja"]
FILE_EXT = "/Events/run_01/tag_1_delphes_events.root"
SIDE_SIZE = 30

def log10(val):
    if val <= 0:
        return np.NaN
    return np.log10(val)
log10 = np.vectorize(log10)

class EventData:
    def __init__(self, B_dict, M_dict, I_dict): 
        #Current plan to make filenames be [background, main, inc signal]
        self.data = np.zeros(1, SIDE_SIZE, SIDE_SIZE)
        self.file_array = [[], [], [], []]
        self.load_files(B_dict, BGROUND)
        self.load_files(M_dict, M_SIG)
        self.load_files(I_dict, I_SIG)

        self.labels = np.zeros(1,1)

    def load_files(self, type_dict, file_names):
        #appends jets of given file label of event
        for i in range(0, len(file_names)):
            add_labels = type_dict[i]
            for label_num in add_labels:
                curr_val = np.load(f"{DATA_DIR}/{file_names[i]}/label_{label_num}.npy")
                self.file_array[label_num].append(curr_val)


    def get_data(self, size, seed = 0):
        #return train_test_split (array, label), (array, label)
        #return (array, label) when train_size = 1
        stats = [0, 0, 0, 0] #This currently tracks the distribution of data 
        np.random.seed(seed)
        for i in range(0, size):
            label_select = np.random.uniform(0, 4)
            arr_select = np.random.uniform(0, len(self.file_array[label_select]))
            curr_arr = self.file_array[label_select][arr_select]
            #A shuffle iteration of all arrays for O(n) performance 
            # (humanely impossible to get any better in these conditions) 
            # Issues: 
            #   - Inputting size of total data does not allow 
            #     control over how much over each label specifically. In this case, 
            #     we need basically all label 0 possible. It's only a matter of 
            #     choosing how of the other labels

        pass


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
        orig_pid = self.pdict[e][1]
        elems = [e]
        if pflag:
            print(self.pdict[e])
        d1 = self.pdict[e][3]
        d2 = self.pdict[e][4]
        lim = 0
        while d1 == d2 and d1 != -1:
            elems.append(d1)
            if self.pdict[d1][1] != orig_pid:
                break
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
    def where(self, params, values, gen_track = False):
        """
        Filter function with given parameters: \n
        "Elem" : pdict list element number \n
        "PID": Particle PID \n
        "Status": Particle Status \n
        "D" List containing list of the daughter reg particles (abs) \n
        (if single particle -- any, if double then exact)
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
                        if gen_track:
                            temp, d1, d2 = self.give_ds(i[0])
                            d1, d2 = self.pdict[d1], self.pdict[d2]
                        else:
                            d1, d2 = self.pdict[i[3]], self.pdict[i[4]] 
                        if not(d1[1] in values[f] or d2[1] in values[f]):
                            flag = flag and False
                    elif i[filters[f]] != values[f]:
                        flag = flag and False
                if flag:       
                    output.append(i)
        return output

    def give_ds(self, e, limit = 200):
        """
        Backend function that returns daughter particles
        """
        orig_pid = self.pdict[e][1]
        d1 = self.pdict[e][3]
        d2 = self.pdict[e][4]
        lim = 0
        while d1 == d2 and d1 != -1:
            if self.pdict[d1][1] != orig_pid:
                break
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
        ps = []
        for i in self.pdict:
            if i[3] == elem or i[4] == elem:
                ps.append(i)
        return ps

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

    def p_in_jet(self, elem, jPhi, jEta):
        delt_r = np.sqrt(np.power(self.params[elem][0] - jPhi, 2) + np.power(self.params[elem][1] - jEta, 2))
        if delt_r < 0.5:
            return True
        
def shortlist_particles(p_obj, gen_track = False):
    """
    shortlists all potential decay particles that could fall under some label \n
    Found a gluon decaying into a single b-quark. Removing such events from given list will be beyond this function \n
    Label 2 here refers to the original label 2 plan of g->any other quark. \n
    gen_track: weather particle elements that continue throughout are evaluated with eventual daughters
    """
    def clean(l): #removing collision events
        elem = 0
        while elem < len(l):
            if len(p_obj.get_parents(l[elem][3])) > 1 or len(p_obj.get_parents(l[elem][4])) > 1:
                l.pop(elem)
            else:
                elem += 1
    #Higgs decaying into bb
    label_0 = p_obj.where(["PID", "Status"], [25, 62])

    #uu' colliding forming g that decays into bb'
    #uu' collision event gives bb' status 23
    b_5 = p_obj.where(["PID", "Status"], [5, 23])
    bb_5 = p_obj.where(["PID", "Status"], [-5, 23])
    gluons = p_obj.where(["PID"], [21])
    label_1 = [b_5, bb_5, gluons]

    #label_2 = p_obj.where(["PID", "D"], [21, [1, 2, 3, 4, 6]], gen_track)
    #clean(label_2)
    return label_0, label_1

def filter_blind(p_obj, label_0, label_1, jPhi, jEta):
    """
    File-blind function that returns label based on distance to jet \n
    Need to remove faulty decay events like the g->b here ------ *Imp*
    """
    for elem in label_0:
        if p_obj.p_in_jet(elem[0], jPhi, jEta) and p_obj.p_in_jet(elem[3], jPhi, jEta) and p_obj.p_in_jet(elem[4], jPhi, jEta):
            return 0
    label1_conf = [0, 0, 0]
    for part in range(0, len(label_1)):
        for elem in label_1[part]:
            if p_obj.p_in_jet(elem[0], jPhi, jEta):
                label1_conf[part] = 1
                break
    if all(label1_conf):
        return 1
    return 2     

def filter_func(file, jPhi, jEta, p_obj):
    pass

def gbba_shortlist(p_obj, gen_track = False):
    """
    Personalized gbba shortlist. Includes separate label lists for H, b, b'
    """
    def clean(l): #removing collision events
        elem = 0
        while elem < len(l):
            if l[elem][3] == l[elem][4] or len(p_obj.get_parents(l[elem][3])) > 1 or len(p_obj.get_parents(l[elem][4])) > 1:
                l.pop(elem)
            else:
                elem += 1

    label_0 = p_obj.where(["PID", "D"], [25, [5]], gen_track)
    clean(label_0)

    label_1H = p_obj.where(["PID"], [21], gen_track)
    clean(label_1H)
    label_1B = p_obj.where(["PID"], [5], gen_track)
    clean(label_1B)
    label_1BB = p_obj.where(["PID"], [-5], gen_track)
    clean(label_1BB)

    label_2 = p_obj.where(["PID", "D"], [21, [1, 2, 3, 4, 6]], gen_track)
    clean(label_2)
    return label_0, label_1H, label_1B, label_1BB, label_2

def gbba_filter(p_obj, label_0, label_1H, label_1B, label_1BB, label_2, jPhi, jEta):
    """
    Personalized gbba filter. Checks for any
    """
    for elem in label_0:
        if p_obj.p_in_jet(elem[0], jPhi, jEta) and p_obj.p_in_jet(elem[3], jPhi, jEta) and p_obj.p_in_jet(elem[4], jPhi, jEta):
            return 0
    flag_1 = [0, 0 , 0]
    for elem in label_1H:
        if p_obj.p_in_jet(elem[0], jPhi, jEta) and p_obj.p_in_jet(elem[3], jPhi, jEta) and p_obj.p_in_jet(elem[4], jPhi, jEta):
            flag_1[0] = 1
            break
    for elem in label_1B:
        if p_obj.p_in_jet(elem[0], jPhi, jEta) and p_obj.p_in_jet(elem[3], jPhi, jEta) and p_obj.p_in_jet(elem[4], jPhi, jEta):
            flag_1[1] = 1
            break
    for elem in label_1BB:
        if p_obj.p_in_jet(elem[0], jPhi, jEta) and p_obj.p_in_jet(elem[3], jPhi, jEta) and p_obj.p_in_jet(elem[4], jPhi, jEta):
            flag_1[2] = 1
            break
    if all(flag_1):
        return 1
    return 2     
# %%
