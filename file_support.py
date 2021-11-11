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


class ParticleDict:
    def __init__(self, t) -> None:
        self.pdict = [] #PID, Status, Phi, Eta, D1, D2
        self.params = []
        i = 0
        for p in t.Particle:
            self.pdict.append([i, p.PID, p.Status, p.D1, p.D2])
            self.params.append([p.Phi, p.Eta])
            i += 1
    def track(self, e, pflag = False, limit = 10):
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
            d1 = self.pdict[d1][3]
            d2 = self.pdict[d1][4]
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
        translate = ["Elem", "PID", "Status", "D1", "D2"]
        filters = []
        output = []
        for i in params:
            if i in translate:
                filters.append(translate.index(i))
            else:
                print(f"Incorrect filter {i}")
                return []
        for i in self.pdict:
                flag = sum([i[filters[e]] == values[e] for e in range(0, len(filters))])
                if flag:
                    output.append(i)
        return output

    def give_ds(self, e, limit = 20):
        d1 = self.pdict[e][3]
        d2 = self.pdict[e][4]
        lim = 0
        while d1 == d2:
            d1 = self.pdict[d1][3]
            d2 = self.pdict[d1][4]
            lim += 1
            if lim == limit:
                return 1, d1, d2
        return 0, d1, d2


# %%
