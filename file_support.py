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


def filter_func(PID, Status, delt_r, flag, file):
    if delt_r < 0.5:
        if file == "PROC_gbba": #Background
            #needs p.Status == 23 for jets from hard process
            if PID in flag.keys(): #and Status == 23:??
                if PID != 5 and PID != 5:
                    flag[PID] += 1
                elif Status == 23:
                    flag[PID] += 1
        elif file == "PROC_hbbwlnu":
            #Specific scenario of Higgs decaying -- Uncertain as to how the daughter branch works again
            #
            if Status == 62 and PID == 25:
                flag[25] += 1
                flag[5] += 1 
                flag[-5] += 1
            elif PID != 25 and PID in flag.keys():
                flag[PID] += 1
        elif file == "PROC_ja":
            if PID in flag.keys():
                flag[PID] += 1
        else:
            raise SyntaxError("Invalid file name for filter function")  