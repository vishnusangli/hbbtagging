import numpy as np

SAMPLEDIR='/global/cfs/cdirs/atlas/kkrizka/hbbvsgbb/samples'
DATA_DIR = "../data"
BGROUND = ["PROC_gbba"]
M_SIG = ["PROC_hbbwlnu"]
I_SIG = ["PROC_ja"]
FILE_EXT = "/Events/run_01/tag_1_delphes_events.root"

class EventData:
    def __init__(self, filenames): 
        #Current plan to make filenames be [background, main, inc signal]
        self.main = np.load(f"{DATA_DIR}/filenames[0]")

    def load_file(name):
        #returns some struct that holds all 4 labels of event
        pass

    def get_data(file_select, seed = 0):
        #return train_test_split (array, label), (array, label)
        #return (array, label) when train_size = 1
        
        np.random.seed(seed)
        np.random.shuffle()
        pass

