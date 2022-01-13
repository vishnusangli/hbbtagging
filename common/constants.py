
DATA_DIR = "/global/homes/v/vsangli/starters/data" 
MODEL_DIR = "/global/homes/v/vsangli/starters/models/"
TEMP_DIR = "/global/homes/v/vsangli/starters/temp_saves/"

class MG_sim:
    SAMPLEDIR='/global/cfs/cdirs/atlas/kkrizka/hbbvsgbb/samples'
    BGROUND = ["PROC_gbba"]
    M_SIG = ["PROC_hbbwlnu"]
    I_SIG = ["PROC_ja"]
    FILE_EXT = "/Events/run_01/tag_1_delphes_events.root"
    SIDE_SIZE = 30
    FEATURE_VARS = ["tau1", "tau2", "tau3", "tau4", "tau5", "PT", "EhadOverEem", "Mass"]

class MC_sim:
    FILE_DIR='/global/projecta/projectdirs/atlas/zhicaiz/Hbb/h5'