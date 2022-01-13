# %%
import pandas as pd
import matplotlib.pyplot as plt

#%% Load and prepare input file
datadir='/global/projecta/projectdirs/atlas/zhicaiz/Hbb/h5'

# Higgs + jet sample, higgs decays to bb
#  Use this for label 0
path=f'{datadir}/user.zhicaiz.309450.NNLOPS_nnlo_30_ggH125_bb_kt200.hbbTrain.e6281_s3126_r10201_p4258.2020_ftag5dev.v0_output.h5/user.zhicaiz.27133291._000001.output.h5'
# jet + jet sample, JZ4
#  Use this for label 1 and 2
#path=f'{datadir}/user.zhicaiz.361024.A14NNPDF23LO_jetjet_JZ4W.hbbTrain.e3668_s3126_r10201_p4258.2020_ftag5dev.v0_output.h5/user.zhicaiz.27133295._000001.output.h5'
# jet + jet sample, JZ3
#path=f'{datadir}/user.zhicaiz.361023.A14NNPDF23LO_jetjet_JZ3W.hbbTrain.e3668_s3126_r10201_p4258.2020_ftag5dev.v0_output.h5/user.zhicaiz.27133297._000001.output.h5'
df=pd.read_hdf(path,key='fat_jet')

# Ensure that we consider only boosted jets
#df=df[df.pt>500]
df

# %%
df.columns

# %% Calculate labels
label0=(df.GhostHBosonsCount==1)&(df.GhostBHadronsFinalCount==2)
label1=(df.GhostHBosonsCount==0)&(df.GhostBHadronsFinalCount==2)
label2=(df.GhostHBosonsCount==0)&(df.GhostBHadronsFinalCount!=2)

# %%
plt.hist(df[label0].pt/1e3, bins=100, range=(0,1000), label='0', histtype='step',density=True)
plt.hist(df[label1].pt/1e3, bins=100, range=(0,1000), label='1', histtype='step',density=True)
plt.hist(df[label2].pt/1e3, bins=100, range=(0,1000), label='2', histtype='step',density=True)
plt.xlabel('$p_T$ [GeV]')
plt.legend()

# %%
plt.hist(df[label0].mass/1e3, bins=100, range=(0,200), label='0', histtype='step',density=True)
plt.hist(df[label1].mass/1e3, bins=100, range=(0,200), label='1', histtype='step',density=True)
plt.hist(df[label2].mass/1e3, bins=100, range=(0,200), label='2', histtype='step',density=True)
plt.xlabel('mass [GeV]')
plt.legend()
#%%
plt.hist(df[label0].C2, bins=100, range=(0,0.5), label='0', histtype='step',density=False)
plt.hist(df[label1].C2, bins=100, range=(0,0.5), label='1', histtype='step',density=False)
plt.hist(df[label2].C2, bins=100, range=(0,0.5), label='2', histtype='step',density=False)
plt.xlabel('$C_2$')
plt.legend()

# %%

# %%
df.pt
# %%
