# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:28:47 2016

@author: renatobottermaiolopesrodrigues
"""

import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path as op
import mne
#from mne.datasets import sample
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs 
from solvenan2 import *
from run_coupled_ntf import *
from scipy import signal
from spectro_analysis import *
from scipy.stats.stats import pearsonr   

from ValidationICA import * 
from sklearn import metrics

import seaborn
 
###############################################################################
# close all figures
 
plt.close("all")

###############################################################################

#data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')

data_path='/Users/renatobottermaiolopesrodrigues/Desktop/Backup/Data/'

raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'), preload=True)
events = mne.read_events(op.join(data_path, 'sample_audvis_raw-eve.fif'))
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)

###############################################################################
#Hight-Pass Filter
raw.filter(l_freq=1, h_freq=None)

#Get results from ICA method
eeg_only_after_ica=ica_method(raw.copy(), picks, save='y')

###############################################################################
raw.plot_sensors(ch_type='eeg')
plt.show()
pylab.savefig('0.png')

################################################################################

eeg_only = raw.copy().pick_types(meg=False, eeg=True)
eog_only = raw.copy().pick_types(meg=False, eeg=False, eog=True)

####################################################################################

sfreq = eeg_only.info['sfreq']
Fs=sfreq
Tau = 0.7
hopsize=1/8
win='h'
Nw=np.round(Tau * Fs)
R = np.round(hopsize * Nw) 
N = eeg_only._data.shape[1]
Nt = np.array([np.fix((N-Nw)/R)])

#Channels to analyse 
ch=np.arange(0,59,1)
 
V2 = np.zeros([int(Nw), int (Nt[0]), len(ch)]) 
v2 = np.zeros([int(Nw), int (Nt[0]), len(ch)])                       
Ph2 = np.zeros([int(Nw), int (Nt[0]), len(ch)])            
n2=np.zeros([len(ch),1])
#
#
####################################################################################

for m in range(0, len(ch)):   
    V2[:,:,m], Ph2[:,:,m], tr2, f2 = spectrogramme(eeg_only._data[ch[m],:],Fs,Tau,hopsize, win)
    v2[:,:,m] = V2[:,:,m]/np.max(np.absolute(V2[:,:,m]))
    n2[m]=np.max(np.abs(V2[:,:,m]))

solvenan2(eog_only._data)

V1,Ph1,tr1,f1 = spectrogramme(eog_only._data[0,:],Fs,Tau,hopsize, win)
v1 = V1/np.max(np.absolute(V1))
n1=np.max(np.abs(V1))

###################################################################################


plt.figure(figsize=(7, 15))
plt.subplot(2,1,1)
plot_spectrogram(v2[:,:,0], np.max(f2), np.max(eeg_only._times))
plt.title('EEG signal')
plt.subplot(2,1,2)
plot_spectrogram(v1, np.max(f1), np.max(eog_only._times))
plt.title('EOG signal')
pylab.savefig('1.png')
           
###################################################################################

w1, h1, w2, h2, q2=ntf_coupled2(v1,v2, ninit=2, niter=200)

K1=24
K2=100


Vhat1=rebuild (w1,h1)

Vhat2=rebuild (w2,h2,Q=q2)
Vclean_T=rebuild(w2[:,0:K1],h2[:,0:K1], q2[:,0:K1])
Vres_T=rebuild(w2[:,K1:w2.shape[1]],h2[:,K1:h2.shape[1]], q2[:,K1:q2.shape[1]])


I = np.round(hopsize * Nw)                       
M = 2 * (Vclean_T.shape[0]-1)                       
L = np.ceil( (Vclean_T.shape[1]-1)*I + M )

xclean=np.zeros([L,59])
xexg=np.zeros([L,59])

for m in tqdm(range(0,v2.shape[2])):
#for m in range(0,v2.shape[2]):
        
   V_clean=n2[m]*v2[:,:,m]*(np.power(Vclean_T[:,:, m],2)/(np.power(Vclean_T[:,:, m],2) + np.power(Vres_T[:,:,m],2)))
   V_artefacts=n1*v2[:,:,m]*(np.power(Vres_T[:,:, m],2)/(np.power(Vclean_T[:,:, m],2) + np.power(Vres_T[:,:,m],2)))
   
   xclean[:,m]=spectrogramme_inv(V_clean,Ph2[:,:,m],Fs,Tau,hopsize,win)[:,0]
   xexg[:,m]=spectrogramme_inv(V_artefacts,Ph1,Fs,Tau,hopsize,win)[:,0]

################################################################################
#Get the evoked potencial
eeg_only1 = raw.copy().pick_types(meg=False, eeg=True)
epochs_before = mne.Epochs(eeg_only1, events)
evoked_before = epochs_before.average()

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
for k in range(0,59):
    plt.plot(evoked_before.times,evoked_before.data[k,:], color='k', linewidth=0.5)
plt.title('Raw Data')

epochs_after_ICA = mne.Epochs(eeg_only_after_ica, events)
evoked_after_ICA = epochs_after_ICA.average()

plt.subplot(1,3,2)
for k in range(0,59):
    plt.plot(evoked_after_ICA.times,evoked_after_ICA.data[k,:], color='k', linewidth=0.5)
    plt.title('After ICA')

for m in range(0, len(ch)):
    eeg_only._data[ch[m],:]=xclean[0:eeg_only._data.shape[1],ch[m]]

epochs_after_NMF = mne.Epochs(eeg_only, events)
evoked_after_NMF = epochs_after_NMF.average()
plt.subplot(1,3,3)
for k in range(0,59):
    plt.plot(evoked_after_NMF.times,evoked_after_NMF.data[k,:], color='k', linewidth=0.5)
    plt.title('After NMF')
pylab.savefig('8.png')


##############################################################################
#Plot results

tmax=2500
tmax_a=2500

fig1=plt.figure(figsize=(12,6), dpi=80)
plt.subplot(2,1,1)
for k in range(0,59):
    plt.plot(eeg_only1._times[0:tmax],np.transpose (eeg_only1._data)[0:tmax,k],  color='b', linewidth=0.5)
    plt.plot(eeg_only1._times[0:tmax],xclean[0:tmax,k], color='r')
    plt.title('EEG signal')
plt.subplot(2,1,2)
plt.plot(eeg_only1._times[0:tmax], np.mean(eeg_only1._data[:,0:tmax], axis=0),  color='b', linewidth=0.5)
plt.plot(eeg_only1._times[0:tmax],np.mean(xclean[0:tmax,:], axis=1),  color='r')
plt.title('Average of EEG signal')

pylab.savefig('9.png')

plt.figure(figsize=(8,4), dpi=80)
plt.plot(eog_only._times[0:tmax_a],eog_only._data[0,0:tmax_a], color='b')
plt.plot(eog_only._times[0:tmax_a],xexg[0:tmax_a,0], color='r')
plt.title('Artefact signal')

##############################################################################
#Comparacao com ICA

N=eeg_only1._data.shape[0]

cm_ICA=np.zeros([1,N])
cm_NMF=np.zeros([1,N])
mi_ICA=np.zeros([1,N])
mi_NMF=np.zeros([1,N])

for k in range(0,N):
    temp=pearsonr(evoked_before.data[k,:],evoked_after_ICA.data[k,:])    
    cm_ICA[0,k]=temp[0]
    temp=pearsonr(evoked_before.data[k,:],evoked_after_NMF.data[k,:])    
    cm_NMF[0,k]=temp[0]


N = 1
NMFMeans = (np.mean(cm_NMF))#, np.mean(mi_NMF))
NMFStd = (np.std(cm_NMF))#, np.std(mi_NMF))

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

plt.figure(figsize=(6,4))
#fig, ax = plt.subplots()
rects1 = plt.bar(ind, NMFMeans, width, color='r', yerr=NMFStd)

ICAMeans = (np.mean(cm_ICA))#, np.mean(mi_ICA))
ICAStd = (np.std(cm_ICA))#, np.std(mi_ICA))
rects2 = plt.bar(ind + width, ICAMeans, width, color='y', yerr=ICAStd)

# add some text for labels, title and axes ticks
plt.ylabel('Scores')
plt.title('Perason correlation by method with the raw Data')
plt.xticks(ind + width)
#plt.xlabel(('CP', 'MI'))
plt.legend((rects1[0], rects2[0]), ('NMF', 'ICA'))

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.show()
pylab.savefig('10.png')

###################################################################################

