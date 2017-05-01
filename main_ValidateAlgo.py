# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:40:28 2016

@author: renatobottermaiolopesrodrigues
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from run_coupled import *
from spectrogramacerto import *
from sklearn.decomposition import FastICA, PCA

plt.close('all')
data_eeg=np.load('data_tmp_eeg.npy')
data_eog=np.load('data_tmp_eog.npy')
times=np.load('time_tmp.npy')

np.random.seed(0)  # set seed for reproducible results
n_samples = data_eeg.size
#n_samples=2000;
#times = np.linspace(0, 8, n_samples)



s1 = data_eeg  # Signal 1 : data eeg
s2 = data_eog  # Signal 1 : data eog

#s1 = np.sin(10 * times)  # Signal 1 : sinusoidal signal
#s2 = np.sign(np.sin(3 * times))  # Signal 2 : square signal

S = np.c_[s1, s2]
#S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([5, -5])  # Mixing matrix
X = np.dot(S, A)  # Generate observations


Fs=1/(times[1]-times[0])
Tau = 0.7
hopsize=1/8
win='h'
Nw=np.round(Tau * Fs)
R = np.round(hopsize * Nw) 
N = X.shape[0]
Nt = np.array([np.fix((N-Nw)/R)])



V2, Ph2, tr2, f2 = spectrogramme(X[:],Fs,Tau,hopsize, win)
V2 = np.expand_dims(V2,axis=2)
Ph2 = np.expand_dims(Ph2,axis=2)

v2 = V2/np.max(np.absolute(V2))
n2=np.max(np.abs(V2))

V1,Ph1,tr1,f1 = spectrogramme(A[1]*s2[:],Fs,Tau,hopsize, win)
V1=np.expand_dims(V1,axis=2)
v1 = V1/np.max(np.absolute(V1))
Ph1=np.expand_dims(Ph1,axis=2)
n1=np.max(np.abs(V1))


figure=plt.figure(figsize=(7, 15))
plt.subplot(2,1,1)
plot_spectrogram(v2[:,:,0], np.max(f2)/10, np.max(times[0:n_samples]))
plt.title('Observations')
plt.subplot(2,1,2)
plot_spectrogram(v1[:,:,0], np.max(f1)/10, np.max(times[0:n_samples]))
plt.title('s2')
#pylab.savefig('spectrogram.png')



w1, h1,q1, w2, h2, q2=ntf_coupled2(v1,v2, ninit=2, niter=20)

Vhat1=rebuild (w1,q1,h1)
Vhat2=rebuild (w2,q2,h2)

K1=34
K2=100

Vclean_T=rebuild(w2[:,0:K1],q2[:,0:K1],h2[:,0:K1])
Vres_T=rebuild(w2[:,K1:w2.shape[1]],q2[:,K1:q2.shape[1]],h2[:,K1:h2.shape[1]])

     
V_clean=n2*v2[:,:,0]*(np.power(Vclean_T[:,:,0],2)/(np.power(Vclean_T[:,:,0],2) + np.power(Vres_T[:,:,0],2)))
V_artefacts=n1*v2[:,:,0]*(np.power(Vres_T[:,:,0],2)/(np.power(Vclean_T[:,:,0],2) + np.power(Vres_T[:,:,0],2)))
   
xclean=spectrogramme_inv(V_clean,Ph2[:,:,0],Fs,Tau,hopsize,win)[:,0]
xexg=spectrogramme_inv(V_artefacts,Ph1[:,:,0],Fs,Tau,hopsize,win)[:,0]
   
###################################################################################

n_samples_plot=5000
#n_samples_plot=n_samples


figure=plt.figure(figsize=(6, 15))
plt.subplot(5,1,1)
plt.plot(times[0:n_samples_plot], X[0:n_samples_plot],  color='b')
plt.title('Observations')
plt.subplot(5,1,2)
plt.plot(times[0:n_samples_plot],A[1]*s2[0:n_samples_plot])

plt.title('Original Signal s2')
plt.subplot(5,1,3)
plt.plot(times[0:n_samples_plot],xclean[0:n_samples_plot])
plt.title('s2')
plt.subplot(5,1,4)
plt.plot(times[0:n_samples_plot],A[0]*s1[0:n_samples_plot])
plt.title('Original Signal s1')
plt.subplot(5,1,5)
plt.plot(times[0:n_samples_plot],xexg[0:n_samples_plot])
plt.title('s1')


###################################################################################
