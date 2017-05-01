# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:52:13 2016

@author: renatobottermaiolopesrodrigues
"""
import numpy as np
from scipy import signal
from scipy.fftpack import fft,ifft, fftshift
import matplotlib.pyplot as plt


def ola(w, hop, Nb=10):
    
    w = np.transpose(w)
    N = len(w)
    output = np.zeros([1,(Nb-1)*hop+N])

    for k in range(0,Nb):
        deb = (k)*hop;
        fin = deb+N
        output[0,deb:fin] = output[0,deb:fin]+w
    return output
    
def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return m_i    

def spectrogramme(s,Fs,Tau,hopsize, win):
    
    N = s.shape[0]
    Nw=np.round(Tau * Fs)
    R = np.round(hopsize * Nw) 
    Nt = np.array([np.fix((N-Nw)/R)])
    I = np.round(hopsize * Nw)                   


    if win=='h':
        w = signal.hamming(Nw, sym=True)
    elif win=='b':
        w = signal.blackman(Nw, sym=True)
    else:
        w = np.ones(Nw)
        
    
    aux_window = ola(np.power(w,2),R,int (Nt[0]))
    window=np.sqrt((1/np.max(aux_window))*np.power(w,2))
    
    Spectro = np.zeros([Nw, int (Nt[0])])            
    Phase   = np.zeros([Nw, int (Nt[0])])
    
    for r in range(0,int (Nt[0])):
        
        deb = (r)*R 
        fin = deb + Nw
        tx = s[deb:fin]*window
        X = fft(tx)
        Spectro[:,r] = np.abs(X[0:len(tx)])
        Phase[:,r] = np.angle(X[0:len(tx)])
        
    tr=np.arange(0,Nt-1,1)*(I/Fs)+(N/2)/Fs
    f=np.arange(0,len(tx)-1,1)*(Fs/2)/len(tx)
    
    return (Spectro,Phase,tr,f)
    
def spectrogramme_inv(Spectro,Phase,Fs,Tau,hopsize,win):
    
    X = Spectro * np.exp(1j*Phase)
    [Nf,Nt] = X.shape
    Nw=np.round(Tau * Fs)
    R = np.round(hopsize * Nw) 

    if win=='h':
        w = signal.hamming(Nw, sym=True)
    elif win=='b':
        w = signal.blackman(Nw, sym=True)
    else:
        w = np.ones(Nw)
    
    I = np.round(hopsize * Nw)                       
    M = 2 * (Nf-1)                       
    L = np.ceil( (Nt-1)*I + M )
    
    y = np.zeros([L,1])
         
    aux_window = ola(np.power(w,2),R, Nt)
    window=np.sqrt((1/np.max(aux_window))*np.power(w,2))
    
    y_aux=np.zeros([Nw,1])

    for r in range(0, Nt):
        
        deb = (r)*R 
        fin = deb + Nw   
        Xtilde_inv = ifft(X[:,r])
        y_aux[:,0] = Xtilde_inv*window
        y[deb:fin,0]=y[deb:fin,0]+y_aux[:,0]
        
    return y
    
def plot_spectrogram(v2, maxf, maxt, figure=plt.figure(), log='y' ,colormap="jet"):
    
    if log=='y':
        figure=plt.imshow( np.log(np.abs((v2))), aspect="auto", cmap=colormap)
    else:
        figure=plt.imshow( v2, aspect="auto", cmap=colormap)
    plt.colorbar()
    plt.axis([0, maxt, 0, maxf])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    
    return 
   
   

    
