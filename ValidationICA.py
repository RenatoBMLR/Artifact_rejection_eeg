# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:55:17 2016

@author: renatobottermaiolopesrodrigues
"""
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import numpy as np
import matplotlib.pyplot as plt
import pylab

 

def ica_method(raw, picks, plot='n', save ='n'):
    
    
    ###############################################################################
    # 1) Fit ICA model using the FastICA algorithm
    # Other available choices are `infomax` or `extended-infomax`
    # We pass a float value between 0 and 1 to select n_components based on the
    # percentage of variance explained by the PCA components.
    
    ica = ICA(n_components=0.95, method='fastica')
    
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')
    
    ica.fit(raw, picks=picks, decim=3)
    
    # maximum number of components to reject
    n_max_eog =  1  # here we don't expect horizontal EOG components
    
    ###############################################################################
    # 2) identify bad components by analyzing latent sources.
    
    
    # detect EOG by correlation
    
    eog_inds, scores = ica.find_bads_eog(raw, threshold=2.5)    
    show_picks = np.abs(scores).argsort()[::-1][:5]
    eog_inds = eog_inds[:n_max_eog]
    ica.exclude += eog_inds
    
    ###############################################################################
    # 3) Assess component selection and unmixing quality
    eog_evoked = create_eog_epochs(raw, tmin=-.5, tmax=.5, picks=picks).average()
    
    if plot=='y':
        
        title = 'Sources related to %s artifacts (red)'
        ica.plot_scores(scores, exclude=eog_inds, title=title % 'eog', labels='eog')
        if save=='y':
            pylab.savefig('2.png')

        ica.plot_sources(raw, show_picks, exclude=eog_inds, title=title % 'eog')
        if save=='y':
            pylab.savefig('3.png')

        ica.plot_components(eog_inds, title=title % 'eog', colorbar=True)
        if save=='y':

            pylab.savefig('4.png')

        ica.plot_overlay(raw)  # EOG artifacts remain
        if save=='y':

            pylab.savefig('5.png')

        ica.plot_sources(eog_evoked, exclude=eog_inds)  # plot EOG sources + selection
        if save=='y':

            pylab.savefig('6.png')

        ica.plot_overlay(eog_evoked, exclude=eog_inds)  # plot EOG cleaning
        if save=='y':
            pylab.savefig('7.png')


    ica.apply(raw, exclude=eog_inds)    
    eeg_only_after=raw.pick_types(meg=False, eeg=True)    
    

    return eeg_only_after
    