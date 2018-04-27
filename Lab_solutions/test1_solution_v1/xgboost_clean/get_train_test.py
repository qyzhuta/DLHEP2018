#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:37:48 2018

@author: qingyuzhu
"""

import pandas as pd



def gen_training_data(fname,names):
    
    
    
    df=pd.read_csv(fname,header=None,names=names)
    
    # Map string labels to 0/1
    df['Label'] = df['Label'].map({'b':0, 's':1})
    
    useful_inputs=['DER_mass_MMC','DER_mass_transverse_met_lep','DER_mass_vis',
                   'DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet',
                   'DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt',
                   'DER_pt_ratio_lep_tau','DER_met_phi_centrality','DER_lep_eta_centrality',
                   'Weight','Label']
    
    cols_new = df.columns.tolist()
    cols_new = [c for c in useful_inputs]
    
    df_new=df[cols_new]
    X_new = df_new[cols_new[:-2]].values
    labels = df_new['Label'].values
    weights = df_new['Weight'].values
    
    return df_new, X_new, labels, weights

titles=['EventId','DER_mass_MMC','DER_mass_transverse_met_lep','DER_mass_vis',
        'DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet',
        'DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt',
        'DER_pt_ratio_lep_tau','DER_met_phi_centrality','DER_lep_eta_centrality',
        'PRI_tau_pt','PRI_tau_eta','PRI_tau_phi','PRI_lep_pt','PRI_lep_eta',
        'PRI_lep_phi','PRI_met','PRI_met_phi','PRI_met_sumet',
        'PRI_jet_num','PRI_jet_leading_pt','PRI_jet_leading_eta',
        'PRI_jet_leading_phi','PRI_jet_subleading_pt','PRI_jet_subleading_eta',
        'PRI_jet_subleading_phi','PRI_jet_all_pt','Weight','Label']

fname='mytraining.csv'
df, X_new, labels, weights=gen_training_data(fname,titles)
df.to_csv('X_y',encoding='utf-8', index=False)