#!/bin/bash

import numpy as np
import matplotlib.pyplot as plt

"""
 ===================================================================================== PLOT ALL 3 MODELS =========================================================================================================================
"""


def plot_models(V_init, V_upd, V_tru, n_pml, extent, vmax, vmin, figsize, aspect = 'auto', cmap = 'viridis'):
    
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = figsize)
    plt.suptitle('MODELS', fontweight="bold")
    
    fig1 = axes[0].imshow(V_init[:-n_pml, n_pml:-n_pml], vmax = vmax, vmin = vmin, aspect = aspect, extent = extent, cmap = cmap)
    axes[0].set_title('Initial')
    axes[0].set_xlabel('X    (m)')
    axes[0].set_ylabel('Z    (m)')

    fig1 = axes[1].imshow(V_upd[:-n_pml, n_pml:-n_pml], vmax = vmax, vmin = vmin, aspect = aspect, extent = extent, cmap = cmap)
    axes[1].set_title('Inverted')
    axes[1].set_yticks([])
    axes[1].set_xlabel('X    (m)')

    fig1 = axes[2].imshow(V_tru[:-n_pml, n_pml:-n_pml], vmax = vmax, vmin = vmin, aspect = aspect, extent = extent, cmap = cmap)
    axes[2].set_title('True')
    axes[2].set_yticks([])
    axes[2].set_xlabel('X    (m)')
    
    fig.colorbar(fig1, ax = axes.ravel().tolist(), shrink = 1)
    
    plt.savefig('./results/models.png')
    

"""
 ============================================================================= PLOT HORIZONTAL & VERTICAL PROFILE ==============================================================================================================
"""


def plot_profile(V_init, V_upd, V_tru, n_pml, X, Z, xline, zline, figsize = (12, 6)):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
    
    ax[0].plot(X, V_init[zline, n_pml:-n_pml], color = 'red', ls = 'dotted', label = 'initial')
    ax[0].plot(X, V_upd[zline, n_pml:-n_pml], color = 'green', ls = 'solid', label = 'inverted')
    ax[0].plot(X, V_tru[zline, n_pml:-n_pml], color = 'blue', ls = 'dashed', label = 'true')
    ax[0].legend()
    ax[0].invert_yaxis()
    ax[0].set_title('Horizontal Profile')
    ax[0].set_ylabel('Vel    (m/s)')
    ax[0].set_xlabel('X    (m)')
    
    ax[1].plot(V_init[:-n_pml, xline], Z, color = 'red', ls = 'dotted', label = 'initial')
    ax[1].plot(V_upd[:-n_pml, xline], Z, color = 'green', ls = 'solid', label = 'inverted')
    ax[1].plot(V_tru[:-n_pml, xline], Z, color = 'blue', ls = 'dashed', label = 'true')
    ax[1].legend()
    ax[1].invert_yaxis()
    ax[1].set_title('Vertical Profile')
    ax[1].set_xlabel('Vel    (m/s)')
    ax[1].set_ylabel('Z    (m)')

    plt.savefig('./results/profile.png')

"""
 ============================================================================= PLOT INITIAL & FINAL PROFILE ==============================================================================================================
"""

def plot_error(data, data_init, data_fin, figsize = (10, 5)):
    
    fig, ax = plt.subplots(figsize = figsize)
    plt.suptitle('initial vs final error')
    ax.plot(data - data_init, ls = 'dotted', color = 'red', label = 'initial')
    ax.plot(data - data_fin, color = 'blue', label = 'final')
    ax.set_ylabel('error')
    ax.set_xlabel('data index')
    ax.legend()

    plt.savefig('./results/data_error.png')
