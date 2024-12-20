#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:14:14 2024

@author: tonyshara
"""
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import numpy as np

def simple_plot_summary(_df,show=True):
    fig, axes = plt.subplots(8,3, figsize = (15,25), constrained_layout=True, sharex=True)
    for j,c in enumerate(_df.columns[2:-1]):
        axes[j//3][j%3].plot(_df['time'], _df[c])
        axes[j//3][j%3].set_title(c)
    plt.savefig('./figs/plot_summary')
    if show:
        plt.show()
    else:
        plt.close()

def comparison_plot_summary(_samples, _labels,show=True):
    s1,s2,s3 = _samples
    l1,l2,l3 = _labels
    features = s1.columns[4:-1]
    fig, axes = plt.subplots(5,3, figsize = (15,25), constrained_layout=True, sharex=True)
    fig.suptitle('Error Summary', fontsize=30)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for j,c in enumerate(features):
        axes[j // 3, j % 3].plot(s1['time'], s1[c]-s3[c], c = 'lightblue')
        axes[j // 3, j % 3].set_title(c, fontsize=20)
    plt.savefig('./figs/error_summary')
    if show:
        plt.show()
    else:
        plt.close()
     
    fig, axes = plt.subplots(5,3, figsize = (15,25), constrained_layout=True, sharex=True)
    fig.suptitle('Preprocessing Comparison', fontsize=30)

    for j,c in enumerate(features):
        axes[j // 3, j % 3].plot(s1['time'], s1[c], c = 'lightblue', label = l1)
        axes[j // 3, j % 3].plot(s2['time'], s2[c], c = 'salmon', label = l2)
        axes[j // 3, j % 3].plot(s3['time'], s3[c], c = 'orange', label = l3)
        axes[j // 3, j % 3].set_title(c, fontsize=18)
        axes[j // 3, j % 3].legend(fontsize=12)
    plt.savefig('./figs/comparison_summary')
    if show:
        plt.show()
    else:
        plt.close()   
     
    c = 'Physical Fan Speed'
    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.set_title(c)
    ax.plot(s1['time'], s1[c], c = 'lightblue', label = l1)
    ax.plot(s2['time'], s2[c], c = 'salmon'   , label = l2)
    ax.plot(s3['time'], s3[c], c = 'orange'   , label = l3)
    ax.set_title(c)
    plt.savefig(f'./figs/comparison_summary_{c}')
    if show:
        plt.show()
    else:
        plt.close()

    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.set_title(c)
    ax.plot(s1['time'], s1[c]-s3[c], c = 'lightblue')
    plt.savefig(f'./figs/error_summary_{c}')
    if show:
        plt.show()
    else:
        plt.close()

def pre_processed_plots(_samples, _labels,show=True):
    s1,s2,s3 = _samples
    l1,l2,l3 = _labels
    features = s1.columns[4:-1]
     
    fig, axes = plt.subplots(5,3, figsize = (12,10), constrained_layout=True, sharex=True)
    fig.suptitle('Preprocessing Comparison', fontsize=10)
    for j,c in enumerate(features):
        axes[j // 3, j % 3].plot(s1['time'], s1[c], c = 'lightblue', label = l1)
        axes[j // 3, j % 3].plot(s2['time'], s2[c], c = 'salmon', label = l2)
        axes[j // 3, j % 3].plot(s3['time'], s3[c], c = 'orange', label = l3)
        axes[j // 3, j % 3].set_title(c, fontsize=8)
        axes[j // 3, j % 3].legend(fontsize=9)
    
    if show:
        plt.show()
    else:
        plt.close()   


def plot_fft(signal, name='', show=False):
    # Compute the Fourier transform & frequencies
    fft_signal = fft(signal)
    freq = fftfreq(len(signal), 0.001)
    
    # Plot the magnitude spectrum
    plt.plot(freq, np.abs(fft_signal))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Fourier Spectrum " + name)
    # plt.savefig(f'./figs/fft_{name}'.replace('/','_'))
    plt.close()
    
    plt.title('Frequency Response')
    plt.magnitude_spectrum(signal)
    # plt.savefig(f'./figs/fft_{name}'.replace('/','_'))
    if show:
        plt.show()
    else:
        plt.close()


def plot_lpf_summary(num,den,show=False):
    # Compute the impulse response
    t, y = signal.impulse((num, den))

    # Compute the frequency response
    w, h = signal.freqz(num, den)
    fc = w[np.argmin(np.abs(np.abs(h) - 0.5*np.sqrt(2)))]

    # Plot the frequency response (magnitude and phase)
    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.plot(w, np.abs(h))
    ax.set_title('Low Pass Filter Frequency Response')
    ax.set_xlabel('Frequency (rad/sample)')
    ax.set_ylabel('Magnitude')
    ax.grid(True, which="both", ls="-")
    ax.plot(fc, 0.5*np.sqrt(2), 'ko')
    ax.axvline(fc, color='k')
    ax.set_xscale('log')
    ax.text(0.5, 0.8, f'fc = {fc:.4f} Hz', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='black', pad=10))
    plt.savefig('./figs/FrequencyResponse')
    plt.show()


colors = ['lightblue', 'salmon', 'orange']
def plot_signal(_samples, _name, _labels):
    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.set_title(_name)
    ax.set_xlabel('cycles')
    for i in range(len(_samples)):
        color = 'blue' if len(_samples) == 1 else colors[i]
        ax.plot(_samples[i]['time'], _samples[i][_name], c = color, label = _labels[i])

    if len(_samples) == 1:
        ax.set_title(f'{_name} - {_labels[0]}')
        plt.savefig(f'./figs/signal_{_name}_{_labels}')
    else:
        ax.set_title(f'{_name} - comparison')
        ax.legend()
        plt.savefig(f'./figs/signal_{_name}_comparison')
    plt.show()

def comparison_plot_summary_simple(_samples, _labels):
    features = _samples[0].columns[4:-1]
    fig, axes = plt.subplots(5,3, figsize = (15,25), constrained_layout=True, sharex=True)
    fig.suptitle('Raw Data Comparison', fontsize=30)
    for j,c in enumerate(features):
        axes[j // 3, j % 3].plot(_samples[0]['time'], _samples[0][c], c = 'blue')
        axes[j // 3, j % 3].set_title(c, fontsize=20)
    plt.savefig('./figs/raw_data_signals_comp')
    plt.show()

def plot_err_summary(_minmax, _lpf):
    c = 'Physical Fan Speed'
    
    # Plot Minmax Signal
    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.set_title(f'{c} - minmax')
    ax.set_xlabel('cycles')    
    ax.plot(_minmax['time'], _minmax[c], c = 'blue')
    plt.savefig('./figs/signal_features_minmax')
    plt.show()
    
    # Plot exponential Signal
    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.set_title(f'{c} - LPF (Exponential)')
    ax.set_xlabel('cycles')    
    ax.plot(_lpf['time'], _lpf[c], c = 'blue')
    plt.savefig('./figs/signal_features_exponential')
    plt.show()
    
    # Plot HF noise Signal
    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.set_title(f'{c} - High Frequency Noise')
    ax.set_xlabel('cycles')    
    ax.plot(_lpf['time'], np.abs(_lpf[c]-_minmax[c]), c = 'blue')
    plt.savefig('./figs/signal_features_hf_error')
    plt.show()
    
    
    
    
    
    