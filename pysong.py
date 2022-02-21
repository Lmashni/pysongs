#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:03:08 2020

@author: lyth
"""

import time
import numpy as np
import matplotlib.pyplot as plt
#import sounddevice as sd
from scipy.io.wavfile import write
from scipy import signal

def gaus(x,m,s):
    return np.exp(-((x-m)**2)/s)        # incase u need a gaus  not used

def prob(x,scl,P,w):
    pp=np.zeros(len(x))
    for sc,p in zip(scl,P):
        pp += gaus(x,sc,w)*p       # not used
    return pp

def tembre(sc,x,tet,w,w1):
    return np.sin(w*x**(np.sin((2**(sc/tet))*w1*w*x)))*0.1    #if sin waves get boring

def strech(x,um):
    x2 = np.zeros(len(x))
    for i in range(um):
        x2[i::um] =x[:int(len(x)/um)]       # given sequence of notes this streches them (fugemässig)
    return x2

def spiegel(sc,sp):
    return np.ones(len(sc))*sp -(sc-sp) 


def fuge_that_shit(sc1,n_voices=2,strech_by=2,shift=0):
    
    #sc1 = np.array([0,0,0,0,3,3,3,3,0,0,0,0,3,3,3,3,0,1,2,3,2,3,4,5,0,0,0,0,3,3,3,3,5,4,3,2,1,0,0,0,0]*50)*4
    sc = np.zeros((n_voices,len(sc1)))
    sc[0] = sc1
    for i in range(1,n_voices):
        sc[i] = np.roll(strech(sc[i-1],strech_by),shift)
        
    
    return sc
    
def numpy_to_voices(sc,file,wr_wav = False,tet = 7):
    f = 10000000       # Number of frames

    x = np.linspace(0,1000,f)   # x-axis for make benefit y-axis

    w= 100 *np.ones(len(x))# + 0.02*np.sin(1*x)            # frequency units depend on selected frame rate
    y= np.zeros(len(x))     # signal to become sound

    tet = 7 # tuning system    
    n_notes=sc.shape[1]
    n_voices =sc.shape[0]
    voices = 2**np.arange(0,n_voices)
    vol = 1/(np.arange(1,n_voices+1))
    #vol = 5-np.arange(1,n_voices+1)
    #vol = vol + vol.min() +0.2
    vol = vol/vol.sum()
   

    for j in range(n_voices):
        mm=int(f / n_notes)
        for i in range(n_notes):
 
            if sc[j,i]!=1j:      
                y[i*mm:(i+1)*mm] += vol[j]*np.sin(voices[j]*w[:mm]*x[i*mm:(i+1)*mm]*(2**(sc[j,i]/tet)))
    
    if wr_wav == True:            
        sos = signal.butter(10,20 ,fs=1000, output='sos')#,btype='highpass')

        yf = signal.sosfilt(sos, y)#makes wav file given frame rate out of signal for stereo data =Y.T
        write(file,40000,data=yf)
        
    return y 

if __name__ == '__main__':
    
    file = './sounds/go.way'
    n_notes=1000
    n_voices =7
    #sc=np.random.randint(0,14,(n_voices,n_notes)) 
    
    sc1 = np.array([0,0,0,0,3,3,3,3,0,0,0,0,3,3,3,3,0,1,2,3,2,3,4,5,0,0,0,0,3,3,3,3,5,4,3,2,1,0,0,0,0,3,3,3,3,0,0,0,0,3,3,3,3,0,1,2,3,2,3,4,5,0,0]*3*3*3)*8
    sc =fuge_that_shit(sc1,5,3)
    y = numpy_to_voices(sc,file,wr_wav=1)

