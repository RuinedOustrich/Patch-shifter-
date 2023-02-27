#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 20:21:27 2023

@author: gleb
"""

import scipy.io.wavfile
import argparse
import numpy as np
from scipy.fft import fft, ifft
import os
import sys

#Defune hyperparameters
WINDOW_SIZE = 2048
HOP = int(2048 * 0.25)

class MyException(Exception):
    pass

def createFrame(frame, hop, out_hop):
    
    last_phase = np.zeros(frame.shape)
    phaseCumulative = 0
        
    k = 2*np.pi*np.arange(len(frame))/len(frame)
    magFrame = np.abs(frame)
    phase = np.angle(frame)
    deltaPhi = phase - last_phase
    last_phase = np.copy(phase)
    deltaPhiPrime =  deltaPhi - hop * k
    deltaPhiPrimeMod = np.mod(deltaPhiPrime + np.pi, 2*np.pi) - np.pi
    trueFreq = k + deltaPhiPrimeMod/hop
    phaseCumulative += out_hop * trueFreq
        
    return magFrame * np.exp(phaseCumulative*1j)

def returnFrame(frames, hop, out_hop):
    for frame in frames:
        yield createFrame(frame, hop, out_hop)
    
def PitchShifter(args = {}):
    
    #Define stretch or compress audio 
    if args.ratio == 1:
        rate = 2
    elif args.ratio == 0:
        rate = 0.5
    else:
        raise MyException("Ratio required only 0 or 1")

    #Read .wav file
    FS, audio = scipy.io.wavfile.read(args.input)
    
    if len(audio.shape) > 1:
        audio = audio.mean(axis = 1)
    
    OUTPUT_HOP = int(HOP * rate)

    hunning_window = np.hanning(WINDOW_SIZE)
    
    #Create frames after fft
    frames_fft = []
    for i in range(0, len(audio) - WINDOW_SIZE, HOP):
        frame = fft(hunning_window*audio[i:i+WINDOW_SIZE])/np.sqrt(((float(WINDOW_SIZE)/float(HOP))/2.0))
        frames_fft.append(frame)
    
    frames = [frame for frame in returnFrame(frames_fft, HOP, OUTPUT_HOP)]
    
    #Get result audio
    frames_ifft = np.zeros(len(frames)*OUTPUT_HOP)
    
    for i, j in enumerate(range(0, len(frames_ifft) - WINDOW_SIZE, OUTPUT_HOP)):
        frames_ifft[j:j+WINDOW_SIZE] += hunning_window * np.real(ifft(frames[i]))
    
    #Save result audio
    scipy.io.wavfile.write(args.output, FS, np.asarray(frames_ifft, dtype=np.int16))
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('ratio', type=(int)) 

    args = parser.parse_args()
    PitchShifter(args)

if __name__ == "__main__":
    main()
    
    
            