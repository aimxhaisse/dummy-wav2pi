#!/usr/bin/env python3
#
# Failed attempt to extract main sines from a wav file and retranscript
# it to Sonic Pi.
#
# This kind-of work (i.e: we recognize the sound), but it's far from usable

import sys
import math
import numpy as np

from scipy.io import wavfile
from collections import OrderedDict

OCTAVE_FIRST = 3
OCTAVE_LAST = 7
TOP_NOTES = 7
NB_POINTS = 8
BASE_NOTES_HZ = [
    ('C', 16.352),
    ('Cs', 17.324),
    ('D', 18.354),
    ('Ds', 19.445),
    ('E', 20.602),
    ('F', 21.827),
    ('Fs', 23.125),
    ('G', 24.500),
    ('As', 25.957),
    ('A', 27.500),
    ('As', 29.135),
    ('B', 30.868),
]


def dump(analyzed):
    for k, v in analyzed.items():
        print('{0} = {1}'.format(k, v))


def analyze(input):
    """ I analyze the input audio file.
    """
    analyzed = {}
    rate, snd = wavfile.read(input)
    snd = snd / (2. ** 15)
    entire_chan = snd[:, 0]

    analyzed['duration'] = snd.shape[0] / rate

    entire_nb_samples = len(entire_chan)
    nb_samples = entire_nb_samples / NB_POINTS

    # compute notes we want to watch
    notes = []
    for i in range(OCTAVE_FIRST, OCTAVE_LAST + 1):
        for label, freq in BASE_NOTES_HZ:
            notes.append(('{0}{1}'.format(label, i), i * freq))

    # compute ranges we want to analyze
    franges = []
    prev = None
    for label, freq in notes:
        if prev:
            half = abs((freq - prev) / 2.0)
            franges.append((label, (freq - half, freq + half)))
        prev = freq

    # we split the sample into N chunks which we'll analyze (N is the
    # number of points we want.)
    
    point = 0
    while point < entire_nb_samples:
        chan = entire_chan[point:(point + nb_samples)]

        transform = np.fft.fft(chan)
        freqs = np.fft.fftfreq(int(entire_nb_samples))

        # let's fetch amplitudes, aggregated by note
        namps = {}
        for amp, freq in zip(transform, freqs):
            if not amp:
                continue
            freq = np.abs(freq * rate)
            for frange in franges:
                start, end = frange[1]
                if freq >= start and freq <= end:
                    label = frange[0]
                    if label not in namps:
                        namps[label] = []
                    namps[label].append(np.abs(amp))

        # now compute the sum for each note
        npower = OrderedDict()
        for label, amp in namps.items():
            # here we divide by the len of the note frequency range,
            # since this sum is per sample (we want low frequencies to
            # be as amplified as high frequencies).
            size = None
            for note, frange in franges:
                if label == note:
                    size = (frange[1] - frange[0]) * 1000
                    break
            npower[label] = sum(amp) / size
        snpower = sorted(npower.items(), key=lambda tup: tup[1], reverse=True)

        for label, value in snpower[0:TOP_NOTES]:
            print("synth :sine, note: :{0}, sustain_level: {1}, sustain: 1, release: 0".format(label, value))
        print("sleep 1")
        print("")
            
        point += nb_samples

if __name__ == '__main__':
    if len(sys.argv) == 2:
        analyze(sys.argv[1])
    else:
        print('usage: {} input.wav'.format(sys.argv[0]))
