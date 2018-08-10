
# coding: utf-8

import json
import numpy as np

import celegans_dynome as c_dynome

def load_Json(filename):

    with open(filename) as content:

        content = json.load(content)

    return content

def construct_dyn_inputmat(t0, tf, dt, type, neuron_indices, normalized_amp = False, freq = False):

    timepoints = np.arange(t0, tf, dt)
    input_mat = np.zeros((len(timepoints) + 1, c_dynome.N))

    if type == 'sinusoidal':

        amp = normalized_amp / 2.

        for k in xrange(len(timepoints)):
            
            input_mat[k, neuron_indices] = amp * np.sin(freq * timepoints[k]) + amp

    return input_mat

def redblue(m):

    m1 = m * 0.5
    r = np.divide(np.arange(0, m1)[:, np.newaxis], np.max([m1-1,1]))
    g = r
    r = np.vstack([r, np.ones((int(m1), 1))])
    g = np.vstack([g, np.flip(g, 0)])
    b = np.flip(r, 0)
    x = np.linspace(0, 1, m)[:, np.newaxis]

    red = np.hstack([x, r, r])
    green = np.hstack([x, g, g])
    blue = np.hstack([x, b, b])

    red_tuple = tuple(map(tuple, red))
    green_tuple = tuple(map(tuple, green))
    blue_tuple = tuple(map(tuple, blue))

    cdict = {
    	'red': red_tuple,
        'green': green_tuple,
        'blue': blue_tuple
        }

    return cdict