#!/usr/bin/env python

"""
    whiten.py
"""

import h5py
import numpy as np
from sklearn.decomposition import PCA

def h5py2iterator(x, k=''):
    if isinstance(x, h5py._hl.group.Group):
        for k1 in x.keys():
            for k2,v in h5py2iterator(x[k1], k=k1):
                yield k2,v
    else:
        yield k,x.value

# --
# Params

d = 128
whiten_features = './paris.h5'
index_features = './oxford.h5'

# --
# Run

# Learn whitening params from one dataset
wdata = np.vstack([v.sum(axis=(1, 2)) for k,v in h5py2iterator(h5py.File(whiten_features))])
nwdata = normalize(wdata)
white_pca = PCA(n_components=d, whiten=True).fit(nwdata)

# Applying whitening to second dataset
data = np.vstack([v.sum(axis=(1, 2)) for k,v in h5py2iterator(h5py.File(index_features))])
ndata = normalize(data)
white_ndata = white_pca.transform(ndata)
nwhite_ndata = normalize(white_ndata)

