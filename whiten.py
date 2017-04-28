#!/usr/bin/env python

"""
    whiten.py
    
    Learns whitening featurization from one dataset, applies to another
"""

import sys
import h5py
import argparse
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def h5py2iterator(x, k=''):
    if isinstance(x, h5py._hl.group.Group):
        for k1 in x.keys():
            for k2,v in h5py2iterator(x[k1], k=k1):
                yield k2,v
    else:
        yield k,x.value

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-features', type=str, default='./oxford.h5')
    parser.add_argument('--whiten-features', type=str, default='./paris.h5')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--outpath', type=str, required=True)
    return parser.parse_args()

# --
# Run

if __name__ == "__main__":
    args = parse_args()
    
    # Learn whitening params from one dataset
    print >> sys.stderr, 'learning transform from %s' % args.whiten_features
    wdata = np.vstack([v.sum(axis=(1, 2)) for k,v in h5py2iterator(h5py.File(args.whiten_features))])
    nwdata = normalize(wdata)
    white_pca = PCA(n_components=args.dim, whiten=True, svd_solver='full').fit(nwdata)

    # Applying whitening to second dataset
    print >> sys.stderr, 'applying transform to %s' % args.index_features
    data = np.vstack([v.sum(axis=(1, 2)) for k,v in h5py2iterator(h5py.File(args.index_features))])
    ndata = normalize(data)
    white_ndata = white_pca.transform(ndata)
    nwhite_ndata = normalize(white_ndata)
    
    print >> sys.stderr, 'saving to %s' % args.outpath
    np.save(args.outpath, nwhite_ndata)