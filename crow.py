# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import scipy
import numpy as np
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA


def compute_crow_spatial_weight(X, a=2, b=2):
    S = X.sum(axis=0)
    z = (S**a).sum()**(1./a)
    return (S / z)**(1./b) if b != 1 else (S / z)


def compute_crow_channel_weight(X):
    K, w, h = X.shape
    area = float(w * h)
    nonzeros = np.zeros(K, dtype=np.float32)
    for i, x in enumerate(X):
        nonzeros[i] = np.count_nonzero(x) / area

    nzsum = nonzeros.sum()
    for i, d in enumerate(nonzeros):
        nonzeros[i] = np.log(nzsum / d) if d > 0. else 0.

    return nonzeros


def apply_crow_aggregation(X):
    S = compute_crow_spatial_weight(X)
    C = compute_crow_channel_weight(X)
    X = X * S
    X = X.sum(axis=(1, 2))
    return X * C


def apply_ucrow_aggregation(X):
    return X.sum(axis=(1, 2))


def normalize(x, copy=False):
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
    else:
        return sknormalize(x, copy=copy)


def run_feature_processing_pipeline(features, d=128, whiten=True, copy=False, params=None):
    # Normalize
    features = normalize(features, copy=copy)

    # Whiten and reduce dimension
    if params:
        pca = params['pca']
        features = pca.transform(features)
    else:
        pca = PCA(n_components=d, whiten=whiten, copy=copy)
        features = pca.fit_transform(features)
        params = { 'pca': pca }

    # Normalize
    features = normalize(features, copy=copy)

    return features, params


def save_spatial_weights_as_jpg(S, path='.', filename='crow_sw', size=None):
    img = scipy.misc.toimage(S)
    if size is not None:
        img = img.resize(size)

    img.save(os.path.join(path, '%s.jpg' % str(filename)))
