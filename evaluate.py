# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import os
import sys
import glob
import argparse
import numpy as np
from functools import partial
from tempfile import NamedTemporaryFile

from crow import run_feature_processing_pipeline, apply_crow_aggregation, apply_ucrow_aggregation, normalize

def get_nn(x, data, k=None):
    if k is None:
        k = len(data)

    dists = ((x - data) ** 2).sum(axis=1)
    idx = np.argsort(dists) 
    dists = dists[idx]

    return idx[:k], dists[:k]


def simple_query_expansion(Q, data, inds, top_k=10):
    Q += data[inds[:top_k],:].sum(axis=0)
    return normalize(Q)


def load_features(feature_dir, verbose=True):
    if type(feature_dir) == str:
        feature_dir = [feature_dir]

    for directory in feature_dir:
        for i, f in enumerate(os.listdir(directory)):
            name = os.path.splitext(f)[0]

            # Print progress
            if verbose and not i % 100:
                sys.stdout.write('\rProcessing file %i' % i)
                sys.stdout.flush()

            X = np.load(os.path.join(directory, f))

            yield X, name

    sys.stdout.write('\n')
    sys.stdout.flush()


def load_and_aggregate_features(feature_dir, agg_fn):
    print 'Loading features %s ...' % str(feature_dir)
    features = []
    names = []
    for X, name in load_features(feature_dir):
        names.append(name)
        X = agg_fn(X)
        features.append(X)

    return features, names


def get_ap(inds, dists, query_name, index_names, groundtruth_dir, ranked_dir=None):
    if ranked_dir is not None:
        # Create dir for ranked results if needed
        if not os.path.exists(ranked_dir):
            os.makedirs(ranked_dir)
        rank_file = os.path.join(ranked_dir, '%s.txt' % query_name)
        f = open(rank_file, 'w')
    else:
        f = NamedTemporaryFile(delete=False)
        rank_file = f.name
    
    f.writelines([index_names[i] + '\n' for i in inds])
    f.close()
    
    groundtruth_prefix = os.path.join(groundtruth_dir, query_name)
    cmd = './compute_ap %s %s' % (groundtruth_prefix, rank_file)
    ap = os.popen(cmd).read()
    
    # Delete temp file
    if ranked_dir is None:
        os.remove(rank_file)
    
    return float(ap.strip())


def fit_whitening(whiten_features, agg_fn, d):
    # Load features for fitting whitening
    data, _ = load_and_aggregate_features(whiten_features, agg_fn)
    print 'Fitting PCA/whitening wth d=%d on %s ...' % (d, whiten_features)
    _, whiten_params = run_feature_processing_pipeline(data, d=d)
    return whiten_params


def run_eval(queries_dir, groundtruth_dir, index_features, whiten_params, out_dir, agg_fn, qe_fn=None):
    data, image_names = load_and_aggregate_features(index_features, agg_fn)
    data, _ = run_feature_processing_pipeline(np.vstack(data), params=whiten_params)
    
    image_names = np.array(image_names)
    
    # Iterate queries, process them, rank results, and evaluate mAP
    aps = []
    for Q, query_name in load_features(queries_dir):
        Q = agg_fn(Q)
        
        # Normalize and PCA to final feature
        Q, _ = run_feature_processing_pipeline(Q, params=whiten_params)
        
        inds, dists = get_nn(Q, data)
        
        # perform query_expansion
        if qe_fn is not None:
            Q = qe_fn(Q, data, inds)
            inds, dists = get_nn(Q, data)
        
        open('%s/%s' % (out_dir, query_name), 'w').write('\n'.join(image_names[inds]))
    #     ap = get_ap(inds, dists, query_name, image_names, groundtruth_dir, out_dir)
    #     aps.append(ap)
    
    # return np.array(aps).mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wt', dest='weighting', type=str, default='crow', help='weighting to apply for feature aggregation')
    parser.add_argument('--index_features', dest='index_features', type=str, default='oxford/pool5', help='directory containing raw features to index')
    parser.add_argument('--whiten_features', dest='whiten_features', type=str, default='paris/pool5', help='directory containing raw features to fit whitening')
    
    parser.add_argument('--queries', dest='queries', type=str, default='oxford/pool5_queries/', help='directory containing image files')
    parser.add_argument('--groundtruth', dest='groundtruth', type=str, default='oxford/groundtruth/', help='directory containing groundtruth files')
    parser.add_argument('--d', dest='d', type=int, default=128, help='dimension of final feature')
    parser.add_argument('--out', dest='out', type=str, default=None, help='optional path to save ranked output')
    parser.add_argument('--qe', dest='qe', type=int, default=0, help='perform query expansion with this many top results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Which aggregation function?
    agg_fn = apply_crow_aggregation if args.weighting == 'crow' else apply_ucrow_aggregation
    
    # Use query expansion?
    qe_fn = partial(simple_query_expansion, top_k=args.qe) if args.qe > 0 else None
        
    # compute whitening params
    whitening_params = fit_whitening(args.whiten_features, agg_fn, args.d)

    # compute aggregated features and run the evaluation
    mAP = run_eval(args.queries, args.groundtruth, args.index_features, whitening_params, args.out, agg_fn, qe_fn)
    
    print 'mAP: %f' % mAP

    exit(0)

