"""
    extract_features.py
    
    python extract_features.py --images tmp/* --out tmp/pool5
"""

import sys
sys.path.append('/home/bjohnson/projects/py-faster-rcnn/caffe/python')

import os
import caffe
import scipy
import argparse
import numpy as np
from PIL import Image

def load_img(path):
    try:
        img = Image.open(path)
        rgb_img = Image.new("RGB", img.size)
        rgb_img.paste(img)
        rgb_img = rgb_img.resize((586, 586), Image.ANTIALIAS)
        return rgb_img
    except:
        return None


def format_img_for_vgg(img):
    d = np.array(img, dtype=np.float32)
    d = d[:,:,::-1]
    d -= np.array((104.00698793,116.66876762,122.67891434))
    return d.transpose((2,0,1))


def extract_raw_features(net, layer, d):
    net.blobs['data'].reshape(1, *d.shape)
    net.blobs['data'].data[...] = d
    net.forward()
    return net.blobs[layer].data[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', dest='images', type=str, nargs='+', required=True, help='glob pattern to image data')
    parser.add_argument('--layer', dest='layer', type=str, default='pool5', help='model layer to extract')
    parser.add_argument('--prototxt', dest='prototxt', type=str, default='vgg/VGG_ILSVRC_16_pool5.prototxt', help='path to prototxt')
    parser.add_argument('--caffemodel', dest='caffemodel', type=str, default='vgg/VGG_ILSVRC_16_layers.caffemodel', help='path to model params')
    parser.add_argument('--out', dest='out', type=str, default='', help='path to save output')
    return parser.parse_args()

if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(1)
    
    args = parse_args()
    
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    for path in args.images:
        img = load_img(path)
        
        if img is None:
            print >> sys.stderr, "Error @ %s" % path
            continue
        
        d = format_img_for_vgg(img)
        X = extract_raw_features(net, args.layer, d)
        
        filename = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(args.out, filename), X)
