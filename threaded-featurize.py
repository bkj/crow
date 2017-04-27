#!/usr/bin/env python

"""
    threaded-featurize.py
    
    !! VGG16.pool5 is hardcoded
    !! Might want to allow input of different sizes -- VGG16 w/o last layers is an FCN
"""

import os
import sys
import h5py
import argparse
import numpy as np

import urllib
import cStringIO
from time import time

from multiprocessing import Process, Queue
from Queue import Empty

from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# --
# Init

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, default='./db.h5')
    parser.add_argument('--n-threads', type=int, default=3)
    return parser.parse_args()

# --
# Threaded IO

def prep_images(in_, out_, target_dim=224):
    while True:
        try:
            path = in_.get(timeout=5)
            try:
                if 'http' == path[:4]:
                    img = cStringIO.StringIO(urllib.urlopen(path).read())
                    img = image.load_img(img, target_size=(target_dim, target_dim))
                else:
                    img = image.load_img(path, target_size=(target_dim, target_dim))
                
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                out_.put((path, img))
            except KeyboardInterrupt:
                raise
            except:
                print >> sys.stderr, "Error: Cannot load @ %s" % path
        
        except KeyboardInterrupt:
            raise
        
        except Empty:
            return

def read_stdin(gen, out_):
    for line in gen:
        out_.put(line.strip())

# --
# Run

if __name__ == "__main__":
    args = parse_args()
    
    db = h5py.File(args.outpath)
    
    model = VGG16(weights='imagenet', include_top=False)
    
    # Thread to read from std
    filenames = Queue()
    newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
    stdin_reader = Process(target=read_stdin, args=(newstdin, filenames))
    stdin_reader.start()
    
    # Thread to load images    
    processed_images = Queue()
    image_processors = [Process(target=prep_images, args=(filenames, processed_images)) for _ in range(args.n_threads)]
    for image_processor in image_processors:
        image_processor.start()
    
    i = 0
    start_time = time()
    while True:
        i += 1
        
        if not i % 100:
            print >> sys.stderr, "%f seconds to featurize %d images" % (time() - start_time, i + 1)
        
        try:
            path, img = processed_images.get(timeout=5)
            feat = model.predict(img)
            db[path] = feat
        except KeyboardInterrupt:
            raise
        except Empty:
            db.close()
            os._exit(0)
        except:
            pass
    
    print >> sys.stderr, "Total: %f seconds to featurize %d images" % (time() - start_time, i + 1)