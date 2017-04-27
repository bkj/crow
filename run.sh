#!/bin/bash

find ./oxford/data/ -type f | ./threaded-featurize.py --outpath ./oxford.h5
find ./paris/data/ -type f | ./threaded-featurize.py --outpath ./paris.h5