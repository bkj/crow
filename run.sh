#!/bin/bash

find ./oxford/data/ -type f | ./threaded-featurize.py --outpath ./oxford.h5
find ./paris/data/ -type f | ./threaded-featurize.py --outpath ./paris.h5

python whiten.py --index-features ./oxford.h5 --whiten-features ./paris.h5 --outpath ./oxford-white
python whiten.py --index-features ./paris.h5 --whiten-features ./oxford.h5 --outpath ./paris-white