#!/bin/bash

python ./src/model/PhyloSpec_train_test.py -t ./example/Unclassified/phylogeny.nwk -c ./example/Unclassified/example_train.csv -taxo ./example/Unclassified/example_taxonomy.csv --PhyloSpec train

python ./src/model/PhyloSpec_train_test.py -t ./example/Unclassified/phylogeny.nwk -c ./example/Unclassified/example_test.csv -taxo ./example/Unclassified/example_taxonomy.csv --PhyloSpec test

