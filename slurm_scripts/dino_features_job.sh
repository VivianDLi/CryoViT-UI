#!/bin/bash

sample=$1

env_dir=/tmp/$USER/"$(uuidgen)"
mkdir -p $env_dir
tar -xf ~/projects/libs/cryovit_env_test.tar -C $env_dir

$env_dir/tmp/cathy/cryovit_env_test/bin/python -m \
    cryovit.dino_features \
    sample=$sample 

rm -rf $env_dir