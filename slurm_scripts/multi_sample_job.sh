#!/bin/bash
# multi

# split_id=$1
model=$1
label_key=$2
samples=$3
test_samples=$4

env_dir=/tmp/$USER/"$(uuidgen)"
mkdir -p $env_dir
tar -xf ~/projects/libs/cryovit_env_test.tar -C $env_dir

$env_dir/tmp/cathy/cryovit_env_test/bin/python -m \
    cryovit.train_model \
    model=$model \
    label_key=$label_key \
    exp_name="multi_sample_${model}_${label_key}" \
    dataset=multi \
    dataset.sample=$samples \
    dataset.test_samples=$test_samples \


$env_dir/tmp/cathy/cryovit_env_test/bin/python -m \
    cryovit.eval_model \
    model=$model \
    label_key=$label_key \
    exp_name="multi_sample_${model}_${label_key}" \
    dataset=multi \
    dataset.sample=$samples \
    dataset.test_samples=$test_samples \

rm -rf $env_dir