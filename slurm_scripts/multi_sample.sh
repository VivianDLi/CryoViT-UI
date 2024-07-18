#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 model (cryovit or unet3d) label_key (mito or granule) samples test_samples"
    exit 1
fi

exp_cmd="$(dirname "$0")/multi_sample_job.sh $1 $2 $3 $4"
job_name="multi_sample_${3}_${4}_${1}_${2}"
out_dir="$(dirname "$0")/outputs"

sbatch \
    --partition="cryoem" \
    --job-name="$job_name" \
    --output="${out_dir}/${job_name}.out" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem-per-cpu=6gb \
    --gres=gpu:v100:1 \
    --time=04:00:00 \
    --wrap="$exp_cmd"
