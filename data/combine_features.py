"""Generate Train Data

Reads the annotations downloaded from hasty.ai and makes labels
The mito and granule labels are stored along with the raw_data
Z limits from the annotation file are used to annotate slices
which are confirmed to have no mito or granules

Usage:
  generate_train_data.py <sample>
  generate_train_data.py (-h | --help)

Options:
  -h --help         Show this screen

"""
import os

import h5py
import numpy as np
import pandas as pd
from docopt import docopt
from scipy import ndimage
from tqdm import tqdm
import argparse

def combine_annotations(results_dir, features, sample, tomo_name, dst_dir):
    features = features.split(",")
    paths = {}

    dst_dir = os.path.join(dst_dir, sample)
    os.makedirs(dst_dir, exist_ok=True)

    output_tomo_path = os.path.join(dst_dir, tomo_name)
    with h5py.File(output_tomo_path, "w") as fh_output:
        for feat in features:
            input_tomo_path = os.path.join(results_dir, feat, sample, tomo_name)
            with h5py.File(input_tomo_path) as fh_input:
                preds = fh_input["preds"][()]
                fh_output.create_dataset(feat, data=preds, compression="gzip")
                data = "data"
                if data not in fh_output:
                    data = fh_input["data"][()]
                    fh_output.create_dataset("data", data=data, compression="gzip")

    

def process_sample(results_dir, features, sample, dst_dir):
    feat = features.split(",")[0]
    r_dir = os.path.join(results_dir, feat, sample)

    if not os.path.exists(r_dir):
        raise RuntimeError(f"No {feat} annotations for {sample} were found")
    
    for tomo_name in os.listdir(r_dir): 
        combine_annotations(results_dir, features, sample, tomo_name, dst_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str, help='sample name')
    parser.add_argument('features', type=str, help='string of features separated by a comma')
    parser.add_argument('results_dir', type=str, help='path to directory with annotated tomograms')
    parser.add_argument('dst_dir', type=str, help='path to dest directory for combined tomograms')
    args = parser.parse_args()

    process_sample(args.results_dir, args.features, args.sample, args.dst_dir)


if __name__ == "__main__":
    main()
