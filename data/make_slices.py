import os
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import io
from tqdm import tqdm
import sys
from PIL import Image
import csv
import argparse


def generate_slices(sample, tomo_name, slices, z_limits, tomo_dir, slices_dir):
    input_tomo_path = os.path.join(tomo_dir, sample, tomo_name)
    output_tomo_path = os.path.join(slices_dir, sample, tomo_name)
    z_min, z_max = z_limits

    with h5py.File(input_tomo_path) as fh:
        data = fh["data"][()]
        # data = ((data + 1) * 0.5).astype(np.uint8)
        # data = (data * 255).astype(np.uint8)

    print(data.shape)
    for idx in slices:
        out_path = os.path.join(
            slices_dir, sample, f"{tomo_name[:-4]}_{idx}.png"
        )
        img = data[idx]
        img = ((img+1)*0.5 * 255 / np.max(img)).astype('uint8')
        img = Image.fromarray(img)
        img.save(out_path)


def save_slices(sample, tomo_dir, csv_dir, slices_dir):
    slices_file = os.path.join(csv_dir, f"{sample}.csv")
    print(f"Processing sample {sample}")

    if not os.path.exists(slices_file):
        raise RuntimeError(f"No slices for {sample} were found")

    dst_dir = os.path.join(slices_dir, sample)
    os.makedirs(dst_dir, exist_ok=True)
    df = pd.read_csv(slices_file)

    for row in tqdm(df.itertuples()):
        generate_slices(sample=sample, tomo_name=row[1], slices=row[4:], z_limits=row[2:4], tomo_dir=tomo_dir, slices_dir=slices_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str, help='sample name')
    parser.add_argument('tomo_dir', type=str, help='path to directory with tomograms')
    parser.add_argument('csv_dir', type=str, help='path to directory with csv files')
    parser.add_argument('slices_dir', type=str, help='path to directory to put slices')
    args = parser.parse_args()

    save_slices(args.sample, args.tomo_dir, args.csv_dir, args.slices_dir)


if __name__ == "__main__":
    main()