import os
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image
from tqdm import tqdm
import argparse


def insert_annotation(annotation, idx, feature_labels):
    for i, (feat, feat_labels) in enumerate(feature_labels.items()):
        label = np.where(annotation == 254 - i, 1, 0)
        label = ndimage.binary_fill_holes(label)
        feat_labels[idx] = label.astype(np.int8)


def generate_data(sample, features, preprocessed_dir, train_dir, annotation_dir, tomo_name, slices, z_limits):
    input_tomo_path = os.path.join(preprocessed_dir, sample, tomo_name)
    output_tomo_path = os.path.join(train_dir, sample, tomo_name)
    z_min, z_max = z_limits
    features = features.split(",")
    feature_labels = {}

    with h5py.File(input_tomo_path) as fh:
        data = fh["data"][()]

    for feat in features:
        labels = -1 * np.ones_like(data, dtype=np.int8)
        labels[:z_min] = 0
        labels[z_max:] = 0
        feature_labels[feat] = labels

    for idx in slices:
        in_path = os.path.join(
            annotation_dir, sample, f"{tomo_name[:-4]}_{idx}.png"
        )

        if os.path.exists(in_path): # some slices have no annotations
            annotation = np.asarray(Image.open(in_path))
            insert_annotation(annotation, idx, feature_labels)

    with h5py.File(output_tomo_path, "w") as fh:
        fh.create_dataset("data", data=data, compression="gzip")
        for feat in features:
            fh.create_dataset(feat, data=feature_labels[feat], compression="gzip")


def process_sample(sample, features, csv_dir, preprocessed_dir, train_dir, annotation_dir):
    annotation_file = os.path.join(csv_dir, f"{sample}.csv")
    print(f"Processing sample {sample}")

    if not os.path.exists(annotation_file):
        raise RuntimeError(f"No annotations for {sample} were found")
    
    dst_dir = os.path.join(train_dir, sample)
    os.makedirs(dst_dir, exist_ok=True)
    df = pd.read_csv(annotation_file)

    for row in tqdm(df.itertuples()):
        generate_data(sample=sample, features=features, preprocessed_dir=preprocessed_dir, train_dir=train_dir, annotation_dir=annotation_dir, tomo_name=row[1], slices=row[4:], z_limits=row[2:4])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str, help='sample name')
    parser.add_argument('features', type=str, help='string of features (in descending pixel value) separated by a comma')
    parser.add_argument('csv_dir', type=str, help='path to directory with csv files')
    parser.add_argument('preprocessed_dir', type=str, help='path to directory with preprocessed tomograms')
    parser.add_argument('train_dir', type=str, help='path to dest directory for tomograms to use for training')
    parser.add_argument('annotation_dir', type=str, help='path to directory with annotated slices')
    args = parser.parse_args()

    process_sample(args.sample, args.features, args.csv_dir, args.preprocessed_dir, args.train_dir, args.annotation_dir)


if __name__ == "__main__":
    main()
