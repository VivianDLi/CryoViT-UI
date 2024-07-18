import os
import h5py
import numpy as np
import torch
import argparse

pooler = torch.nn.AvgPool3d(kernel_size=2, stride=2, ceil_mode=True)

def copy_file(src_path, dst_path):
    with h5py.File(src_path) as fh:
        data = fh["MDF"]["images"]["0"]["image"][()]

    data = np.expand_dims(data, [0, 1])
    data = torch.tensor(data)

    data = pooler(data)
    # data = torch.nn.functional.interpolate(data, size=(128, 512, 512))

    data = data.squeeze().numpy()
    print(f"Shape is {data.shape}, {data.dtype}")

    # Contrast normalized samples are clipped to +/-3 std devs.
    # We further rescale to [-1, 1]
    # data = (data - np.mean(data))/np.std(data)
    data = np.clip(data, -3.0, 3.0) / 3.0

    with h5py.File(dst_path, "w") as fh:
        fh.create_dataset("data", data=data)


def copy_samples(sample, raw_dir, processed_dir):
    src_dir = os.path.join(raw_dir, sample)
    dst_dir = os.path.join(processed_dir, sample)
    os.makedirs(dst_dir, exist_ok=True)

    print(f"Found {len(os.listdir(src_dir))} samples of {sample}")

    for file_name in os.listdir(src_dir):
        if not (
            file_name.endswith(".rec")
            or file_name.endswith(".mrc")
            or file_name.endswith(".hdf")
        ):
            continue

        print(f"Copying {file_name}")
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        dst_path = dst_path.split(".")[0] + ".hdf"

        try:
            copy_file(src_path, dst_path)
        except Exception as e:
            print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str, help='sample name or All')
    parser.add_argument('raw_dir', type=str, help='path to directory with raw tomograms')
    parser.add_argument('processed_dir', type=str, help='path to dest directory for processed tomograms')
    args = parser.parse_args()

    copy_samples(args.sample, args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
