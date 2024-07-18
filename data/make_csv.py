import os
import csv
import argparse


def make_csv(sample, processed_dir, csv_dir):
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, sample + ".csv")
    tomo_path = os.path.join(processed_dir, sample)
    tomograms = sorted(os.listdir(tomo_path))
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        header = ["tomo_name","z_min","z_max","slice_0","slice_1","slice_2","slice_3","slice_4"]
        writer.writerow(header)

        for tomo in tomograms:
            writer.writerow([tomo, 0, 0, 0, 0, 0, 0, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str, help='sample name')
    parser.add_argument('processed_dir', type=str, help='path to directory with processed tomograms')
    parser.add_argument('csv_dir', type=str, help='path to dest directory for csv files')
    args = parser.parse_args()

    make_csv(args.sample, args.processed_dir, args.csv_dir)


if __name__ == "__main__":
    main()