"""Simple script to run inference on a pre-trained CryoViT model (from weights.pt) without Hydra overhead."""

from pathlib import Path
import argparse

import h5py
import mrcfile
import pandas as pd
from omegaconf import OmegaConf

from cryovit.processing.preprocessing import run_preprocess
from cryovit.config import ExpPaths, PretrainedModel, CryoVIT, TrainerInfer, Inference, DinoFeaturesConfig, InferModelConfig
from cryovit import dino_features, infer_model

def rescale_input(data_path: Path):
    all_files = []
    all_files.extend(data_path.glob("**/*.mrc"))
    all_files.extend(data_path.glob("**/*.hdf"))
    result_path = data_path.parent / "Rescaled"
    for file in all_files:
        result_file = result_path / (file.relative_to(data_path))
        try:
            match file.suffix:
                case ".mrc":
                    with mrcfile.open(str(file), "r") as f:
                        data = f.data[:]
                case ".hdf":
                    with h5py.File(file, "r") as fh:
                        if "data" in fh.keys():
                            data = fh["data"][()]
                        else:
                            data = fh["MDF"]["images"]["0"]["image"][()]
                case _:
                    print(f"Unknown suffix for {file}, skipping.")
                    continue
        except OSError as e:
            print(f"Error loading file {file}. {e}.")
            continue
        _, H, W = data.shape
        new_H, new_W = 16 * round(H / 16), 16 * round(W / 16)
        run_preprocess(file, result_file, bin_size=1, resize_image=(new_H, new_W), normalize=False, clip=False)
    return result_path

def create_dataset(data_path: Path, result_path: Path):
    # Treat subfolders of data_path as samples
    samples = [d.name for d in data_path.iterdir() if d.is_dir()]
    # Create dataframe of ["sample", "tomo_name"]
    df_dict = {"sample": [], "tomo_name": []}
    valid_files = []
    valid_files.extend(data_path.glob("**/*/.mrc"))
    valid_files.extend(data_path.glob("**/*.hdf"))
    for f in valid_files:
        sample = f.parent.name
        if sample in samples:
            df_dict["sample"].append(sample)
            df_dict["tomo_name"].append(f.name)
    df = pd.DataFrame.from_dict(df_dict)
    csv_path = (result_path / "samples.csv").resolve()
    df.to_csv(str(csv_path))
    return csv_path

def create_infer_config(data_path: Path, result_path: Path, weights_path: Path, label_key: str):
    exp_paths = ExpPaths(exp_dir=result_path, tomo_dir=data_path, split_file=create_dataset(data_path, result_path))
    pretrained_config = PretrainedModel(name=weights_path.stem, label_key=label_key, model_weights=str(weights_path.resolve()), model_type="CRYOVIT", model=CryoVIT())
    infer_config = InferModelConfig(models=[pretrained_config], trainer=TrainerInfer(), dataset=Inference(), exp_paths=exp_paths)
    return OmegaConf.structured(infer_config)

def run_dino_features(dino_dir: Path, data_path: Path, result_path: Path, batch_size: int = 8):
    print("Running DINOv2...")
    samples = [d.name for d in data_path.iterdir() if d.is_dir()]
    dino_config = DinoFeaturesConfig(dino_dir=dino_dir, tomo_dir=data_path, csv_dir=None, feature_dir=result_path, batch_size=batch_size, sample=samples)
    dino_config = OmegaConf.structured(dino_config)
    print("Config created.")
    dino_features.main(dino_config)

def run_inference(data_path: Path, result_path: Path, weights_path: Path, label_key: str):
    print("Running inference...")
    infer_config = create_infer_config(data_path, result_path, weights_path, label_key)
    print("Config created.")
    infer_model.main(infer_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs CryoViT inference on a dataset of tomograms wihtout any extra fuss.")
    parser.add_argument("data_path", help="path to the directory of the tomogram dataset")
    parser.add_argument("features_path", help="path to the directory to save intermediate DINOv2 features")
    parser.add_argument("result_path", help="path to the directory to save segmentation results")
    parser.add_argument("weights_path", help="path to the file with pretrained CryoViT weights (a .pt file)")
    parser.add_argument("--dino_path", default="pretrained_models", help="path to the directory to save DINOv2. defaults to a pretrained_models folder in the current working directory")
    parser.add_argument("--label_key", default="unknown", help="label for calculated predictions. defaults to unknown")
    parser.add_argument("--batch_size", default=8, help="batch size to process dino features. defaults to 8")
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    features_path = Path(args.features_path)
    result_path = Path(args.result_path)
    weights_path = Path(args.weights_path)
    dino_path = Path(args.dino_path)
    label_key = args.label_key
    batch_size = args.batch_size
    
    rescaled_path = rescale_input(data_path)
    run_dino_features(dino_path, rescaled_path, features_path, batch_size)
    run_inference(features_path, result_path, weights_path, label_key)
    