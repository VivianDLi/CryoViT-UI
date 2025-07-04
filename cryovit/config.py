"""Config file for CryoVIT experiments."""

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Dict, Union
from typing import List
from typing import Optional
from typing import Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class Sample(Enum):
    BACHD = "BACHD"
    WT = "WT"
    Q109 = "Q109"
    Q18 = "Q18"
    Q66 = "Q66"
    Young = "Young"
    Aged = "Aged"


samples = [sample.name for sample in Sample]

tomogram_exts = [".rec", ".mrc", ".hdf"]


@dataclass
class DinoFeaturesConfig:
    """Configuration for managing Dino features within CryoVIT experiments.

    Attributes:
        dino_dir (Path): Path to the DINOv2 foundation model.
        tomo_dir (Path): Directory containing tomograms for feature extraction.
        csv_dir (Path): Directory containing CSV files with tomogram metadata.
        feature_dir (Path): Destination to save the generated DINOv2 features.
        batch_size (int): Batch size: number of slices in one batch.
        sample (Sample): Enum representing a specific sample under study.
        all_samples (str): Comma-separated string of all sample names.
        cryovit_root (Optional[Path]): Root directory for the CryoVIT package.
    """

    dino_dir: Path
    tomo_dir: Path
    csv_dir: Union[Path, None]
    feature_dir: Path
    batch_size: int
    sample: List[Sample]
    all_samples: str = ",".join(samples)
    cryovit_root: Optional[Path] = None


@dataclass
class Model:
    """Base class for model configurations used in CryoVIT experiments.

    Attributes:
        lr (float): Learning rate for the model training.
        weight_decay (float): Weight decay (L2 penalty) rate. Default is 1e-3.
        losses (Tuple[Dict]): Configuration for loss functions used in training.
        metrics (Tuple[Dict]): Configuration for metrics used during model evaluation.
    """

    lr: float
    weight_decay: float = 1e-3
    losses: Tuple[Dict] = (dict(_target_="cryovit.models.losses.DiceLoss"),)
    metrics: Tuple[Dict] = (
        dict(_target_="cryovit.models.metrics.DiceMetric", threshold=0.5),
    )


@dataclass
class CryoVIT(Model):
    """Configuration for the CryoVIT model.

    Attributes:
        lr (float): Learning rate for the CryoVIT model, default set to 1e-4.
        _target_ (str): Class identifier for instantiating a CryoVIT model.
    """

    lr: float = 1e-4
    _target_: str = "cryovit.models.CryoVIT"


@dataclass
class UNet3D(Model):
    """Configuration for the UNet3D model.

    Attributes:
        lr (float): Learning rate for the UNet3D model, default set to 3e-3.
        _target_ (str): Class identifier for instantiating a UNet3D model.
    """

    lr: float = 3e-3
    _target_: str = "cryovit.models.UNet3D"


@dataclass
class PretrainedModel:
    """Configuration for managing and loading pretrained models in CryoVIT experiments.

    Attributes:
        name (str): Name of the pretrained model, must be unique for each configuration.
        label_key (str): Key used to specify the training label.
        samples (List[str]): List of sample names used for training.
        model_weights (Path): Directory containing the pretrained model weights.
        model_type (ModelArch): Architecture type of the pretrained model.
        model (Model): The pretrained model configuration.
    """

    name: str = MISSING
    label_key: str = MISSING
    model_weights: Path = MISSING
    model_type: str = MISSING
    model: Model = MISSING


@dataclass
class Trainer:
    """Base class for trainer configurations used in CryoVIT experiments.

    Attributes:
        accelerator (str): Type of hardware acceleration ('gpu' for this configuration).
        devices (str): Number of devices to use for training.
        precision (str): Precision configuration for training (e.g., '16-mixed').
        callbacks (Optional[List]): List of callback functions for training sessions.
        enable_checkpointing (bool): Flag to enable or disable model checkpointing.
        _target_ (str): Class identifier for instantiating a Trainer object.
    """

    accelerator: str = "gpu"
    devices: str = 1
    precision: str = "16-mixed"
    callbacks: Optional[List] = None
    enable_checkpointing: bool = False
    _target_: str = "pytorch_lightning.Trainer"


@dataclass
class TrainerFit(Trainer):
    """Specific configuration for fitting (training) models in CryoVIT experiments.

    Attributes:
        max_epochs (int): Maximum number of training epochs.
        logger (Optional[List]): Logging configuration for training process.
        log_every_n_steps (int): Interval of logging within the training process.
        num_sanity_val_steps (int): Number of validation steps to perform at start for sanity check.
    """

    max_epochs: int = 50
    logger: Optional[List] = None

    log_every_n_steps: int = 1
    num_sanity_val_steps: int = 0


@dataclass
class TrainerEval(Trainer):
    """Configuration for model evaluation in CryoVIT experiments.

    Attributes:
        logger (bool): Flag to enable or disable logging during evaluation.
        enable_model_summary (bool): Flag to enable or disable generation of model summaries.
    """

    logger: bool = False
    enable_model_summary: bool = False


@dataclass
class TrainerInfer(Trainer):
    """Configuration for model inference in CryoVIT experiments.

    Attributes:
        logger (bool): Flag to enable or disable logging during inference.
        enable_model_summary (bool): Flag to enable or disable generation of model summaries.
    """

    logger: bool = False
    enable_model_summary: bool = False


@dataclass
class Dataset:
    """Base class for dataset configurations in CryoVIT experiments.

    Attributes:
        _partial_ (bool): Flag to indicate this is a partial configuration.
    """

    _partial_: bool = True


@dataclass
class SingleSample(Dataset):
    """Configuration for a dataset involving a single sample in CryoVIT experiments.

    Attributes:
        split_id (Optional[int]): Optional split ID to be excluded from training and used for eval.
        sample (Sample): Specific sample used in this dataset configuration.
        _target_ (str): Class identifier for instantiating this dataset.
    """

    split_id: Optional[int] = None
    sample: Sample = MISSING
    _target_: str = "cryovit.data_modules.SingleSampleDataModule"


@dataclass
class MultiSample(Dataset):
    """Configuration for a dataset involving multiple samples in CryoVIT experiments.

    Attributes:
        split_id (Optional[int]): Optional split ID for validation.
        sample (Tuple[Sample]): Tuple of samples used for training.
        test_samples (Tuple[Sample]): Tuple of samples used for testing.
        _target_ (str): Class identifier for instantiating this dataset.
    """

    split_id: Optional[int] = None
    sample: Tuple[Sample] = MISSING
    test_samples: Tuple[Sample] = ()
    _target_: str = "cryovit.data_modules.MultiSampleDataModule"


@dataclass
class LOOSample(Dataset):
    """Leave-One-Out (LOO) dataset configuration for CryoVIT experiments.

    Attributes:
        split_id (Optional[int]): Optional split ID for validation.
        sample (Sample): Sample excluded from training (used for testing).
        all_samples (Tuple[Sample]): Tuple of all samples, including the LOO sample.
        _target_ (str): Class identifier for instantiating this dataset.
    """

    split_id: Optional[int] = None
    sample: Sample = MISSING
    all_samples: Tuple[Sample] = tuple(s for s in Sample)
    _target_: str = "cryovit.data_modules.LOOSampleDataModule"


@dataclass
class FractionalLOO(Dataset):
    """Fractional Leave-One-Out (LOO) dataset configuration in CryoVIT experiments.

    Attributes:
        split_id (int): Number of splits to be used for training.
        sample (Sample): Sample excluded from training (used for testing).
        all_samples (Tuple[Sample]): Tuple of all samples, including the LOO sample.
        _target_ (str): Class identifier for instantiating this dataset.
    """

    split_id: int = MISSING
    sample: Sample = MISSING
    all_samples: Tuple[Sample] = tuple(s for s in Sample)
    _target_: str = "cryovit.data_modules.FractionalSampleDataModule"


@dataclass
class Inference(Dataset):
    """Configuration for inference dataset in CryoVIT experiments.

    Attributes:
        _target_ (str): Class identifier for instantiating this dataset.
    """

    _target_: str = "cryovit.data_modules.InferenceDataModule"


@dataclass
class ExpPaths:
    """Configuration for managing experiment paths in CryoVIT experiments.

    Attributes:
        exp_dir (Path): Directory path for saving results from an experiment.
        tomo_dir (Path): Directory path for tomograms with their DINOv2 features.
        split_file (Path): Path to the CSV file specifying data splits.
        cryovit_root (Optional[Path]): Root directory for the CryoVIT package.
    """

    exp_dir: Path
    tomo_dir: Path
    split_file: Path | None
    cryovit_root: Optional[Path] = None


@dataclass
class DataLoader:
    """Configuration for data loader settings used in CryoVIT experiments.

    Attributes:
        num_workers (int): Number of worker processes for loading data.
        prefetch_factor (Optional[int]): Number of batches to prefetch (default is 1).
        persistent_workers (bool): If True, the data loader will not shutdown workers between epochs.
        pin_memory (bool): If True, enables memory pinning for faster data transfer to GPU.
        batch_size (Optional[int]): Number of samples per batch, can be unspecified for default.
        _target_ (str): Class identifier for instantiating the data loader.
        _partial_ (bool): Flag to indicate this is a partial configuration.
    """

    num_workers: int = 8
    prefetch_factor: Optional[int] = 1
    persistent_workers: bool = False
    pin_memory: bool = True
    batch_size: Optional[int] = None
    _target_: str = "torch.utils.data.DataLoader"
    _partial_: bool = True


@dataclass
class TrainModelConfig:
    """Configuration for training a model in CryoVIT experiments.

    Attributes:
        exp_name (str): The name of the experiment, must be unique for each configuration.
        label_key (str): Key used to specify the training label: mito or mito_ai.
        aux_keys (Tuple[str]): Additional keys to load auxiliary data from tomograms.
        random_seed (int): Random seed for reproducibility.
        save_pretrained (bool): Flag to indicate if to use simplified exp_dir and save model as a hydra config.
        model (Model): The model configuration to be used for training.
        trainer (TrainerFit): Trainer configuration tailored for training sessions.
        dataset (Dataset): Dataset configuration to be used for model training.
        exp_paths (ExpPaths): Configuration paths relevant to the experiment.
        dataloader (DataLoader): DataLoader configuration for handling input data efficiently.
    """

    exp_name: str = MISSING
    label_key: str = MISSING
    aux_keys: Tuple[str] = ()
    random_seed: int = 42
    save_pretrained: bool = False

    model: Model = MISSING
    trainer: TrainerFit = MISSING
    dataset: Dataset = MISSING
    exp_paths: ExpPaths = MISSING
    dataloader: DataLoader = field(default=DataLoader())


@dataclass
class EvalModelConfig:
    """Configuration for evaluating a model in CryoVIT experiments.

    Attributes:
        exp_name (str): The name of the experiment, must be unique for each configuration.
        label_key (str): Key used to specify the training label: mito or mito_ai.
        aux_keys (Tuple[str]): Additional keys to load auxiliary data from tomograms.
        model (Model): The model configuration to be used for evaluation.
        trainer (TrainerEval): Trainer configuration specifically designed for evaluation sessions.
        dataset (Dataset): Dataset configuration intended for model evaluation.
        exp_paths (ExpPaths): Configuration paths relevant to the experiment.
        dataloader (DataLoader): DataLoader configuration optimized for evaluation scenarios.
    """

    exp_name: str = MISSING
    label_key: str = MISSING
    aux_keys: Tuple[str] = ("data",)

    model: Model = MISSING
    trainer: TrainerEval = MISSING
    dataset: Dataset = MISSING
    exp_paths: ExpPaths = MISSING
    dataloader: DataLoader = field(default=DataLoader())


@dataclass
class InferModelConfig:
    """Configuration for inferring using model(s) for CryoViT use.

    Attributes:
        aux_keys (Tuple[str]): Additional keys to load auxiliary data from tomograms.
        models (List[PretrainedModel]): The model configurations to be used for inference.
        trainer (TrainerInfer): Trainer configuration specifically designed for inference sessions.
        dataset (Dataset): Dataset configuration intended for model inference.
        exp_paths (ExpPaths): Configuration paths relevant to the experiment.
        dataloader (DataLoader): DataLoader configuration optimized for inference scenarios.
    """

    aux_keys: Tuple[str] = ("data",)

    models: List[PretrainedModel] = MISSING
    trainer: TrainerInfer = MISSING
    dataset: Dataset = MISSING
    exp_paths: ExpPaths = MISSING
    dataloader: DataLoader = field(default=DataLoader())


cs = ConfigStore.instance()
cs.store(name="dino_features_config", node=DinoFeaturesConfig)

cs.store(group="model", name="cryovit", node=CryoVIT)
cs.store(group="model", name="unet3d", node=UNet3D)

cs.store(group="trainer", name="trainer_fit", node=TrainerFit)
cs.store(group="trainer", name="trainer_eval", node=TrainerEval)
cs.store(group="trainer", name="trainer_infer", node=TrainerInfer)

cs.store(group="dataset", name="single", node=SingleSample)
cs.store(group="dataset", name="multi", node=MultiSample)
cs.store(group="dataset", name="loo", node=LOOSample)
cs.store(group="dataset", name="fractional", node=FractionalLOO)
cs.store(group="dataset", name="inference", node=Inference)

cs.store(name="train_model_config", node=TrainModelConfig)
cs.store(name="eval_model_config", node=EvalModelConfig)
cs.store(name="infer_model_config", node=InferModelConfig)
