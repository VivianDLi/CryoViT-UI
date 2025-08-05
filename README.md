# CryoVIT: Efficient Segmentation of Cryo-electron Tomograms with Vision Foundation Models

## Installation Instructions
This codebase only contains a GUI to help setup data for running CryoViT training. The CryoViT code repository is also required, which can be downloaded [here]([https://github.com/sanketx/CryoVIT).

Distribution-specific installers can be found in `releases`, but these are untested, so follow the instructions for [Building from Source Code](#Building-from-Source-Code) below.

### Building from Source Code

Both CryoViT GUI and CryoViT use `mamba` to manage python packages and dependencies, which can be downloaded [here](https://github.com/conda-forge/miniforge). You should also be able to use `conda` instead of Mamba but setting up the environment may take an unreasonably long time.

To setup the environment:

1. Clone this github repository: `git clone https://github.com/VivianDLi/CryoViT-UI.git`
2. Go into the project directory: `cd CryoViT-UI`
3. Setup the mamba environment: `mamba env create -f env.yml`
4. Activate the mamba environment: `mamba activate cryovit_gui_env`
5. Setup the python package (this step should only need to be done once): `pip install -e .`

The same steps can be followed to setup CryoViT, only replacing `CryoViT-UI` with `CryoViIT` and `cryovit_gui_env` with `cryovit_env`.

## Usage Guide

This application is meant to be used in conjunction with the [CryoViT source code](https://github.com/sanketx/CryoVIT), and as such, does not support model training or inference out of the box. Instead, the main use is to help format the commands to train and run CryoViT models, and to facilitate the annotation process for training CryoViT.

To train and use a model from scratch, follow the following protocol:

### 1. Setup your dataset directory:

CryoViT expects your data to be in the following structure:

```
+-- Data (can be any name)
|  +-- tomograms
|  +-- csv
|  +-- slices
|  +-- annotations
```

You can put your `raw training data` in a `raw` folder (or something similar) in the `Data` directory. This data is expected to be organized into `samples` (i.e., subfolders containing either `.mrc` or `.hdf` tomogram files). If you just have a bunch of tomograms, just create a new sample folder in your `raw` folder with the tomograms inside.

### 2. Pre-processing:

Launch the CryoViT GUI by running `cryovit_gui/app.py` with the `cryovit_gui_env` environment activated (see [Building from Source Code](#Building-from-Source-Code)).

The application will open on the **pre-processing tab**, with two main sections, a dropdown configuration setter and a button. Using the dropdown, setup the pre-processing settings and directories.

By default, you should only need to specify the `Raw Data Directory` (this should be your `raw` folder, or something similar), and the `Target Directory` (this should be your `tomograms` folder).
> **__Directory Selection__**: Double-clicking config values (i.e., the right column) allows you to **edit** their values.
> 
> For directories, *single-clicking* in edit-mode opens a file-select screen, while *double-clicking* in edit-mode allows manual text editing.

When the config is setup, click the button to run pre-processing. This will copy a command into your clipboard that can be pasted and run in the CryoViT environment to run pre-processing, or you can use the dialog box to run pre-processing locally (i.e., using the GUI).

### 3. Selecting Annotation Slices:

CryoViT is a minimally-supervised model (i.e., it trains using a small number of annotated data). This is typically 5 good-quality slices in each training tomogram that are annotated with what you want to segment. The **Annotations** tab (next to the **Pre-processing** tab) is used to select and setup these slices.

First, select the `Project Root Directory` at the top. This should be your `Data` directory. This can either be manually entered using the text box, or selected with a prompt using the button on the right. The application will provide warnings and the root directory will not be set if the selected directory doesn't follow the structure specified in [Step 1](#1.-Setup-your-dataset-directory).

With the root directory selected, all `sample` sub-directories will be shown in the section below, with the current progress displayed, and completed samples highlighted in green. The progress indicates how many tomograms in each sample have either: (1) had slices extracted to later be annotated, or (2) have annotations present in an `annotations` folder.

To select which slices to annotate, select a sample, and then, in the section below, check all tomograms that you want to select for. Tomograms with no extracted slices are automatically selected by default, and the **Select All**, **Deselect All**, and **Reset to Default**, change your selections appropriately.

Then, press the `Launch ChimeraX` button to open ChimeraX to select slices to later annotate.
> **__Specifying ChimeraX Settings__**: The path to your ChimeraX executable needs to be set in the application settings before this step.
> 
> This is accessed through the File Menu > Settings (Preferences on Mac), and under the **Annotation** dropdown, set the `Chimera Path` settings. The `Number of Slices` setting can also be set to specify the number of slices to use as training data (by default, 5).

> **__Finding the ChimeraX Path__**: On **Windows**, the ChimeraX path is a `ChimeraX.exe` file in your ChimeraX install location in the `bin` folder.
> 
> On **Linux**, setting this is not necessary.
> 
> On **Mac**, this should be a `ChimeraX` file in the `Contents/MacOS/` directory of your ChimeraX application location (e.g., `Applications/ChimeraX.app/Contents/MacOS/ChimeraX`).

#### 3.i. Using ChimeraX to Select Slices:

When launching ChimeraX, a command will be saved to your clipboard. When ChimeraX successfully launches, paste and run that command in the ChimeraX command line. This will open a tomogram to select z-limits and slices for.

When the tomogram loads, select plane markers (`Markers > Plane`), and under `Marker Placement` on the right, select `Options`, and under `Marker set`, select `zlimits`. Then, navigating between tomogram slices, place **two** markers (_right-click_) on the slices where you can first see or first stop seeing objects of interest in the tomogram. If objects are always visible in all slices, set the markers on the first and last slices.

Then, switch to the `slices` marker set, and place **n** markers on the slices you want to export for annotation (where **n** is set in the application settings).

When finished, type `next` in the ChimeraX command line to save your z-limits and slices, and open the next tomogram. If the number of markers for either marker set is wrong, a warning will be shown.

At any point, if you want to stop, simply close ChimeraX. Progress is saved whenever you type `next`. When all selected tomograms have been marked, you can close ChimeraX.

### 4. Annotating Slices:

You can double-check you've selected slices for all training tomograms with the `Slices exported?` or `Slice Progress` columns. When all tomograms are completed, press the `Extract Slices` button to export slices as `.png` images for annotation in [_Hasty.ai_](https://app.hasty.ai). If not all tomograms in the sample have been processed, a warning will be shown, and missing tomograms can be easily found by looking in the `Slices Exported?` column.

In Hasty, you can import your slices at `Content > Import Annotations`, and start annotating them using the `Start annotating` button on the top left.
> **__Annotation Tips__**: Remember important keybinds, like `B` to enable brush mode, `E` to enable eraser, and `Shift + F` to fill any holes.
> 
> For large objects, I recommend outlining the shape, and then using `Shift + F` to fill in the outlines.
> 
> Ergonomically, I recommend using a tablet, having the non-pen hand on a keyboard for shortcuts.
> 
> Don't be afraid to use `Ctrl + Z` for undoing bad lines, or `Ctrl + Shift + Z` for redos. Also helpful are `Shift + N` and `Shift + B` for moving to the next or previous slices respectively.

When you have finished annotating your slices, export them as segmentation masks (`Content > Export data`; use `grayscale descending` pixel order and order by `z-value`) and put them in an `annotations` folder under the `project root directory` (i.e., `project_root/annotations/sample/segmentation.png`).

### 5. Creating Training Data and Splits:

With your annotations present (double-checking that the `Annotation Progress` column is green, or correct), specify the annotation labels as a comma-separated list in decreasing pixel-value order. Alternatively, if exporting from _Hasty_, use the button to the right of the labels text box to specify the `.json` file included in the _Hasty_ export.

Then, press the `Add Annotations` button to add your annotations to the original tomograms, which will serve as the input for CryoViT. These are saved in a `tomo_annot` folder in your `Data` directory.

After adding annotations, create training splits using the `Generate Training Splits` button (this needs to be done per sample). If this is the first sample, create an empty `splits.csv` file in the `csv` directory, and select this file when asked for a "splits file". 

After the training data and splits.csv file have been created, the dataset is now setup for CryoViT training and evaluation. The entire `Data` folder can then be sent to the cluster to use cluster gpus for training.

### 6. Training and Evaluation

For new datasets, add any new samples created to the `Sample` enum in `CryoVIT/cryovit/config.py`.
Additionally, change the entries under `exp_paths` in `CryoVIT/cryovit/configs/{dino_features.yaml/train_model.yaml/eval_model.yaml}` to the corresponding directories.

Then, from the `CryoViIT` directory with `cryovit_env` activated, extract DINOv2 features with `python -m cryovit.dino_features sample=[your_sample]`.

For just one sample, run training with `python -m cryovit.train_model model=cryovit label_key=[label_to_segment_for] exp_name=[unique_identifier_for_your_experiment] dataset=single dataset.sample=[your_sample]`.
For multiple samples, run training with `python -m cryovit.train_model model=cryovit label_key=[label_to_segment_for] exp_name=[unique_identifier_for_your_experiment] dataset=multi datset.sample=[your_sample_1,your_sample_2,...]`.
> **__Training a different model__**: Replace `model=cryovit` with whatever model you want to train. Currently, only a 3D U-Net model is available (`model=unet3d`).

When the model is trained on some data, the rest of the slices for that data can be segmented using the trained model in evaluation mode:
`python -m cryovit.eval_model model=cryovit label_key=[label_to_segment_for] exp_name=[same_exp_name_as_training] dataset=[same_as_training] dataset.sample=[same_as_training]`

The results should be in the `exp_dir` you've specified in the `.yaml` file.
> **__Future Notes__**: This experiment directory setup is currently being re-worked. When this refactor is finalized, the CryoViT codebase will be updated accordingly.

## Tools

The `Tools` menu provides basic utility functions for ease-of-use.

`Tools > Format Downloaded Data` is for automatically detecting tomograms from downloaded datasets (e.g., CZI), and moves them into the directory structure required by CryoViT.

## Settings

The application settings control application-wide defaults, including which directory file selection dialogs start from, and how many slices to annotate per training tomogram.

Settings can be changed through `File > Settings` in the menu bar. Like the main tabs, settings can be edited by _double-clicking_ on the value. File or directory settings also support selection through a file/folder selection dialog by clicking again on the text box, or edited manually by double-clicking.

Pressing `Ok` on the Settings window will save your current settings, and will be loaded the next time you open the application (unlike tab configurations). Pressing `Cancel` will revert all settings to how they were before.

### Presets

Settings can also be saved and loaded from **presets**. These are also accessed under the `File` menu.

`File > Load` loads settings from an existing preset. If no presets have been saved before, a warning will be shown. This is also accessible through the `Ctrl + O` keyboard shortcut.

`File > Save` saves the current settings to the currently selected preset. If there is no currently selected preset, a warning will be shown. Presets can either be selected using `File > Load` or manually by editing the settings. This is also accessible through the `Ctrl + S` keyboard shortcut.

`File > Save As` saves the current settings to a new preset, also accessible through the `Ctrl + Shift + S` keyboard shortcut. From the dialog window, add a new preset by typing the preset name into the text box and clicking the `Add` button (or pressing `Enter` or `Return`). Select the preset to save to using the dropdown.

When saving or loading settings, presets can be removed by typing the preset name into the text box and clicking the `Remove` button.
