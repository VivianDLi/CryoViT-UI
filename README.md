# CryoVIT: Efficient Segmentation of Cryo-electron Tomograms with Vision Foundation Models

## Installation Instructions

Distribution-specific installers can be found in `releases`.

### Building from Source Code

The CryoViT GUI uses `mamba` to manage python packages and dependencies and can be downloaded [here](https://github.com/conda-forge/miniforge). You should also be able to use `conda` instead of Mamba but setting up the environment may take an unreasonably long time.

To setup the environment:

1. Clone this github repository: `git clone https://github.com/VivianDLi/CryoViT-UI.git`
2. Go into the project directory: `cd CryoViT-UI`
3. Build the application: `make install`

Or, 3. Setup the mamba environment: `mamba env create -f env.yml` 4. Activate the mamba environment: `mamba activate cryovit_gui_env` 5. Launch the application with python: `python cryovit_gui/app.py`

## Usage Guide

This application is meant to be used in conjunction with the [CryoViT source code](https://github.com/sanketx/CryoVIT), and as such, does not support model training or inference out of the box. Instead, the main use is to help format the commands to train and run CryoViT models, and to facilitate the annotation process for training CryoViT.

The application is split into 5 main tabs, and some additional utility functions found in the menu bar.

### Pre-processing

The **pre-processing** tab is used to generate the CLI command to run pre-processing using the CryoViT python library. Pre-processing configuration settings can be edited by _double-clicking_ on config values in the main window.

When finished, the CLI command to run can be generated using the `Generate Command` button at the bottom. This will copy the command to your clipboard to paste into a terminal with `Ctrl + V`. Additionally, the command will be printed in the console log at the bottom.

Model performance relies heavily on the input data having the same pre-processing steps as the training data, so pre-processing your data is important.

### Annotations

The **annotations** tab is used to setup training data for model training. This includes:

- selecting a subset of slices to annotate per tomogram
- exporting those slices as `.png` files to annotate in `Hasty.ai` or some other annotation software
- importing annotated segmentation masks back into tomogram files
- creating a `.csv` file with information about the annotated slices and the tomogram for training
- optionally, generating splits for cross-validation

1. Select the `project root directory`. This is the parent folder that contains all of the training data. This can either be manually typed into the text box, or selected from a folder selection dialog with the button on the right.

This root directory must have a specific structure, containing a `tomograms` folder with the raw or pre-processed tomogram files organized into **sample** folders, a `csv` folder, where `.csv` files will be saved, and a `slices` folder, where exported slices will be saved for annotation. The application will also provide warnings if this directory structure is not met.

From there, all sample sub-directories in the `tomograms` folder will be read, and their progress displayed (i.e., how many tomograms have had slices extracted, or annotations found).

2. Select a sample to produce annotations for. All tomogram files for that sample will then be shown in the section below. From there, you can select specific tomograms to select slices for, or use the `Select All`, `Deselect All`, or `Reset to Default` buttons.

In general, **green** means that the tomogram file or sample has been fully completed (i.e., slices extracted and annotations present in an `annotations` folder).

3. Press the `Launch ChimeraX` button to open ChimeraX to extract slices. This requires the path to your ChimeraX executable to be set in the application [settings](#settings).

When launching ChimeraX, a command will be saved to your clipboard. When ChimeraX successfully launches, paste and run that command in the ChimeraX command line. This will open a tomogram to select z-limits and slices for.

When the tomogram loads, select plane markers (`Markers > Plane`), and under `Marker Placement` on the right, select `Options`, and under `Marker set`, select `zlimits`. Then, navigating between tomogram slices, place **two** markers (_right-click_) on the slices where you can first see or first stop seeing objects of interest in the tomogram. If objects are always visible in all slices, set the markers on the first and last slices.

Then, switch to the `slices` marker set, and place **n** markers on the slices you want to export for annotation (where **n** is set in the application settings).

When finished, type `next` in the ChimeraX command line to save your z-limits and slices, and open the next tomogram. If the number of markers for either marker set is wrong, a warning will be shown.

At any point, if you want to stop, simply close ChimeraX. Progress is saved whenever you type `next`. When all selected tomograms have been marked, you can close ChimeraX.

4. Press the `Extract Slices` button to export slices selected with ChimeraX as `.png` for annotation elsewhere (e.g., `Hasty.ai`). If not all tomograms in the sample have been processed, a warning will be shown, and missing tomograms can be easily found by looking in the `Slices Exported?` column.

When you have finished annotating your slices, export them as segmentation masks and put them in an `annotations` folder under the `project root directory` (i.e., `project_root/annotations/sample/segmentation.png`).

These segmentation masks should be grayscale and have a 0-value for non-labeled pixels, and values starting from 254 and decreasing per label. So, for a slice labeled with mitochondria and cristae, pixels equal to 254 would be mitochondria, and pixels equal to 253 would be cristae. If using `Hasty.ai`, this is automatically done for you by exporting annotations in `decreasing` mode.

5. After inputting the annotation labels as a comma-separated list in decreasing order, press the `Add Annotations` button to add the segmentation masks to the original tomogram files.

Optionally, if exporting from `Hasty.ai`, a `.json` file will be included in the annotations directory containing the annotation labels. Using the button on the right of the labels text box, you can import labels from this file.

6. When all samples have been processed (i.e., all highlighted with green), press `Generate Training Splits` to generate cross-validation training and evaluation splits for CryoViT.

### Model Training

### Model Evaluation

### Model Inference

### Tools

The `Tools` menu provides basic utility functions for ease-of-use.

`Tools > Format Downloaded Data` is for automatically detecting tomograms from downloaded datasets (e.g., CZI), and moves them into the directory structure required by CryoViT.

### Settings

The application settings control application-wide defaults, including which directory file selection dialogs start from, and how many slices to annotate per training tomogram.

Settings can be changed through `File > Settings` in the menu bar. Like the main tabs, settings can be edited by _double-clicking_ on the value. File or directory settings also support selection through a file/folder selection dialog by clicking again on the text box, or edited manually by double-clicking.

Pressing `Ok` on the Settings window will save your current settings, and will be loaded the next time you open the application (unlike tab configurations). Pressing `Cancel` will revert all settings to how they were before.

#### Presets

Settings can also be saved and loaded from **presets**. These are also accessed under the `File` menu.

`File > Load` loads settings from an existing preset. If no presets have been saved before, a warning will be shown. This is also accessible through the `Ctrl + O` keyboard shortcut.

`File > Save` saves the current settings to the currently selected preset. If there is no currently selected preset, a warning will be shown. Presets can either be selected using `File > Load` or manually by editing the settings. This is also accessible through the `Ctrl + S` keyboard shortcut.

`File > Save As` saves the current settings to a new preset, also accessible through the `Ctrl + Shift + S` keyboard shortcut. From the dialog window, add a new preset by typing the preset name into the text box and clicking the `Add` button (or pressing `Enter` or `Return`). Select the preset to save to using the dropdown.

When saving or loading settings, presets can be removed by typing the preset name into the text box and clicking the `Remove` button.
