"""Script to select tomogram slices to label and set-up min and max z-values for annotation and export slices as pngs in ChimeraX."""

src_path = None
dst_path = None
csv_path = None
slice_n = 5
files = None
file_n = 0
csv_data = []
zlimit_markers = None
slice_markers = None


def open_next_tomogram(session):
    from chimerax.markers import MarkerSet

    global zlimit_markers, slice_markers

    session.logger.debug(f"Opening {files[file_n]}")
    models, _ = session.open_command.open_data(str(files[file_n]))
    # Create marker sets
    zlimit_markers = MarkerSet(session, name="zlimits")
    slice_markers = MarkerSet(session, name="slices")
    session.models.add([models[0], zlimit_markers, slice_markers])


def save_slices(session, filename, slices):
    import os
    from PIL import Image
    import numpy as np
    from chimerax.map import Volume

    global dst_path

    # Create destination directory if it doesn't exist
    os.makedirs(dst_path, exist_ok=True)

    # Get tomogram data from Volume model
    data = None
    for model in session.models:
        if isinstance(model, Volume):
            data = model.data.read_matrix(
                (0, 0, 0), model.data.size, (1, 1, 1), False
            )
    # Check data was found
    if data is None:
        session.logger.error(f"No tomogram data found to save.")
        return

    # Save slices as image
    for idx in slices:
        out_path = os.path.join(dst_path, f"{filename[:-4]}_{idx}.png")
        img = data[idx]
        # Normalize and convert to uint8
        img = ((img + 1) * 0.5 * 255 / np.max(img)).astype("uint8")
        img = Image.fromarray(img)
        img.save(out_path)


def set_tomogram_slices(
    session,
    src_dir: str,
    dst_dir: str = None,
    csv_file: str = None,
    num_slices: int = 5,
):
    from pathlib import Path

    global src_path, dst_path, csv_path, slice_n, files, file_n, csv_data

    src_path = Path(src_dir).resolve()
    # Check for .png dst folder
    if dst_dir:
        dst_path = Path(dst_dir).resolve()
    else:
        dst_path = src_path.parent.resolve() / "slices"
    # Check for .csv file
    if csv_file:
        csv_path = Path(csv_file).resolve()
    else:
        csv_path = src_path.parent.resolve() / "slices.csv"
    slice_n = num_slices
    files = list(
        p.resolve()
        for p in src_path.glob("*")
        if p.suffix in {".rec", ".mrc", ".hdf"}
    )
    file_n = 0
    csv_data = []

    # Preview the number of valid tomogram files
    session.logger.info(
        f"Found {len(list(files))} tomogram files in {src_path}."
    )
    # Start looping through all tomogram files
    open_next_tomogram(session)


def next_tomogram(session):
    from chimerax.std_commands.close import close

    global dst_path, slice_n, files, file_n, csv_data

    # Check that labelling has started
    if not src_path:
        session.logger.error(
            "Please start labelling tomograms first using 'start slice labels'."
        )
        return

    # Check if markers are placed
    if zlimit_markers and len(zlimit_markers.atoms) != 2:
        session.logger.error(
            f"Please place two markers for z-limits. Currently there are {len(zlimit_markers.atoms)} markers."
        )
        return
    if slice_markers and len(slice_markers.atoms) != slice_n:
        session.logger.error(
            f"Please place {slice_n} markers for slices. Currently there are {len(slice_markers.atoms)} markers."
        )
        return
    # Save data to .csv file
    zlimits = sorted([round(a.coord[2]) for a in zlimit_markers.atoms])
    slices = sorted([round(a.coord[2]) for a in slice_markers.atoms])
    csv_data.append([str(files[file_n].name), zlimits[0], zlimits[1]] + slices)

    # Save tomogram slices to .png file
    save_slices(session, str(files[file_n].name), slices)

    # Close tomogram
    close(session)
    # Open next tomogram if available
    file_n += 1
    if file_n < len(files):
        open_next_tomogram(session)
    else:
        import csv

        # Save csv data to file
        with open(src_path / "slices.csv", "w", newline="") as f:
            writer = csv.writer(f)
            header = ["tomo_name", "z_min", "z_max"] + [
                f"slice_{i}" for i in range(slice_n)
            ]
            writer.writerow(header)
            writer.writerows(csv_data)
        session.logger.info(
            f"Saved {len(csv_data)} tomograms to {src_path / 'slices.csv'}."
        )


def register_commands(logger):
    from chimerax.core.commands import (
        CmdDesc,
        register,
        OpenFolderNameArg,
        SaveFolderNameArg,
        SaveFileNameArg,
        IntArg,
    )

    slices_desc = CmdDesc(
        required=[("src_dir", OpenFolderNameArg)],
        keyword=[
            ("dst_dir", SaveFolderNameArg),
            ("csv_file", SaveFileNameArg),
            ("num_slices", IntArg),
        ],
        synopsis="Start setting the z-limits and slice numbers to label for a tomogram sample.",
    )
    next_desc = CmdDesc(
        required=[],
        keyword=[],
        synopsis="Finish processing current tomogram, save as an image, and open the next tomogram to label.",
    )
    register(
        "start slice labels", slices_desc, set_tomogram_slices, logger=logger
    )
    register("next", next_desc, next_tomogram, logger=logger)


register_commands(session.logger)
