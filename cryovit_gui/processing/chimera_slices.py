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
    from chimerax.markers import MarkerSet  # type: ignore
    from chimerax.core.colors import Color  # type: ignore

    global zlimit_markers, slice_markers

    session.logger.info(f"Opening {files[file_n]}")
    try:
        models, _ = session.open_command.open_data(str(files[file_n]))
        # Create marker sets
        zlimit_markers = MarkerSet(session, name="zlimits")
        zlimit_markers.set_color(Color((1, 0, 0)))
        slice_markers = MarkerSet(session, name="slices")
        slice_markers.set_color(Color((0, 1, 0)))
        session.models.add(models)
        session.models.add([zlimit_markers, slice_markers])
    except Exception as e:
        session.logger.error(f"Error opening tomogram {files[file_n]}: {e}")
        return


def save_slices(session, filename, slices):
    import os
    from PIL import Image
    import numpy as np
    from chimerax.map import Volume  # type: ignore

    global dst_path

    try:
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
    except Exception as e:
        session.logger.error(f"Error saving slices for {filename}: {e}")
        return


def set_tomogram_slices(
    session,
    src_dir: str,
    sample: str = None,
    dst_dir: str = None,
    csv_dir: str = None,
    num_slices: int = 5,
):
    import os
    from pathlib import Path

    global src_path, dst_path, csv_path, slice_n, files, file_n, csv_data

    # Check for .png dst folder
    if dst_dir:
        dst_path = Path(dst_dir).resolve()
    else:
        dst_path = Path(src_dir).parent.resolve() / "slices"

    if sample:
        src_path = Path(src_dir).resolve() / sample
    else:
        src_path = Path(src_dir).resolve()
        sample = src_path.name
    dst_path = dst_path / sample
    # Create destination directory if it doesn't exist
    os.makedirs(dst_path, exist_ok=True)
    # Check for .csv file
    if csv_dir:
        csv_path = Path(csv_dir).resolve() / f"{sample}.csv"
    else:
        csv_path = Path(src_dir).parent.resolve() / "csv" / f"{sample}.csv"
    # Create csv directory if it doesn't exist
    os.makedirs(csv_path.parent, exist_ok=True)
    slice_n = num_slices
    files = list(
        p.resolve() for p in src_path.glob("*") if p.suffix in [".rec", ".mrc", ".hdf"]
    )
    file_n = 0
    csv_data = []

    # Preview the number of valid tomogram files
    session.logger.info(f"Found {len(list(files))} tomogram files for {sample}.")
    # Start looping through all tomogram files
    open_next_tomogram(session)


def next_tomogram(session):
    from chimerax.std_commands.close import close  # type: ignore

    global csv_path, slice_n, files, file_n, csv_data

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
    # Adjust for end exclusion
    zlimits[1] += 1
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
        with open(csv_path, "w+", newline="") as f:
            writer = csv.writer(f)
            header = ["tomo_name", "z_min", "z_max"] + [
                f"slice_{i}" for i in range(slice_n)
            ]
            writer.writerow(header)
            writer.writerows(csv_data)
        session.logger.info(f"Saved {len(csv_data)} tomograms to {csv_path}.")


def register_commands(logger):
    from chimerax.core.commands import (  # type: ignore
        CmdDesc,
        register,
        ListOf,
        OpenFolderNameArg,
        OpenFileNameArg,
        SaveFolderNameArg,
        SaveFileNameArg,
        StringArg,
        IntArg,
    )

    slices_desc = CmdDesc(
        required=[("src_dir", OpenFolderNameArg), ("sample", StringArg)],
        keyword=[
            ("num_slices", IntArg),
            ("tomograms", ListOf(OpenFileNameArg)),
            ("dst_dir", SaveFolderNameArg),
            ("csv_dir", SaveFileNameArg),
        ],
        synopsis="Start setting the z-limits and slice numbers to label for a tomogram sample.",
    )
    next_desc = CmdDesc(
        required=[],
        keyword=[],
        synopsis="Finish processing current tomogram, add entry to .csv, and open the next tomogram to label.",
    )
    register("start slice labels", slices_desc, set_tomogram_slices, logger=logger)
    register("next", next_desc, next_tomogram, logger=logger)


session.logger.info(  # type: ignore
    """
Start labelling tomograms by running 'start slice labels'.
    This expects as arguments a tomogram directory and, optionally, a sample name. By default, the slices will be saved in a folder called 'slices/sample' in the parent directory of the tomogram directory, and a .csv file will be saved in a folder called 'csv/sample' in that same parent directory.
    
    Optional keyword arguments are: 'dst_dir' for specifying the slice directory, 'csv_dir' for specifying the csv directory, and 'num_slices' for specifying the number of slices to label (default is 5).

    For example: 'start slice labels "./Raw Tomograms" Q66' will label 5 slices of the tomograms in the directory "./Raw Tomograms/Q66" and save them in the directory "./slices/Q66" and the csv file in "./csv/Q66".
    
After running 'start slice labels', use plane markers to select slices and z-limits, and run 'next' to save the slices and open the next tomogram.
"""
)
register_commands(session.logger)  # type: ignore
